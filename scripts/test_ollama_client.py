"""End-to-end: the official ``ollama`` client against this server.

Starts ``scripts/serve_ollama.py`` in a thread (or targets ``--host`` for a
running server, including a real Ollama instance in Track C) and drives the
canonical client-executes-tools loop with tools whose names are NOT in the
training name pools, so a passing test proves the name was read from the
schema in context rather than remembered.

Two tools, deliberately different:

* ``evaluate_sum(expr)`` - a new *name* over the argument shape the model was
  trained on (every calc-like tool in training takes ``expr``). This is the
  hard gate.
* ``add_numbers(a, b)`` - a new name AND new parameter names. Training never
  varied parameter names, so this is reported as INFO, not a failure: it
  measures a generalization the current model was never taught.

Structural assertions (every response parses, tool names come from the
schema, ``done_reason`` is set, streams concatenate) are separate from the
correctness assertion, because a 41/100 model can be wrong without the server
being broken. Exit code 1 on any hard failure.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/test_ollama_client.py [--no-constrained]
    .venv/bin/python scripts/test_ollama_client.py --host http://localhost:11434 --model newllm
"""

import argparse
import json
import re
import sys
import threading
import urllib.request

sys.path.insert(0, "/home/lmeadows/llm")

SUM_TOOL = {"type": "function", "function": {
    "name": "evaluate_sum",
    "description": "Evaluate an arithmetic expression such as 12 + 8 and return the number.",
    "parameters": {"type": "object",
                   "properties": {"expr": {"type": "string",
                                           "description": "Arithmetic expression with +, -, *, /."}},
                   "required": ["expr"]}}}

ADD_TOOL = {"type": "function", "function": {
    "name": "add_numbers", "description": "Add two integers and return the sum.",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "integer", "description": "first number"},
                                  "b": {"type": "integer", "description": "second number"}},
                   "required": ["a", "b"]}}}

PHRASINGS = ["What is 12 + 8?", "Calculate 12 plus 8.", "Add 12 and 8."]


def execute(name, args) -> str:
    """The client's side of the loop: run the tool the model asked for."""
    if not isinstance(args, dict):
        return "error: arguments must be an object"
    if "expr" in args:
        expr = str(args["expr"])
        if re.fullmatch(r"[\d\s+\-*/().]+", expr):
            try:
                return str(eval(expr, {"__builtins__": {}}, {}))   # noqa: S307
            except Exception as exc:                            # noqa: BLE001
                return f"error: {exc}"
        return "error: not an arithmetic expression"
    if "a" in args and "b" in args:
        try:
            return str(int(args["a"]) + int(args["b"]))
        except (TypeError, ValueError):
            return "error: a and b must be integers"
    return f"error: unsupported arguments {sorted(args)}"


class Check:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail="", soft=False):
        self.rows.append((name, bool(ok), detail, soft))
        tag = ("INFO" if soft else "PASS") if ok or soft else "FAIL"
        if soft and not ok:
            tag = "INFO"
        print(f"  {tag}  {name}{('  ' + detail) if detail else ''}", flush=True)
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1] and not r[3]]


def start_server(checkpoint, device, constrained):
    from scripts.serve_ollama import load_state, make_server
    state = load_state(checkpoint, device=device, constrained=constrained,
                       debug_endpoints=True)
    server = make_server("127.0.0.1", 0, state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", state


def post(host, path, payload):
    req = urllib.request.Request(host + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, json.loads(r.read())


def run_tool_loop(client, model, question, tools):
    """The canonical loop. Returns (first, final, name, args)."""
    messages = [{"role": "user", "content": question}]
    first = client.chat(model=model, messages=messages, tools=tools,
                        options={"temperature": 0}, stream=False)
    calls = first.message.tool_calls or []
    if not calls:
        return first, first, None, None
    fn = calls[0].function
    messages.append(first.message)
    messages.append({"role": "tool", "tool_name": fn.name,
                     "content": execute(fn.name, fn.arguments)})
    final = client.chat(model=model, messages=messages, tools=tools,
                        options={"temperature": 0}, stream=False)
    return first, final, fn.name, fn.arguments


def loop_over_phrasings(client, model, tools, expected_name):
    """Try each phrasing greedily; stop at the first correct answer."""
    structural, correct, detail = True, False, ""
    for q in PHRASINGS:
        first, final, name, args = run_tool_loop(client, model, q, tools)
        structural &= first.done and final.done
        if name is None:
            detail = f"{q!r} -> no tool call: {(first.message.content or '')[:90]!r}"
            continue
        structural &= name == expected_name
        detail = f"{q!r} -> {name}({args}) -> {final.message.content!r}"
        if "20" in (final.message.content or ""):
            correct = True
            break
    return structural, correct, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="target a running server")
    ap.add_argument("--model", default="newllm-agent")
    ap.add_argument("--checkpoint", default="checkpoints_long_M/agent_best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-constrained", dest="constrained", action="store_false",
                    help="run the server without grammar constraints (expect the "
                         "correctness check to fail on the current checkpoint)")
    args = ap.parse_args()

    import ollama
    check = Check()
    state = None
    host = args.host
    if host is None:
        host, state = start_server(args.checkpoint, args.device, args.constrained)
        print(f"server on {host}  constrained={args.constrained}")
    client = ollama.Client(host=host)

    # 1. listing and show parse through the client's pydantic models
    listing = client.list()
    check("list() parses", listing.models and listing.models[0].model,
          str(listing.models[0].model))
    check("list() has modified_at/size/digest",
          listing.models[0].modified_at is not None and (listing.models[0].size or 0) > 0
          and bool(listing.models[0].digest))
    show = client.show(args.model)
    check("show() advertises tools", "tools" in (show.capabilities or []),
          str(show.capabilities))

    # 2. the tool loop: new name, trained argument shape (hard gate)
    structural, correct, detail = loop_over_phrasings(client, args.model,
                                                      [SUM_TOOL], "evaluate_sum")
    check("tool name read from the client's schema (evaluate_sum)", structural, detail)
    check("final answer correct on at least one phrasing", correct, detail)

    # 3. new name AND new parameter names (not trained; informational)
    structural2, correct2, detail2 = loop_over_phrasings(client, args.model,
                                                         [ADD_TOOL], "add_numbers")
    check("generalizes to new parameter names (add_numbers a,b)",
          structural2 and correct2, detail2, soft=True)

    # 4. streaming: every chunk parses, only the last is done, agrees with non-stream
    q = PHRASINGS[0]
    messages = [{"role": "user", "content": q}]
    chunks = list(client.chat(model=args.model, messages=messages, tools=[SUM_TOOL],
                              options={"temperature": 0}, stream=True))
    check("stream: >=1 chunk and last is done", chunks and chunks[-1].done
          and not any(c.done for c in chunks[:-1]), f"{len(chunks)} chunks")
    tool_chunks = [c for c in chunks if c.message.tool_calls]
    non_stream = client.chat(model=args.model, messages=messages, tools=[SUM_TOOL],
                             options={"temperature": 0}, stream=False)
    if non_stream.message.tool_calls:
        check("stream: tool-call turn is exactly one chunk and matches non-stream",
              len(tool_chunks) == 1 and tool_chunks[0].message.tool_calls[0].function.name
              == non_stream.message.tool_calls[0].function.name)
    else:
        streamed = "".join(c.message.content or "" for c in chunks)
        check("stream and non-stream agree (content)",
              streamed == (non_stream.message.content or ""),
              f"{streamed[:60]!r} vs {(non_stream.message.content or '')[:60]!r}")

    # 5. no tools -> a plain content reply, never tool_calls
    plain = client.chat(model=args.model, options={"temperature": 0}, stream=False,
                        messages=[{"role": "user", "content": "Say hello."}])
    check("no tools -> no tool_calls", not plain.message.tool_calls,
          repr(plain.message.content)[:60])

    # 6. system prompt lands after the schema block (local server only)
    if state is not None:
        _, ctx = post(host, "/api/debug/context",
                      {"model": args.model, "tools": [SUM_TOOL],
                       "messages": [{"role": "system", "content": "You are Zed."},
                                    {"role": "user", "content": q}]})
        c = ctx["context"]
        check("system prompt stacked after schema",
              c.startswith('<system>{"tools":') and
              '</system>\n<system>You are Zed.</system>\n<user>' in c)

    # 7. running out of budget is a 200 with done_reason=length
    short = client.chat(model=args.model, messages=messages, tools=[SUM_TOOL],
                        options={"temperature": 0, "num_predict": 8}, stream=False)
    check("num_predict exhausted -> done_reason length", short.done_reason == "length",
          str(short.done_reason))

    # 8. unknown model is a proper error, not a crash
    try:
        client.chat(model="does-not-exist", messages=messages, stream=False)
        check("unknown model -> ResponseError", False)
    except ollama.ResponseError as exc:
        check("unknown model -> ResponseError", exc.status_code == 404, str(exc)[:60])

    # 9. OpenAI-compatible endpoint (local server only)
    if state is not None:
        status, body = post(host, "/v1/chat/completions",
                            {"model": args.model, "messages": messages,
                             "tools": [SUM_TOOL], "temperature": 0})
        choice = body["choices"][0]
        ok = status == 200 and choice["finish_reason"] in ("tool_calls", "stop")
        if choice["finish_reason"] == "tool_calls":
            fn = choice["message"]["tool_calls"][0]["function"]
            ok &= fn["name"] == "evaluate_sum" and isinstance(fn["arguments"], str)
        check("/v1/chat/completions", ok, choice["finish_reason"])

    print()
    if check.failed:
        print(f"{len(check.failed)} FAILED: {[r[0] for r in check.failed]}")
        sys.exit(1)
    print(f"all {len([r for r in check.rows if not r[3]])} hard checks passed")


if __name__ == "__main__":
    main()
