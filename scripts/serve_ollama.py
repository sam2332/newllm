"""Serve a checkpoint as an Ollama server.

Standard Ollama semantics, so the official ``ollama`` Python package, the
``ollama`` CLI (``OLLAMA_HOST=http://127.0.0.1:11500 ollama run newllm-agent``)
and OpenAI-style clients (``/v1/chat/completions``) all work unmodified:

  * the CLIENT defines tools (``tools`` in the request) and executes them;
  * one ``/api/chat`` call produces exactly one assistant turn - a
    ``tool_calls`` message or a final ``content`` - never a server-side loop;
  * ``role: "system"`` messages are placed after the tool-schema block, so a
    persona never displaces the schema the model reads tool names from;
  * ``stream: true`` returns NDJSON with content/thinking deltas.

Run:
    .venv/bin/python scripts/serve_ollama.py \
        --checkpoint checkpoints_long_M/agent_best.pt --port 11500

``--builtin-tools`` restores the old behaviour (the server runs its own
toolbox and returns a finished answer) as a non-standard mode.
"""

import argparse
import collections
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/lmeadows/llm")
import torch

from agent.agent_loop import run_agent, EOT
from agent.chat import load_checkpoint, load_tokenizer_for
from agent.ollama_context import (assemble_context, from_model_text,
                                  system_texts_from_messages)
from agent.ollama_format import (assistant_to_ollama, canonical_args,
                                 tool_call_id, toolbox_to_ollama_tools)
from agent.repo_tools import attach_repo_tools
from agent.tool_schema import schema_block_from_toolbox
from agent.tools import Toolbox
from agent.turn import (generate_turn, make_grammar, sampler_from_options,
                        stream_turn)

VERSION = "0.0.0-newllm"


class ServerState:
    def __init__(self, checkpoint, device, model_name, num_ctx, max_new,
                 constrained, builtin_tools, repo_tools, system_mode,
                 log_context, debug_endpoints, accept_any_model):
        t0 = time.perf_counter_ns()
        self.model = load_checkpoint(checkpoint, device=device)
        self.load_ns = time.perf_counter_ns() - t0
        self.tokenizer = load_tokenizer_for(self.model)
        self.device = device
        self.name = model_name
        self.num_ctx = num_ctx
        self.max_new = max_new
        self.constrained = constrained
        self.builtin_tools = builtin_tools
        self.system_mode = system_mode
        self.log_context = log_context
        self.debug_endpoints = debug_endpoints
        self.accept_any_model = accept_any_model
        self.lock = threading.Lock()
        # (name, canonical args) -> thought, so a client that strips
        # ``thinking`` from echoed messages still yields a faithful context.
        self.thought_cache = collections.OrderedDict()
        # tool_call id -> name, for OpenAI tool messages that carry only an id.
        self.call_names = collections.OrderedDict()
        self.toolbox = None
        if builtin_tools:
            tb = Toolbox()
            if repo_tools:
                tb = attach_repo_tools(tb)
            self.toolbox = tb
        st = os.stat(checkpoint)
        self.checkpoint = checkpoint
        self.size = st.st_size
        self.modified_at = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
        self.digest = hashlib.sha256(f"{checkpoint}:{st.st_mtime}".encode()).hexdigest()
        self.params = self.model.count_parameters()

    def remember(self, mapping, key, value, limit=256):
        mapping[key] = value
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def known_model(self, name) -> bool:
        if self.accept_any_model or not name:
            return True
        return name in (self.name, f"{self.name}:latest")

    def details(self):
        return {"parent_model": "", "format": "pt", "family": "newllm",
                "families": ["newllm"], "parameter_size": f"{self.params/1e6:.1f}M",
                "quantization_level": "F32"}

    def model_entry(self):
        return {"name": f"{self.name}:latest", "model": f"{self.name}:latest",
                "modified_at": self.modified_at, "size": self.size,
                "digest": self.digest, "details": self.details()}


def _now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------- /api/chat

def _prepare(state, payload):
    """Everything before generation: context, sampler, limits, grammar."""
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    options = payload.get("options") or {}
    num_ctx = int(options.get("num_ctx") or state.num_ctx)
    num_predict = int(options.get("num_predict") or 0)
    max_new = num_predict if num_predict > 0 else state.max_new
    reserve = min(max_new, max(256, num_ctx // 4))
    context, tokens, info = assemble_context(
        messages, tools, num_ctx=num_ctx, reserve=reserve,
        tokenizer=state.tokenizer, system_mode=state.system_mode,
        thought_cache=state.thought_cache)
    allowed = [t.get("function", t).get("name") for t in tools
               if isinstance(t, dict)] if tools else None
    allowed = [n for n in allowed if n] if allowed else None
    stop = [s for s in (options.get("stop") or []) if isinstance(s, str)]
    stop_texts = tuple([EOT] + [s.encode("utf-8").decode("latin-1") for s in stop])
    if state.log_context:
        print(f"--- context ({info['prompt_tokens']} tokens, "
              f"dropped {info['dropped_turns']} turns) ---\n{context}---",
              flush=True)
    return dict(context=context, tokens=tokens, info=info, allowed=allowed,
                sampler=sampler_from_options(options), max_new=max_new,
                stop_texts=stop_texts, seed=options.get("seed"),
                grammar=make_grammar(state.tokenizer, allowed, state.constrained))


def _envelope(state, payload, message, result=None, done=True, done_reason=None):
    env = {"model": payload.get("model") or state.name, "created_at": _now(),
           "message": message, "done": done}
    if done:
        env["done_reason"] = done_reason or (result.done_reason if result else "stop")
        if result is not None:
            env.update({
                "total_duration": result.prompt_eval_ns + result.eval_ns,
                "load_duration": state.load_ns,
                "prompt_eval_count": result.prompt_tokens,
                "prompt_eval_duration": result.prompt_eval_ns,
                "eval_count": result.eval_tokens,
                "eval_duration": result.eval_ns,
            })
    return env


def _message_from_result(state, result):
    """The Ollama assistant message for a finished turn."""
    if result.kind == "tool_call":
        name = result.tool_call["name"]
        args = result.tool_call["arguments"]
        state.remember(state.thought_cache, (name, canonical_args(args)),
                       result.thought)
        state.remember(state.call_names, tool_call_id(name, args), name)
        msg = assistant_to_ollama({"thought": result.thought,
                                   "tool_call": result.tool_call})
        return msg
    msg = {"role": "assistant", "content": result.response}
    if result.thought:
        msg["thinking"] = result.thought
    return msg


def _diagnostic(result):
    return {"newllm": {"fallback": result.fallback}} if result.fallback else {}


def chat_once(state, payload) -> dict:
    p = _prepare(state, payload)
    with state.lock:
        result = generate_turn(state.model, p["tokens"], sampler=p["sampler"],
                               tokenizer=state.tokenizer, device=state.device,
                               max_new=p["max_new"], stop_texts=p["stop_texts"],
                               grammar=p["grammar"], seed=p["seed"],
                               allowed_names=p["allowed"])
    env = _envelope(state, payload, _message_from_result(state, result), result)
    env.update(_diagnostic(result))
    return env


def chat_stream(state, payload):
    """Yield NDJSON envelopes for one turn."""
    p = _prepare(state, payload)
    model_name = payload.get("model") or state.name
    with state.lock:
        for ev in stream_turn(state.model, p["tokens"], sampler=p["sampler"],
                              tokenizer=state.tokenizer, device=state.device,
                              max_new=p["max_new"], stop_texts=p["stop_texts"],
                              grammar=p["grammar"], seed=p["seed"],
                              allowed_names=p["allowed"]):
            if ev.type == "content":
                yield {"model": model_name, "created_at": _now(), "done": False,
                       "message": {"role": "assistant", "content": ev.text}}
            elif ev.type == "thinking":
                yield {"model": model_name, "created_at": _now(), "done": False,
                       "message": {"role": "assistant", "content": "",
                                   "thinking": ev.text}}
            elif ev.type == "tool_call":
                yield {"model": model_name, "created_at": _now(), "done": False,
                       "message": _message_from_result(state, ev.result)}
            elif ev.type == "done":
                r = ev.result
                if r.kind != "tool_call":
                    _message_from_result(state, r)  # for cache side effects only
                env = _envelope(state, payload,
                                {"role": "assistant", "content": ""}, r)
                env.update(_diagnostic(r))
                yield env


def chat_builtin(state, payload) -> dict:
    """Old behaviour: the server runs its own toolbox to a finished answer."""
    messages = payload.get("messages") or []
    question = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")
    system = schema_block_from_toolbox(state.toolbox)
    free = "\n".join(system_texts_from_messages(messages))
    if free and state.system_mode == "stacked":
        system += f"\n<system>{free}</system>"
    with state.lock:
        result = run_agent(state.model, question, state.toolbox,
                           device=state.device, greedy=True,
                           tokenizer=state.tokenizer, system=system,
                           max_steps=64, max_new=state.max_new)
    message = {"role": "assistant",
               "content": from_model_text(result.get("final_answer", "")),
               "thinking": result.get("thinking", "")}
    env = {"model": payload.get("model") or state.name, "created_at": _now(),
           "message": message, "done": True, "done_reason": "stop",
           "tool_activity": [{"function": {"name": s["tool"],
                                           "arguments": json.loads(s["arg"])},
                              "result": s["result"]}
                             for s in result.get("steps", [])]}
    return env


# --------------------------------------------------------------- /v1 (OpenAI)

def _openai_messages(state, messages):
    """OpenAI message shapes -> Ollama shapes the context builder reads."""
    out = []
    for m in messages or []:
        m = dict(m)
        if m.get("role") == "tool" and not m.get("tool_name"):
            m["tool_name"] = state.call_names.get(m.get("tool_call_id"), "")
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", {}))
                calls.append({"id": c.get("id"), "function": fn})
            m["tool_calls"] = calls
        if isinstance(m.get("content"), list):          # multipart text
            m["content"] = "".join(part.get("text", "") for part in m["content"]
                                   if isinstance(part, dict))
        out.append(m)
    return out


def _openai_tool_calls(message):
    return [{"id": c.get("id") or tool_call_id(c["function"]["name"],
                                                c["function"]["arguments"]),
             "type": "function",
             "function": {"name": c["function"]["name"],
                          "arguments": json.dumps(c["function"]["arguments"])}}
            for c in message.get("tool_calls", [])]


def openai_once(state, payload) -> dict:
    ollama_payload = {"model": payload.get("model"), "tools": payload.get("tools"),
                      "messages": _openai_messages(state, payload.get("messages")),
                      "options": {k: payload[k] for k in ("temperature", "top_p",
                                                         "seed", "stop")
                                  if k in payload}}
    if payload.get("max_tokens"):
        ollama_payload["options"]["num_predict"] = int(payload["max_tokens"])
    env = chat_once(state, ollama_payload)
    msg = env["message"]
    message = {"role": "assistant", "content": msg.get("content") or None}
    finish = "stop"
    if msg.get("tool_calls"):
        message["tool_calls"] = _openai_tool_calls(msg)
        message["content"] = None
        finish = "tool_calls"
    elif env.get("done_reason") == "length":
        finish = "length"
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": env["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": env.get("prompt_eval_count", 0),
                      "completion_tokens": env.get("eval_count", 0),
                      "total_tokens": env.get("prompt_eval_count", 0)
                      + env.get("eval_count", 0)}}


def openai_stream(state, payload):
    ollama_payload = {"model": payload.get("model"), "tools": payload.get("tools"),
                      "messages": _openai_messages(state, payload.get("messages")),
                      "options": {k: payload[k] for k in ("temperature", "top_p",
                                                         "seed", "stop")
                                  if k in payload}}
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": payload.get("model") or state.name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    finish = "stop"
    for env in chat_stream(state, ollama_payload):
        msg = env.get("message", {})
        if env.get("done"):
            if env.get("done_reason") == "length":
                finish = "length"
            continue
        if msg.get("tool_calls"):
            finish = "tool_calls"
            calls = _openai_tool_calls(msg)
            for i, c in enumerate(calls):
                c["index"] = i
            yield chunk({"tool_calls": calls})
        elif msg.get("content"):
            yield chunk({"content": msg["content"]})
    yield chunk({}, finish)


# ------------------------------------------------------------- /api/generate

def generate_payload_to_chat(payload):
    messages = []
    if payload.get("system"):
        messages.append({"role": "system", "content": payload["system"]})
    messages.append({"role": "user", "content": payload.get("prompt", "")})
    return {"model": payload.get("model"), "messages": messages,
            "options": payload.get("options") or {}}


# -------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: ServerState = None

    def log_message(self, *_args):
        pass

    # -- helpers -----------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text, code=200):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _stream(self, lines, content_type="application/x-ndjson", sse=False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for obj in lines:
                data = json.dumps(obj)
                data = (f"data: {data}\n\n" if sse else data + "\n").encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            if sse:
                tail = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(tail):X}\r\n".encode() + tail + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw)

    def _check_model(self, payload):
        name = payload.get("model") or payload.get("name")
        if not self.state.known_model(name):
            self._json({"error": f"model '{name}' not found"}, 404)
            return False
        return True

    # -- routes ------------------------------------------------------------
    def do_HEAD(self):
        if self.path in ("/", ""):
            return self._text("Ollama is running")
        self._json({"error": "not found"}, 404)

    def do_GET(self):
        s = self.state
        path = self.path.split("?")[0]
        if path in ("/", ""):
            return self._text("Ollama is running")
        if path == "/api/version":
            return self._json({"version": VERSION})
        if path == "/api/tags":
            return self._json({"models": [s.model_entry()]})
        if path == "/api/ps":
            entry = s.model_entry()
            entry.update({"size_vram": s.params * 4,
                          "expires_at": (datetime.now(timezone.utc)
                                         + timedelta(hours=1)).isoformat()})
            return self._json({"models": [entry]})
        if path == "/v1/models":
            return self._json({"object": "list", "data": [
                {"id": s.name, "object": "model", "created": 0,
                 "owned_by": "newllm"}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        s = self.state
        path = self.path.split("?")[0]
        try:
            payload = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)
        try:
            if path == "/api/chat":
                if not self._check_model(payload):
                    return
                if not payload.get("messages"):
                    return self._json({"model": payload.get("model") or s.name,
                                       "created_at": _now(),
                                       "message": {"role": "assistant", "content": ""},
                                       "done": True, "done_reason": "load"})
                if s.builtin_tools:
                    return self._json(chat_builtin(s, payload))
                if payload.get("stream", True):
                    return self._stream(chat_stream(s, payload))
                return self._json(chat_once(s, payload))
            if path == "/api/generate":
                if not self._check_model(payload):
                    return
                if not payload.get("prompt"):
                    return self._json({"model": payload.get("model") or s.name,
                                       "created_at": _now(), "response": "",
                                       "done": True, "done_reason": "load"})
                chat = generate_payload_to_chat(payload)
                if payload.get("stream", True):
                    def lines():
                        for env in chat_stream(s, chat):
                            msg = env.pop("message", {})
                            env["response"] = msg.get("content", "")
                            if msg.get("thinking"):
                                env["thinking"] = msg["thinking"]
                            yield env
                    return self._stream(lines())
                env = chat_once(s, chat)
                msg = env.pop("message", {})
                env["response"] = msg.get("content", "")
                if msg.get("thinking"):
                    env["thinking"] = msg["thinking"]
                return self._json(env)
            if path == "/api/show":
                if not self._check_model(payload):
                    return
                return self._json({
                    "modelfile": f"# newllm\nFROM {s.checkpoint}\n",
                    "parameters": f"num_ctx {s.num_ctx}",
                    "template": "{{ .Prompt }}", "license": "",
                    "details": s.details(),
                    "model_info": {"general.architecture": "newllm",
                                   "general.parameter_count": s.params,
                                   "newllm.context_length": s.num_ctx},
                    "capabilities": ["completion", "tools"],
                    "modified_at": s.modified_at})
            if path == "/v1/chat/completions":
                if not self._check_model(payload):
                    return
                if payload.get("stream"):
                    return self._stream(openai_stream(s, payload),
                                        content_type="text/event-stream", sse=True)
                return self._json(openai_once(s, payload))
            if path == "/api/debug/context" and s.debug_endpoints:
                p = _prepare(s, payload)
                return self._json({"context": p["context"], **p["info"],
                                   "allowed_tools": p["allowed"]})
        except Exception as exc:                                  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        self._json({"error": "not found"}, 404)


# -------------------------------------------------------------------- main

def load_state(checkpoint, device="auto", model_name="newllm-agent",
               num_ctx=16384, max_new=4096, constrained=True,
               builtin_tools=False, repo_tools=False, system_mode="stacked",
               log_context=False, debug_endpoints=False,
               accept_any_model=False) -> ServerState:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ServerState(checkpoint, device, model_name, num_ctx, max_new,
                       constrained, builtin_tools, repo_tools, system_mode,
                       log_context, debug_endpoints, accept_any_model)


def make_server(host, port, state) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_long_M/agent_best.pt")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-name", default="newllm-agent")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--no-constrained", dest="constrained", action="store_false",
                    help="disable grammar-constrained decoding. ON by default: it "
                         "makes a tool name absent from the client's schema "
                         "unrepresentable, which on the current checkpoint is the "
                         "difference between the tool loop working and the model "
                         "emitting a memorized name (scripts/test_ollama_client.py "
                         "12/12 vs 11/12)")
    ap.add_argument("--builtin-tools", action="store_true",
                    help="non-standard: run the server's own toolbox to a "
                         "finished answer instead of returning tool_calls")
    ap.add_argument("--repo-tools", action="store_true",
                    help="with --builtin-tools, also expose read-only repo tools")
    ap.add_argument("--system-mode", choices=["stacked", "drop"], default="stacked")
    ap.add_argument("--log-context", action="store_true")
    ap.add_argument("--debug-endpoints", action="store_true")
    ap.add_argument("--accept-any-model", action="store_true")
    args = ap.parse_args()

    state = load_state(args.checkpoint, args.device, args.model_name,
                       args.num_ctx, args.max_new, args.constrained,
                       args.builtin_tools, args.repo_tools, args.system_mode,
                       args.log_context, args.debug_endpoints,
                       args.accept_any_model)
    if state.toolbox is not None:
        print("builtin tools:",
              [t["function"]["name"] for t in toolbox_to_ollama_tools(state.toolbox)])
    print(f"model {state.name} ({state.params:,} params, vocab "
          f"{state.tokenizer.vocab_size}) on {state.device}")
    print(f"Ollama-compatible API on http://{args.host}:{args.port}  "
          f"(num_ctx {state.num_ctx}, max_new {state.max_new})", flush=True)
    make_server(args.host, args.port, state).serve_forever()


if __name__ == "__main__":
    main()
