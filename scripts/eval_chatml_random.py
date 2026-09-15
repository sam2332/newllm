"""Score a ChatML checkpoint on N random held-out traces.

``scripts/eval_random.py`` predates the ChatML retrain and cannot measure
these checkpoints at all: it decodes the cache with the byte tokenizer and
looks for ``<assistant>{json}</assistant>``, so against a BPE/ChatML dataset
every trace comes back unparseable and the score reads 0/0. This is the same
measurement - the same split, the same seed, the same "did it reach the
reference final answer" question - through the ChatML protocol.

The trace is replayed as a real agent loop: the tools named in the trace's
own ``<tools>`` block are bound to the toolbox, the model is asked for a
turn, a tool call is executed for real and its observation appended, and the
loop runs until the model answers or the hop budget is spent. Nothing about
the reference beyond its first user turn is shown to the model.
"""

import argparse
import json
import random
import re
import sys

sys.path.insert(0, "/home/lmeadows/llm")
import torch
from torch.utils.data import random_split

from agent.chat import load_tokenizer_for
from agent.chatml import IM_END, IM_START, render
from agent.dataset_builder import PrebuiltDataset
from agent.repo_tools import attach_repo_tools
from agent.sandbox_tools import attach_sandbox_tools
from agent.tool_schema import NAME_POOLS, PARAM_SCHEMAS
from agent.tools import Toolbox
from agent.turn import generate_turn, sampler_from_options
from agent.virtual_workspace import VirtualWorkspace
from scripts.eval_agent import load

SURFACE_TO_CANONICAL = {s: c for c, names in NAME_POOLS.items() for s in names}
_TOOL_LINE = re.compile(r"^\{.*\}$", re.M)


def tools_from_trace(text: str) -> list:
    """The tool schemas declared in this trace's ChatML system block."""
    m = re.search(r"<tools>\n(.*?)\n</tools>", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        fn = obj.get("function", obj)
        if fn.get("name"):
            out.append({"type": "function", "function": fn})
    return out


def system_text_of(text: str):
    """Any free-text system prompt, i.e. the part before the # Tools header."""
    m = re.search(re.escape(IM_START) + r"system\n(.*?)" + re.escape(IM_END),
                  text, re.S)
    if not m:
        return None
    body = m.group(1)
    head = body.split("# Tools")[0].strip("\n").strip()
    return head or None


def first_user(text: str):
    m = re.search(re.escape(IM_START) + r"user\n(.*?)" + re.escape(IM_END),
                  text, re.S)
    if not m:
        return None
    body = m.group(1)
    return None if body.lstrip().startswith("<tool_response>") else body.strip()


def reference_answer(text: str):
    """The reference answer to the FIRST user question, not the last one.

    These traces are conversations: a knowledge arc asks a dozen unrelated
    questions in turn. Pairing the first question with the last answer scored
    an awk question against an answer about closures - a miss the model could
    not possibly avoid. Only the first exchange is replayed, so only its
    answer is the reference.
    """
    first_user = text.find(IM_START + "user\n")
    if first_user < 0:
        return None
    for m in re.finditer(re.escape(IM_START) + r"assistant\n(.*?)" + re.escape(IM_END),
                         text[first_user:], re.S):
        body = re.sub(r"<think>.*?</think>", "", m.group(1), flags=re.S)
        if "<tool_call>" in body:
            continue                      # a tool call, keep looking
        body = body.strip()
        if body:
            return body
    return None


def _param_map(canonical: str, schema: dict) -> dict:
    """Surface parameter name -> canonical one, positionally.

    ToolSchemaSampler renames parameters as well as tools, walking
    PARAM_SCHEMAS[canonical] in order, so the nth property of the trace's
    schema is the nth canonical argument. A global reverse map cannot do this:
    the pools overlap, and "q" is a surface name for both expr and query.

    Without this every call in the eval reached the tool with the schema's
    argument names and failed - calc got {"e": "53 * 44"} when it wanted
    "expr" and returned a syntax error, so a model that emitted exactly the
    reference tool call still scored zero.
    """
    canon = list(PARAM_SCHEMAS.get(canonical, {}))
    surface = list((schema or {}).get("properties", {}))
    return {s: c for s, c in zip(surface, canon) if s != c}


def _translated(spec: dict, pmap: dict) -> dict:
    if not pmap:
        return spec
    inner = spec["execute"]
    out = dict(spec)
    out["execute"] = lambda args: inner({pmap.get(k, k): v
                                         for k, v in (args or {}).items()})
    return out


def bind_toolbox(tools: list):
    """Bind the trace's surface names to real tools; unknown names stay
    unbound so calling a distractor fails exactly as it should."""
    base = attach_sandbox_tools(attach_repo_tools(Toolbox()))
    ws = VirtualWorkspace()
    specs = ws.specs()
    renamed = {}
    for entry in tools:
        fn = entry.get("function", entry)
        surface = fn.get("name")
        canonical = SURFACE_TO_CANONICAL.get(surface)
        pmap = _param_map(canonical, fn.get("parameters")) if canonical else {}
        if canonical and canonical in base.tools:
            renamed[surface] = _translated(base.tools[canonical], pmap)
        elif surface in specs:
            renamed[surface] = specs[surface]
        elif canonical and canonical in specs:
            renamed[surface] = _translated(specs[canonical], pmap)
    if "finish" in base.tools:
        renamed["finish"] = base.tools["finish"]
    base.tools = renamed
    return base


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().rstrip("."))


def run_trace(model, tok, tools, system, question, *, device, max_steps,
              max_new, seed):
    """Replay one conversation; return (final_answer, hops)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    tb = bind_toolbox(tools)
    allowed = [t.get("function", t).get("name") for t in tools]
    sampler = sampler_from_options({"temperature": 0})
    for hop in range(max_steps):
        text, _ = render(messages, tools, add_generation_prompt=True)
        ids = tok.encode(text)
        res = generate_turn(model, ids, sampler=sampler, tokenizer=tok,
                            device=device, max_new=max_new, protocol="chatml",
                            allowed_names=allowed, seed=seed,
                            stop_texts=(IM_END,))
        if res.kind == "tool_call" and res.tool_call:
            name = res.tool_call.get("name")
            args = res.tool_call.get("arguments") or {}
            messages.append({"role": "assistant", "content": res.thought or "",
                             "tool_calls": [{"function": {"name": name,
                                                          "arguments": args}}]})
            spec = tb.tools.get(name)
            try:
                obs = spec["execute"](args) if spec else f"unknown tool: {name}"
            except Exception as exc:                        # noqa: BLE001
                obs = f"error: {exc}"
            messages.append({"role": "tool", "content": str(obs)[:12288]})
            continue
        return (res.response or "").strip(), hop
    return None, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cache", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the training seed to reproduce the split")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = torch.load(args.cache, weights_only=False)
    ds = PrebuiltDataset(data)
    val_n = max(1, int(len(ds) * 0.05))
    _, val_set = random_split(ds, [len(ds) - val_n, val_n],
                              generator=torch.Generator().manual_seed(args.seed))
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(val_set)), min(args.n, len(val_set)))

    model = load(args.checkpoint, device=args.device)
    tok = load_tokenizer_for(model)
    ok = skipped = 0
    short_ok = short_total = long_ok = long_total = 0
    hops_ok, hops_total = {}, {}
    for i, idx in enumerate(picks, 1):
        tokens, _ = val_set.dataset.samples[val_set.indices[idx]]
        text = tok.decode(tokens)
        tools = tools_from_trace(text)
        question = first_user(text)
        expected = reference_answer(text)
        if not question or not expected:
            skipped += 1
            continue
        n_calls = text.count("<tool_call>")
        bucket = ("0-1" if n_calls <= 1 else "2-3" if n_calls <= 3 else
                  "4-7" if n_calls <= 7 else "8+")
        hops_total[bucket] = hops_total.get(bucket, 0) + 1
        try:
            got, _ = run_trace(model, tok, tools, system_text_of(text), question,
                               device=args.device, max_steps=args.max_steps,
                               max_new=args.max_new, seed=args.seed)
        except Exception as exc:                            # noqa: BLE001
            got = f"<error: {exc}>"
        short = len(expected) <= 120
        hit = bool(got) and (norm(got) == norm(expected)
                             or norm(expected) in norm(got))
        ok += hit
        if short:
            short_total += 1
            short_ok += hit
        else:
            long_total += 1
            long_ok += hit
        hops_ok[bucket] = hops_ok.get(bucket, 0) + hit
        if args.verbose or (i <= 3):
            print(f"[{i}] {'OK ' if hit else 'MISS'} q={question[:70]!r}\n"
                  f"     want={expected[:90]!r}\n      got={str(got)[:90]!r}",
                  flush=True)
        elif i % 10 == 0:
            print(f"  {i}/{len(picks)} scored, {ok} correct", flush=True)

    scored = len(picks) - skipped
    pct = 100.0 * ok / scored if scored else 0.0
    print(f"\n{ok}/{scored} correct ({pct:.0f}%)"
          + (f"  [{skipped} unparseable, skipped]" if skipped else ""))
    # The comparable number. Long free-text answers cannot be scored by string
    # equality: two correct explanations of asyncio share almost no substring,
    # so a prose miss here is a statement about the metric, not the model.
    if short_total:
        print(f"  short/grounded answers: {short_ok}/{short_total} "
              f"({100.0*short_ok/short_total:.0f}%)  <- comparable to the 41/100 bar")
    if long_total:
        print(f"  long free-text answers: {long_ok}/{long_total} "
              f"(exact match is not a meaningful metric here)")
    print("by tool-calls in the reference trace:")
    for b in ("0-1", "2-3", "4-7", "8+"):
        if hops_total.get(b):
            print(f"  {b:5s} {hops_ok.get(b,0):3d}/{hops_total[b]:3d}")


if __name__ == "__main__":
    main()
