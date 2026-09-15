"""Score a checkpoint on N random held-out traces from the training split.

The 17-case battery in eval_agent.py is a smoke test, not a sample of the data:
13 of its cases are single-hop toy questions, while the training distribution
has a median trace of ~1.6k tokens and chains many tools. At 17 cases one
answer is worth 5.9 points, so the battery cannot express "how many out of 100
would it get right". This can.

Traces carry randomized tool names, so the toolbox is renamed to match the
schema in each trace's own <system> block before the agent runs. Names are
mapped back to capabilities through NAME_POOLS, which is injective; a name
that appears in no pool is a distractor and is deliberately left unbound, so
calling it fails exactly as it should.
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/lmeadows/llm")
import torch
from torch.utils.data import random_split

from agent.agent_loop import (run_agent, _extract_assistant_json,
                              _find_final_response)
from agent.dataset_builder import PrebuiltDataset
from agent.repo_tools import attach_repo_tools
from agent.sandbox_tools import attach_sandbox_tools
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from agent.tool_schema import NAME_POOLS
from agent.tools import Toolbox
from scripts.eval_agent import load, require_legacy_protocol

SURFACE_TO_CANONICAL = {s: c for c, names in NAME_POOLS.items() for s in names}


def renamed_toolbox(system_block: str):
    """A Toolbox whose keys are the surface names this trace actually uses."""
    body = re.search(r"<system>(.*?)</system>", system_block, re.S)
    if not body:
        return None
    try:
        schema = json.loads(body.group(1))
    except json.JSONDecodeError:
        return None
    base = attach_sandbox_tools(attach_repo_tools(Toolbox()))
    renamed = {}
    for entry in schema.get("tools", []):
        surface = entry.get("name")
        canonical = SURFACE_TO_CANONICAL.get(surface)
        if canonical and canonical in base.tools:
            renamed[surface] = base.tools[canonical]
    if "finish" in base.tools:
        renamed["finish"] = base.tools["finish"]
    base.tools = renamed
    return base


def split_trace(text: str):
    """Return (system_block, question, ground-truth final response)."""
    sys_m = re.search(r"<system>.*?</system>", text, re.S)
    usr_m = re.search(r"<user>(.*?)</user>", text, re.S)
    if not usr_m:
        return None, None, None
    # The last assistant block that carries a "response" is the final answer.
    final, pos = "", 0
    while True:
        start, end, parsed = _extract_assistant_json(text, after_pos=pos)
        if start is None:
            break
        if isinstance(parsed.get("response"), str):
            final = parsed["response"]
        pos = end
    return (sys_m.group(0) if sys_m else None), usr_m.group(1), final


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().rstrip("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    # No hardcoded hash: that default named one particular byte-tokenized
    # cache and nothing checked it still existed or matched the checkpoint.
    ap.add_argument("--cache", default=None,
                    help="dataset cache to draw held-out traces from; "
                         "defaults to the largest data/cache/instruct_*.pt")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the training seed to reproduce the split")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=64,
                    help="the deepest reference chains are 77 hops; "
                         "at 20 this scores 14.7% of traces 0 "
                         "by construction, at 64 only 0.2%")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--constrained", action="store_true",
                    help="mask logits to the schema's grammar, so a tool name "
                         "absent from this trace's schema cannot be emitted")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not args.cache:
        import glob
        found = sorted(glob.glob("data/cache/instruct_*.pt"),
                       key=os.path.getsize, reverse=True)
        if not found:
            raise SystemExit("no dataset cache found; pass --cache")
        args.cache = found[0]
        print(f"using cache {args.cache}")
    data = torch.load(args.cache, weights_only=False)
    ds = PrebuiltDataset(data)
    val_n = max(1, int(len(ds) * 0.05))
    # Same split and seed as train_split.py, so these traces are genuinely
    # held out rather than samples the model has already fitted.
    _, val_set = random_split(ds, [len(ds) - val_n, val_n],
                              generator=torch.Generator().manual_seed(args.seed))
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(val_set)), min(args.n, len(val_set)))

    model = load(args.checkpoint, device=args.device)

    require_legacy_protocol(model, "eval_random.py")
    ok = skipped = 0
    hops_ok, hops_total = {}, {}
    for i, idx in enumerate(picks, 1):
        tokens, _ = val_set.dataset.samples[val_set.indices[idx]]
        text = TOK.decode(tokens)
        system, question, expected = split_trace(text)
        tb = renamed_toolbox(system) if system else None
        if tb is None or not question or not expected:
            skipped += 1
            continue
        hops = text.count('"tool_call"')
        bucket = min(hops, 4)
        hops_total[bucket] = hops_total.get(bucket, 0) + 1
        r = run_agent(model, question, tb, device=args.device, greedy=True,
                      tokenizer=TOK, system=system,
                      constrained=args.constrained,
                      max_steps=args.max_steps, max_new=args.max_new)
        # Two things count as the model's answer, and they are different
        # strings. final_answer is the raw tool result (_select_final_answer
        # prefers steps[-1]["result"]), while the reference is the assistant's
        # "response" sentence. Comparing only those two marks a correct "20"
        # wrong against a reference of "The answer is 20".
        tool_answer = r["final_answer"] or ""
        said, _ = _find_final_response(r.get("trace") or "")
        e = norm(expected)
        hit = bool(e) and (
            e == norm(said) or e in norm(said)
            or e == norm(tool_answer)
            or (norm(tool_answer) and norm(tool_answer) in e))
        got = said or tool_answer
        ok += hit
        hops_ok[bucket] = hops_ok.get(bucket, 0) + hit
        if args.verbose:
            print(f"[{i:3d}] {'OK ' if hit else 'MISS'} {hops}h  "
                  f"q={question[:60]!r}\n      want={expected[:70]!r}\n"
                  f"      got ={got[:70]!r}", flush=True)
        elif i % 10 == 0:
            print(f"  {i}/{len(picks)}  {ok} correct", flush=True)

    scored = len(picks) - skipped
    if skipped == len(picks):
        # Every trace unparseable means the cache was tokenized for a
        # different protocol, not that the model failed: a BPE/ChatML cache
        # decoded with the byte tokenizer yields no <assistant> block at all.
        # Printing 0/0 here reads as "the model is broken" when the model was
        # never asked anything, which cost an afternoon once already.
        raise SystemExit(
            f"all {skipped} traces were unparseable: {args.cache} does not "
            f"look like a byte-tokenized legacy cache.\n"
            f"For a ChatML dataset use scripts/eval_chatml_random.py.")
    print(f"\n{ok}/{scored} correct ({ok/max(scored,1):.0%})"
          f"{f'  [{skipped} unparseable, skipped]' if skipped else ''}")
    print("by tool-calls in the reference trace:")
    for b in sorted(hops_total):
        label = f"{b}+" if b == 4 else str(b)
        print(f"  {label} calls: {hops_ok.get(b,0):3d}/{hops_total[b]:<3d}")


if __name__ == "__main__":
    main()
