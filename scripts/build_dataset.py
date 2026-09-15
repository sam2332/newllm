"""Build and cache a training dataset without starting a training run.

Building only ever happened inside train_split.py, so there was no way to
prepare a dataset ahead of time, and no way to find out what a cache file on
disk was built from. The cache path is a hash of the generation parameters, so
an existing cache whose arguments are forgotten cannot be reused - which is how
data/cache/instruct_1d39780d71d5d7f5.pt ended up only loadable by path.

This prints the key and the parameters that produced it, and writes a sidecar
.json next to the cache recording them.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.dataset_builder import build, cache_key, CACHE_DIR


def count_tool_calls(text: str, protocol: str) -> int:
    """Tool calls the ASSISTANT made.

    In ChatML the system block's instructions contain "<tool_call>" twice by
    template, so a naive count reports every schema-carrying trace as having
    at least two calls - which made single-tool and no-tool traces vanish from
    the report entirely.
    """
    if protocol != "chatml":
        return text.count('"tool_call"')
    body = text.split("<|im_end|>", 1)[1] if text.startswith("<|im_start|>system") else text
    return body.count("<tool_call>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="instruct", choices=["instruct", "chat"])
    ap.add_argument("--pool", default="data/ollama_pool.json")
    ap.add_argument("--samples", type=int, default=250000)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", default="data/scenarios.json")
    ap.add_argument("--scenario-fraction", type=float, default=0.0)
    ap.add_argument("--deep-fraction", type=float, default=0.0)
    ap.add_argument("--deep-min", type=int, default=4)
    ap.add_argument("--deep-max", type=int, default=12)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--project-fraction", type=float, default=0.0,
                    help="fraction of traces that are long workspace projects")
    ap.add_argument("--stories",
                    default="data/chapters.json,data/hf_stories.json",
                    help="one path or a comma-separated list")
    ap.add_argument("--max-turns", type=int, default=52)
    ap.add_argument("--char-budget", type=int, default=60000)
    ap.add_argument("--persona-fraction", type=float, default=0.0)
    ap.add_argument("--personas", default="data/personas.json")
    ap.add_argument("--format-fraction", type=float, default=0.0)
    ap.add_argument("--direct-fraction", type=float, default=0.0,
                    help="fraction of traces that call no tool at all")
    ap.add_argument("--knowledge-fraction", type=float, default=0.0,
                    help="fraction of traces drawn from the knowledge library "
                         "(python, bash, science, coding principles)")
    ap.add_argument("--knowledge",
                    default="data/knowledge.json,data/knowledge_b.json,data/knowledge_c.json",
                    help="one path or a comma-separated list")
    ap.add_argument("--knowledge-explain-fraction", type=float, default=0.5,
                    help="of the knowledge slice, the share that is pure Q&A "
                         "with no tool call; the rest are workspace code arcs")
    ap.add_argument("--knowledge-max-turns", type=int, default=24)
    ap.add_argument("--hf-fraction", type=float, default=0.0,
                    help="fraction of traces taken from imported datasets "
                         "(scripts/import_hf_dataset.py). These keep their own "
                         "tool names and schemas rather than ours")
    ap.add_argument("--hf-traces",
                    default="data/hf_openhermes.json,data/hf_xlam.json",
                    help="one path or a comma-separated list")
    ap.add_argument("--tokenizer", default="byte",
                    help="'byte' or a path to an HF tokenizer.json")
    ap.add_argument("--protocol", default=None, choices=["json", "chatml"],
                    help="default chatml for a BPE tokenizer, json for byte")
    args = ap.parse_args()
    from agent.tokenizer_registry import load_tokenizer, spec_for, tokenizer_key
    tok = load_tokenizer(args.tokenizer)
    tokenizer_spec = spec_for(tok)
    protocol = args.protocol or ("chatml" if tokenizer_spec["kind"] == "bpe" else "json")
    extras = {k: v for k, v in dict(
        project_fraction=args.project_fraction, stories=args.stories,
        max_turns=args.max_turns, char_budget=args.char_budget,
        persona_fraction=args.persona_fraction, personas=args.personas,
        format_fraction=args.format_fraction,
        direct_fraction=args.direct_fraction,
        knowledge_fraction=args.knowledge_fraction, knowledge=args.knowledge,
        knowledge_explain_fraction=args.knowledge_explain_fraction,
        knowledge_max_turns=args.knowledge_max_turns,
        hf_fraction=args.hf_fraction, hf_traces=args.hf_traces).items()
        if v not in (0, 0.0, None)} or None

    scenarios = args.scenarios if args.scenario_fraction else None
    info = {}
    data = build(info=info,samples=args.samples, mode=args.mode, pool_path=args.pool,
                 scenarios_path=scenarios,
                 scenario_fraction=args.scenario_fraction,
                 deep_fraction=args.deep_fraction, deep_min=args.deep_min,
                 deep_max=args.deep_max, max_len=args.max_len, seed=args.seed,
                 workers=args.workers, verbose=True,
                 tokenizer_spec=tokenizer_spec, protocol=protocol,
                 extras=extras)
    path = info["path"]
    print(f"cache key : {info['key']}\ncache path: {path}")


    import collections
    TOK = tok
    import random
    rng = random.Random(0)
    hops = collections.Counter()
    lens = []
    for i in rng.sample(range(len(data)), min(20000, len(data))):
        text = TOK.decode(data[i][0])
        hops[count_tool_calls(text, protocol)] += 1
        lens.append(len(data[i][0]))
    total = sum(hops.values())
    lens.sort()
    print(f"\n{len(data):,} samples cached")
    print(f"  length p50 {lens[len(lens)//2]:,}  p95 {lens[int(len(lens)*.95)]:,}"
          f"  max {lens[-1]:,}")
    print("  tool calls per trace:")
    for label, lo, hi in [("0-1", 0, 1), ("2-3", 2, 3), ("4-7", 4, 7),
                          ("8-15", 8, 15), ("16-31", 16, 31), ("32+", 32, 10**9)]:
        n = sum(v for k, v in hops.items() if lo <= k <= hi)
        print(f"    {label:6s} {n:6,}  {n/max(total,1):6.1%}")
    print(f"    max {max(hops)} hops")
    print(f"\nTrain against it with:\n  --dataset-cache {path} --tokenizer {args.tokenizer}"
          f" --protocol {protocol}")


if __name__ == "__main__":
    main()
