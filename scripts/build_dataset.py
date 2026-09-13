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
    args = ap.parse_args()

    scenarios = args.scenarios if args.scenario_fraction else None
    params = dict(samples=args.samples, mode=args.mode, pool=args.pool,
                  scenarios=scenarios, sf=args.scenario_fraction,
                  df=args.deep_fraction, dmin=args.deep_min,
                  dmax=args.deep_max, max_len=args.max_len, seed=args.seed,
                  rt=True)
    key = cache_key(**params)
    path = os.path.join(CACHE_DIR, f"{args.mode}_{key}.pt")
    print(f"cache key : {key}")
    print(f"cache path: {path}")

    data = build(samples=args.samples, mode=args.mode, pool_path=args.pool,
                 scenarios_path=scenarios,
                 scenario_fraction=args.scenario_fraction,
                 deep_fraction=args.deep_fraction, deep_min=args.deep_min,
                 deep_max=args.deep_max, max_len=args.max_len, seed=args.seed,
                 workers=args.workers, verbose=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(params, open(path.replace(".pt", ".json"), "w"), indent=1)

    import collections
    from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
    import random
    rng = random.Random(0)
    hops = collections.Counter()
    lens = []
    for i in rng.sample(range(len(data)), min(20000, len(data))):
        text = TOK.decode(data[i][0])
        hops[text.count('"tool_call"')] += 1
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
    print(f"\nTrain against it with:\n  --dataset-cache {path}")


if __name__ == "__main__":
    main()
