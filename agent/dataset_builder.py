"""Parallel, cached construction of the training dataset.

The single-threaded path spent ~44 minutes before training started: 31 minutes
instantiating scenario traces at 40/s and 12 minutes tokenizing, and under DDP
*each rank did the whole thing independently*. On a 128-core machine that is
two processes using 1/64th of the hardware to do the same work twice.

This module shards generation and tokenization across processes and writes the
result to a cache keyed by the generation parameters, so a rerun with the same
settings starts training immediately and both ranks share one build.

Two wire protocols are supported at build time:
  * ``json``   - the legacy tagged serialization, byte tokenizer, loss on
                 ``<assistant>`` blocks (``encode_agent_training_example``);
  * ``chatml`` - traces re-rendered through ``agent/chatml.render`` in a mix
                 of styles, any tokenizer, loss on the renderer's spans.
The tokenizer and protocol are part of the cache key, so a cache can never be
loaded into a run built for a different one.
"""

import hashlib
import json
import multiprocessing as mp
import os
import random
import time

import torch

from agent.dataset import encode_agent_training_example, encode_with_spans
from agent.tokenizer_registry import load_tokenizer, tokenizer_key

CACHE_DIR = "data/cache"

# How often each rendering style appears in a ChatML build. "hf" is the
# verified Qwen3 Jinja; the others cover Ollama's Go template and a compact
# serialization so the model does not learn one renderer's whitespace.
DEFAULT_STYLE_MIX = {"hf": 0.6, "ollama": 0.3, "compact": 0.1}


def cache_key(**params) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _pick_style(rng, mix):
    r = rng.random()
    acc = 0.0
    for style, weight in mix.items():
        acc += weight
        if r < acc:
            return style
    return next(iter(mix))


def _worker(args):
    """Generate and tokenize one shard. Runs in a separate process."""
    (shard_idx, n, mode, pool_path, scenarios_path, scenario_fraction,
     deep_fraction, deep_min, deep_max, max_len, seed, randomize_tools,
     tokenizer_spec, protocol, style_mix) = args
    # Imports happen per-process; each shard gets its own seed so the shards
    # are different data rather than the same data repeated.
    from agent.rich_dataset import load_pool, generate
    pool = load_pool(pool_path)
    shard_seed = seed * 1000 + shard_idx
    tok = load_tokenizer(tokenizer_spec)

    traces = []
    if scenarios_path and scenario_fraction and os.path.exists(scenarios_path):
        from agent.scenario_traces import generate_from_scenarios
        from agent.tools import Toolbox
        plans = json.load(open(scenarios_path))
        rng = random.Random(shard_seed)

        def thought(situation, fallback):
            opts = pool["thoughts"].get(situation)
            return rng.choice(opts) if opts else fallback

        want = int(n * scenario_fraction)
        got, _ = generate_from_scenarios(
            plans, pool["facts"], Toolbox.DEFAULT_MEMORY, want,
            rng_seed=shard_seed, max_len=max_len, thought_fn=thought,
            sandbox_pool=pool.get("sandbox"))
        traces.extend(got)

    remaining = max(0, n - len(traces))
    if remaining:
        got, _ = generate(pool, remaining, mode=mode, max_len=max_len,
                          seed=shard_seed, deep_fraction=deep_fraction,
                          deep_min=deep_min, deep_max=deep_max)
        traces.extend(got)

    # Randomize tool names and prepend the schema, so the model must read the
    # schema rather than memorize names. Applied here so every generator gets
    # it without each one needing to know about schemas.
    if randomize_tools:
        from agent.tool_schema import randomize_trace, ToolSchemaSampler
        rng = random.Random(shard_seed + 77)
        sampler = ToolSchemaSampler(rng)
        traces = [randomize_trace(t, rng, sampler) for t in traces]

    # Tokenize in the same process: the traces never cross a pipe as strings,
    # only the far smaller token/mask arrays do.
    out = []
    if protocol == "chatml":
        from agent.chatml import render, trace_to_messages
        style_rng = random.Random(shard_seed + 99)
        for text in traces:
            messages, tools = trace_to_messages(text)
            if not any(m["role"] == "assistant" for m in messages):
                continue
            rendered, spans = render(messages, tools,
                                     style=_pick_style(style_rng, style_mix))
            tokens, mask = encode_with_spans(rendered, spans, tok, max_len)
            if len(tokens) < 2 or sum(mask[1:]) == 0:
                continue
            out.append((tokens, mask))
        return out
    for text in traces:
        tokens, mask = encode_agent_training_example(text, tok, max_len)
        if len(tokens) < 2 or sum(mask[1:]) == 0:
            continue
        out.append((tokens, mask))
    return out


def build(samples, mode="instruct", pool_path="data/ollama_pool.json",
          scenarios_path=None, scenario_fraction=0.0, deep_fraction=0.0,
          deep_min=4, deep_max=12, max_len=16384, seed=42, workers=None,
          cache=True, verbose=True, randomize_tools=True,
          tokenizer_spec=None, protocol="json", style_mix=None):
    """Return a list of (tokens, mask) pairs, built in parallel and cached."""
    workers = workers or min(64, max(1, (os.cpu_count() or 8) - 4))
    style_mix = style_mix or DEFAULT_STYLE_MIX
    key = cache_key(samples=samples, mode=mode, pool=pool_path,
                    scenarios=scenarios_path, sf=scenario_fraction,
                    df=deep_fraction, dmin=deep_min, dmax=deep_max,
                    max_len=max_len, seed=seed, rt=randomize_tools,
                    tok=tokenizer_key(tokenizer_spec), protocol=protocol,
                    styles=style_mix if protocol == "chatml" else None)
    path = os.path.join(CACHE_DIR, f"{mode}_{key}.pt")

    if cache and os.path.exists(path):
        if verbose:
            print(f"dataset cache hit: {path}", flush=True)
        return torch.load(path, weights_only=False)

    shards = workers * 2                      # 2 per worker balances stragglers
    per = max(1, samples // shards)
    jobs = [(i, per, mode, pool_path, scenarios_path, scenario_fraction,
             deep_fraction, deep_min, deep_max, max_len, seed,
             randomize_tools, tokenizer_spec, protocol, style_mix)
            for i in range(shards)]

    if verbose:
        print(f"building {samples:,} samples across {workers} processes "
              f"({shards} shards, protocol {protocol}, tokenizer "
              f"{tokenizer_key(tokenizer_spec)})...", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool_proc:
        results = pool_proc.map(_worker, jobs)
    data = [item for shard in results for item in shard]
    dt = time.time() - t0
    if verbose:
        print(f"  built {len(data):,} samples in {dt:.0f}s "
              f"({len(data)/max(dt,1):,.0f}/s)", flush=True)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)
        json.dump({"samples": samples, "mode": mode, "pool": pool_path,
                   "scenarios": scenarios_path, "sf": scenario_fraction,
                   "df": deep_fraction, "dmin": deep_min, "dmax": deep_max,
                   "max_len": max_len, "seed": seed, "rt": randomize_tools,
                   "tokenizer": tokenizer_spec, "protocol": protocol,
                   "styles": style_mix},
                  open(path.replace(".pt", ".json"), "w"), indent=1)
        if verbose:
            print(f"  cached -> {path}", flush=True)
    return data


class PrebuiltDataset(torch.utils.data.Dataset):
    """Dataset over already-tokenized (tokens, mask) pairs."""

    from training.dataset import collate_pad
    collate_pad = staticmethod(collate_pad)

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tokens, mask = self.samples[index]
        return (torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long),
                torch.tensor(mask[1:], dtype=torch.float32))
