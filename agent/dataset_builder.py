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
     tokenizer_spec, protocol, style_mix, extras) = args
    extras = extras or {}
    # Imports happen per-process; each shard gets its own seed so the shards
    # are different data rather than the same data repeated.
    from agent.rich_dataset import load_pool, generate
    pool = load_pool(pool_path)
    shard_seed = seed * 1000 + shard_idx
    tok = load_tokenizer(tokenizer_spec)

    traces = []
    hf_traces = []          # kept apart: never renamed, they own their schemas
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

    # Long multi-turn projects over the virtual workspace (agent/project_traces).
    project_fraction = extras.get("project_fraction", 0.0)
    stories_path = extras.get("stories")
    if project_fraction and stories_path and os.path.exists(stories_path):
        from agent.project_traces import generate_project_traces
        stories = json.load(open(stories_path))
        want = int(n * project_fraction)
        traces.extend(generate_project_traces(
            stories, want, seed=shard_seed + 5, max_turns=extras.get("max_turns", 52),
            char_budget=extras.get("char_budget", 60000)))

    # Subject knowledge: python, bash, science, coding principles
    # (agent/knowledge_traces.py). Half are pure Q&A with no tool call, so
    # they also count toward the "answer directly" skill.
    knowledge_fraction = extras.get("knowledge_fraction", 0.0)
    knowledge_path = extras.get("knowledge")
    n_knowledge_direct = 0
    if knowledge_fraction and knowledge_path:
        from agent.knowledge_traces import generate_knowledge_traces, load_items
        kitems = load_items(knowledge_path)
        want = int(n * knowledge_fraction) if kitems else 0
        ef = extras.get("knowledge_explain_fraction", 0.5)
        got = generate_knowledge_traces(
            kitems, want, seed=shard_seed + 23, explain_fraction=ef,
            max_turns=extras.get("knowledge_max_turns", 24),
            char_budget=extras.get("char_budget", 60000))
        # Only the pure-Q&A half has no tool call; the artifact arcs use the
        # workspace and must keep their real schema.
        n_knowledge_direct = sum(1 for t in got if "tool_call" not in t)
        traces.extend(got)

    # Conversations that call nothing (agent/direct_traces.py).
    direct_fraction = extras.get("direct_fraction", 0.0)
    n_direct = 0
    if direct_fraction:
        from agent.direct_traces import generate_direct_traces
        stories = []
        if stories_path and os.path.exists(stories_path):
            stories = json.load(open(stories_path))
        n_direct = int(n * direct_fraction)
        direct = generate_direct_traces(n_direct, stories, seed=shard_seed + 13)
        traces.extend(direct)

    # Imported outside datasets (scripts/import_hf_dataset.py). These carry
    # their own tool schemas and their own names - xlam alone has 3,605 - so
    # they are excluded from tool-name randomization below: the point of the
    # data is names we did not invent, and renaming them to our thirteen would
    # throw that away.
    hf_fraction = extras.get("hf_fraction", 0.0)
    hf_paths = extras.get("hf_traces")
    n_hf = 0
    if hf_fraction and hf_paths:
        from agent.tool_schema import schema_block_from_entries
        rows = []
        for path in str(hf_paths).split(","):
            path = path.strip()
            if path and os.path.exists(path):
                rows.extend(json.load(open(path)))
        if rows:
            rng_hf = random.Random(shard_seed + 31)
            want = min(int(n * hf_fraction), len(rows))
            picked = rng_hf.sample(range(len(rows)), want)
            for i in picked:
                row = rows[i]
                text = row.get("trace") if isinstance(row, dict) else row
                if not text:
                    continue
                block = schema_block_from_entries(row.get("tools")) \
                    if isinstance(row, dict) else ""
                hf_traces.append(block + "\n" + text if block else text)
            n_hf = len(hf_traces)

    remaining = max(0, n - len(traces) - len(hf_traces))
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
        # Half the no-tool traces get a schema anyway: "tools offered, not
        # needed" is a distinct skill from "no tools offered".
        n_forced = (n_direct + n_knowledge_direct) // 2
        traces = [randomize_trace(t, rng, sampler, force_schema=i < n_forced)
                  for i, t in enumerate(traces)]
    traces.extend(hf_traces)

    # System-prompt families: persona voice and output-format rules. After
    # randomize_trace so the free-text block lands after the schema block.
    persona_fraction = extras.get("persona_fraction", 0.0)
    format_fraction = extras.get("format_fraction", 0.0)
    if persona_fraction or format_fraction:
        from agent.system_prompts import decorate_traces
        personas = []
        if persona_fraction and extras.get("personas") and os.path.exists(extras["personas"]):
            personas = json.load(open(extras["personas"]))
        traces = decorate_traces(traces, random.Random(shard_seed + 31), personas,
                                 persona_fraction if personas else 0.0, format_fraction)

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
          tokenizer_spec=None, protocol="json", style_mix=None, extras=None,
          info=None):
    """Return a list of (tokens, mask) pairs, built in parallel and cached.

    ``extras``: ``project_fraction``, ``stories`` (path), ``max_turns``,
    ``char_budget``, ``persona_fraction``, ``personas`` (path),
    ``format_fraction``, ``direct_fraction``, ``knowledge`` (path),
    ``knowledge_fraction``, ``knowledge_explain_fraction``,
    ``knowledge_max_turns``, ``hf_traces`` (comma-separated paths),
    ``hf_fraction``. All part of the cache key.

    ``info``, if given, receives ``{"path", "key"}``. Callers must use this
    rather than recomputing the key: a caller that rebuilt the parameter dict
    by hand silently dropped the tokenizer, protocol and extras from it and
    printed a path to a file that did not exist.
    """
    workers = workers or min(64, max(1, (os.cpu_count() or 8) - 4))
    style_mix = style_mix or DEFAULT_STYLE_MIX
    extras = extras or {}
    key = cache_key(samples=samples, mode=mode, pool=pool_path,
                    scenarios=scenarios_path, sf=scenario_fraction,
                    df=deep_fraction, dmin=deep_min, dmax=deep_max,
                    max_len=max_len, seed=seed, rt=randomize_tools,
                    tok=tokenizer_key(tokenizer_spec), protocol=protocol,
                    styles=style_mix if protocol == "chatml" else None,
                    extras=extras or None)
    path = os.path.join(CACHE_DIR, f"{mode}_{key}.pt")
    if info is not None:
        info["path"], info["key"] = path, key

    if cache and os.path.exists(path):
        if verbose:
            print(f"dataset cache hit: {path}", flush=True)
        return torch.load(path, weights_only=False)

    shards = workers * 2                      # 2 per worker balances stragglers
    per = max(1, samples // shards)
    jobs = [(i, per, mode, pool_path, scenarios_path, scenario_fraction,
             deep_fraction, deep_min, deep_max, max_len, seed,
             randomize_tools, tokenizer_spec, protocol, style_mix, extras)
            for i in range(shards)]

    if verbose:
        print(f"building {samples:,} samples across {workers} processes "
              f"({shards} shards, protocol {protocol}, tokenizer "
              f"{tokenizer_key(tokenizer_spec)})...", flush=True)
    t0 = time.time()
    # ProcessPoolExecutor rather than mp.Pool: Pool.map blocks forever if a
    # worker dies, which is how the pretraining shard build hung for 1h50m
    # with every child a zombie. The work here is bounded and local so the
    # risk is lower, but the failure mode is identical and the fix is free.
    # map() is safe here (unlike over a stream) because jobs is a finite list.
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool_proc:
        results = list(pool_proc.map(_worker, jobs))
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
                   "styles": style_mix, "extras": extras},
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
