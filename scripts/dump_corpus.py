"""Dump rendered ChatML traces as text, for training the tokenizer.

The dataset cache holds token ids, which is useless for training a tokenizer,
and the generators produce the legacy tagged form. This renders freshly
generated traces through ``agent/chatml.render`` in the same style mix the
dataset builder uses and writes them out, one trace per record, separated by
``\\x1e`` so ``scripts/train_tokenizer.py`` measures compression per trace.

    .venv/bin/python scripts/dump_corpus.py --samples 20000 --out data/corpus/traces.txt
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chatml import render, trace_to_messages
from agent.dataset_builder import DEFAULT_STYLE_MIX, _pick_style
from agent.rich_dataset import load_pool, generate
from agent.tool_schema import randomize_trace, ToolSchemaSampler

RS = "\x1e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/ollama_pool.json")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--chat-fraction", type=float, default=0.3)
    ap.add_argument("--deep-fraction", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/corpus/traces.txt")
    # The tokenizer must see the mix it will encode. Without the project arcs
    # it never sees prose, and a vocabulary tuned only on JSON tool traces
    # compresses chapters badly.
    ap.add_argument("--project-fraction", type=float, default=0.25)
    ap.add_argument("--stories", default="data/chapters.json")
    ap.add_argument("--max-turns", type=int, default=52)
    ap.add_argument("--persona-fraction", type=float, default=0.2)
    ap.add_argument("--personas", default="data/personas.json")
    ap.add_argument("--format-fraction", type=float, default=0.15)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    rng = random.Random(args.seed)
    sampler = ToolSchemaSampler(rng)
    traces = []

    n_project = int(args.samples * args.project_fraction)
    if n_project and os.path.exists(args.stories):
        from agent.project_traces import generate_project_traces
        stories = json.load(open(args.stories))
        traces.extend(generate_project_traces(stories, n_project, seed=args.seed,
                                              max_turns=args.max_turns, min_turns=8))
        print(f"  {len(traces):,} long project arcs from {len(stories)} stories")

    rest = args.samples - len(traces)
    n_chat = int(rest * args.chat_fraction)
    for mode, n in (("instruct", rest - n_chat), ("chat", n_chat)):
        got, _ = generate(pool, n, mode=mode, max_len=args.max_len, seed=args.seed,
                          deep_fraction=args.deep_fraction if mode == "instruct" else 0.0,
                          deep_min=4, deep_max=40)
        traces.extend(got)
    rng.shuffle(traces)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    chars = 0
    personas = (json.load(open(args.personas))
                if args.persona_fraction and os.path.exists(args.personas) else [])
    from agent.system_prompts import decorate_traces
    with open(args.out, "w", encoding="utf-8") as fh:
        for text in traces:
            text = randomize_trace(text, rng, sampler)
            text = decorate_traces([text], rng, personas, args.persona_fraction,
                                   args.format_fraction)[0]
            messages, tools = trace_to_messages(text)
            if not messages:
                continue
            rendered, _ = render(messages, tools, style=_pick_style(rng, DEFAULT_STYLE_MIX))
            fh.write(rendered + "\n" + RS + "\n")
            chars += len(rendered)
    print(f"wrote {len(traces):,} traces, {chars/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
