# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A from-scratch transformer trained to call tools and chat, aimed at loading
natively in Ollama as a GGUF. No pretrained weights. See [goals.md](goals.md)
for intent, [HANDOFF.md](HANDOFF.md) for current measured state (HANDOFF wins
on any disagreement), and [AGENTS.md](AGENTS.md) for the short version of the
rules below.

## Non-negotiables

- **`.venv/bin/python` for everything.** System Python is 3.14 and has no torch
  wheels. Never `pip install` into system Python.
- **One GPU at a time.** Both under load browns the machine out - this is a
  fuse in the building, not a driver problem. `CUDA_VISIBLE_DEVICES=1` is the
  5090; the run scripts set it for you. The whole-machine draw is what trips
  it, so a CPU-saturating job counts too (two Ollama containers spilling a
  34 GB model onto 96 cores did it).
- **The teacher never produces a tool result.** Ollama-generated data supplies
  phrasings, reasoning sentences, prose and code only. Every observation in a
  trace comes from the real `Toolbox` or `VirtualWorkspace`. Break this and the
  dataset stops being ground truth.
- **Keep the three tool tiers separate.** `agent/repo_tools.py` reads the repo
  and runs nothing; `agent/sandbox_tools.py` runs code and cannot see the repo;
  `agent/workspace_tools.py` writes, confined to `workspace/`. Merging them
  lets a confused model read a host path and act on it.
- **Never `pkill -f` a pipeline script.** The pattern matches the shell whose
  own command line contains it, so the kill takes out your session mid-command.
  This has happened three times. Use explicit PIDs, or the launchers in
  `scripts/run/`.
- **Report held-out numbers, not the 17-case battery**, where one case is worth
  5.9 points.
- **Long runs survive logout only with linger on, and belong in named `screen`
  sessions.** Without `loginctl enable-linger`, systemd kills the user slice at
  logout even through `nohup`/`setsid`; the telltale is a log that stops at
  exactly 4096 bytes. After any hard reboot run `git fsck` - a power cut has
  corrupted this repo once.
- **Callers of the dataset builder need a file and a `__main__` guard.** It fans
  out over a spawn process pool whose workers re-import `__main__` by path, so
  code piped in on stdin crashes every worker.

## Commands

```bash
scripts/run/00_preflight.sh          # GPU caps, systemd linger, venv, caches
scripts/run/02_build_dataset.sh      # tokenize traces into a cache
scripts/run/03_train.sh              # detached, one GPU
scripts/run/04_eval.sh <checkpoint>  # picks the harness from the checkpoint
scripts/run/05_export.sh <checkpoint> # GGUF + equivalence test (needs Ollama up)

# two-stage path: pretrain, then SFT on top of the base
scripts/run/06_pretrain.sh           # stage 1, FineWeb-Edu shards, ~24 h
INIT=checkpoints_pretrain/pretrain_best.pt scripts/run/07_sft.sh
screen -dmS sft env FOREGROUND=1 scripts/run/07_sft.sh   # stage 2 in the foreground
```

First-time venv setup (Python 3.12, cu128 torch) and power caps are in
[scripts/run/README.md](scripts/run/README.md).

Everything is overridable by environment variable
(`ITERS=80000 SIZE=L-moe scripts/run/03_train.sh`).

```bash
# tests: the whole suite, one file, one test
.venv/bin/python -m pytest agent/ model/ -q
.venv/bin/python -m pytest agent/test_chatml.py -q
.venv/bin/python -m pytest agent/test_notify.py::test_async_send_does_not_block_the_caller -q

# what the model actually says - run this BEFORE trusting any score
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/coherence_probe.py <checkpoint>

# follow a detached run (tqdm writes \r, not \n)
tr '\r' '\n' < logs/<log> | tail -1
```

`agent/test_agent_regression.py` loads a real checkpoint and asserts accuracy
thresholds; `CHECKPOINT=<path>` overrides which. Two of its gates currently
fail on merit - the thresholds were set against an older checkpoint and were
deliberately left alone.

## Architecture

**Data is generated, not collected.** Nothing here is scraped from a chat log.
Generators compose traces from libraries an Ollama teacher wrote once
(`data/personas.json`, `data/chapters.json`, `data/knowledge.json`), which is
what keeps teacher cost bounded: a few thousand items recombine into hundreds
of thousands of traces. `agent/rich_dataset.py` (tool patterns),
`agent/project_traces.py` (52-turn workspace arcs), `agent/knowledge_traces.py`
(subject Q&A and code artifacts), `agent/direct_traces.py` (no tool call at
all) each emit the **legacy tagged form** - `<user>`, `<assistant>{json}`,
`<tool name=X>`. That form is the common currency: everything downstream
consumes it, including `scripts/import_hf_dataset.py`, which converts outside
datasets into it so they inherit the whole pipeline.

**Two wire protocols, one model.** The legacy JSON protocol
(`<assistant>{"thought","tool_call"}`) serves old byte-tokenizer checkpoints.
The current protocol is **ChatML** (`agent/chatml.py`), rendering Qwen3's
template exactly so llama.cpp and Ollama parse it natively. `agent/turn.py`
holds the adapter: `generate_turn(..., protocol="json"|"chatml")` is the single
path every server and eval goes through. Which one a checkpoint speaks is
recorded in its config (`tokenizer.kind == "bpe"` means chatml) - read it, never
assume. **A harness pointed at the wrong protocol does not fail; it reports
0/0 having asked the model nothing**, which reads exactly like a broken model
and has cost a full afternoon. The legacy evals now refuse instead
(`require_legacy_protocol`), and both evals refuse when nothing parses.

**The pipeline.** `agent/dataset_builder.py` fans generation out across
processes, applies tool-name and parameter randomization
(`agent/tool_schema.py`) and system-prompt decoration
(`agent/system_prompts.py`), renders to the target protocol, tokenizes, and
caches to `data/cache/<mode>_<key>.pt`. The cache key covers every parameter
including the tokenizer - pass `info=` to `build()` to learn the path rather
than recomputing it, since a caller that rebuilt the dict by hand omitted the
tokenizer and printed a path to a file that did not exist.

**Training** (`training/trainer.py`) is length-bucketed by **token budget**,
not batch size: with p50 ~700 tokens and p95 ~16k, fixed batching either wastes
most of the batch on padding or OOMs on the long tail. MoE needs large batches
(measured 6,149 tok/s at batch 2 vs 36,169 at batch 16 - the opposite of what
the old dense byte model wanted). The LR schedule is warmup then cosine to
0.1x across `--iters`, so **the iteration count must be right at launch**:
stopping a too-long run early leaves the weights where a near-peak LR put them.
An iter is **one micro-batch**, not an optimizer step, so epochs are
`iters / batches_per_epoch` and tokens are `iters x batch x seq_len` - both
were once computed with an extra `x grad_accum`, overstating them 4-16x.

**Model** (`model/`): `arch_version=3` is Qwen3-isomorphic so GGUF export is a
name mapping - no attention bias, half-split NEOX RoPE at base 1e6, no
embedding scale, RMSNorm, per-head QK-norm, GQA, SwiGLU, optional MoE with a
Switch-style load-balancing loss plus router z-loss. Deviating from Qwen3 here
breaks `scripts/export_gguf.py`, which is the point of the whole exercise.

**Serving**: `scripts/serve_ollama.py` implements the Ollama HTTP API
(`/api/chat` with NDJSON streaming, `/api/generate`, `/api/tags`, `/api/show`,
plus `/v1/chat/completions`) over `generate_turn`, so the `ollama` Python
client, `ollama run` and OpenAI-shaped clients all work against a checkpoint.

**Pretraining** (stage 1 via `scripts/pretrain.py` / `06_pretrain.sh`; stage 2
`07_sft.sh` deliberately keeps the synthetic slices a minority, with imported
OpenHermes conversations and xlam tool calls carrying the weight):
`scripts/build_pretrain_shards.py` streams FineWeb-Edu, tokenizes with the
project BPE, and packs ids into flat `uint16` shards with EOT as the document
separator. `data/pretrain/manifest.json` describes them. This exists because
the synthetic corpus alone teaches the model to recite - measured 78.8% of its
sentences are repeats of another sentence, and the resulting checkpoint cannot
repeat a four-digit code from two turns earlier.

## Measurement

Scores from a composed corpus flatter the model. Exact match is right for a
grounded answer (`"2336"`) and meaningless for prose - two correct
explanations of asyncio share almost no substring - so
`scripts/eval_chatml_random.py` reports grounded and free-text separately, plus
token F1, plus buckets by tool-call depth. **Run `scripts/coherence_probe.py`
first**: a checkpoint can post a respectable exact-match number while answering
"What is 2 + 2?" with gibberish, because the traces it does well on are
formulaic.

## Notifications

`config.json` (gitignored; `config.example.json` is the template) can carry a
Discord webhook. Training, teachers, the supervisor, evals and export post to
it via `agent/notify.py`. Failures there are swallowed after one stderr warning
and posts go out on a daemon thread, so a chat room cannot break or slow a run;
the last message of a process must be sent `blocking=True`, since a daemon
thread does not survive interpreter shutdown.
