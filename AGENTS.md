# Guide for AI agents working in this repo

Updated 2026-09-15. If something here disagrees with
[HANDOFF.md](HANDOFF.md), HANDOFF wins. [CLAUDE.md](CLAUDE.md) carries the
same rules with the architecture behind them; [docs/CODE_SMELLS.md](docs/CODE_SMELLS.md)
lists the traps that have actually cost time.

## Read first

1. [HANDOFF.md](HANDOFF.md) - current state, measured numbers, open decisions.
2. [scripts/run/README.md](scripts/run/README.md) - how to run anything.
3. [goals.md](goals.md) - what this is for.

Notes in [archive/notes/](archive/notes/) are **superseded**. They are kept for
the reasoning behind past decisions, and several of their headline claims are now
known to be wrong - that file's README lists which.

## Rules that are not negotiable

- **`.venv/bin/python` for everything.** System Python is 3.14 and has no torch
  wheels.
- **Train on one GPU.** Both under load browns the machine out - it is a fuse
  in the building. Use `CUDA_VISIBLE_DEVICES=1` (the 5090) or the run scripts.
  Whole-machine draw is what trips it, so a CPU-saturating job counts: two
  Ollama containers spilling a 34 GB model onto 96 cores did it.
- **Never `pkill -f` a pipeline script.** The pattern matches the shell whose
  own command line contains it and kills your session mid-command. Three times
  so far. Use explicit PIDs or `scripts/run/launch_*.sh`.
- **Read the protocol from the checkpoint.** A byte-tokenizer eval pointed at a
  ChatML checkpoint prints `0/0` having asked the model nothing, which reads as
  a broken model. `04_eval.sh` dispatches on `config["tokenizer"]["kind"]`.
- **The teacher never produces a tool result.** The Ollama generators supply
  phrasings, reasoning sentences, prose and code only. Every observation comes
  from the real `Toolbox` or `VirtualWorkspace` and is verified. Break this and
  the dataset stops being ground truth.
- **Keep the three tool tiers separate.** `repo_tools` reads the repo and runs
  nothing; `sandbox_tools` runs code and cannot see the repo. Merging them lets a
  confused model read a host path and act on it. A write-file tool goes in a
  third module confined to its own workspace.
- **Report held-out numbers, not the battery.** `scripts/eval_random.py`. The
  17-case battery is a smoke test where one case is worth 5.9 points.
- **Do not overwrite `data/ollama_pool.json` directly.** `gen_data_ollama.py`
  builds from scratch and overwrites `--out`. Use `scripts/run/01_gen_pool.sh`,
  which generates to a new file and merges.

## Common commands

```bash
scripts/run/00_preflight.sh                               # before anything else
scripts/run/02_build_dataset.sh                           # ~3 min, ~12 GB
scripts/run/03_train.sh                                   # detached, one GPU
scripts/run/04_eval.sh checkpoints_moe_v2/agent_best.pt   # picks the harness

# what it SAYS, not what it scores - run this first
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/coherence_probe.py <ckpt>
.venv/bin/python scripts/notify_test.py                   # check the webhook

# diagnose the dominant failure (chains of 4+ tool calls)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/diag_deep_chains.py 40

# tests
.venv/bin/python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -q
.venv/bin/python smoke_test.py
.venv/bin/python test_modern_features.py
.venv/bin/python scripts/debug_arch.py

# follow a run
tr '\r' '\n' < logs/<log> | grep -a training: | tail -1
```

## Protocol

Tool schemas are passed per request, in context. Tool names are randomized during
training so they cannot be memorized.

```text
<system>{"tools":[{"name":"op_42","description":"Evaluate an arithmetic expression.","parameters":{...}}]}</system>
<user>What is 12 + 8?</user>
<assistant>{"thought":"I need the calculator.","tool_call":{"name":"op_42","arguments":{"expr":"12 + 8"}}}</assistant>
<tool name=op_42>20</tool>
<assistant>{"thought":"The calculation is complete.","response":"20"}</assistant>
```

- Every assistant turn is valid JSON with a required `thought` string.
- Exactly one of `tool_call` or `response`.
- `tool_call` is `{"name": string, "arguments": object}`.
- Every assistant turn ending in a final response gets a supervised EOT, so the
  model yields instead of writing the user's next message.
- Loss applies only to assistant spans.

## Layout

| path | what |
|---|---|
| `model/transformer.py` | the model; `arch_version=2` is current |
| `model/` | attention, RoPE, RMSNorm, SwiGLU, MoE, SSM, sparse attention |
| `agent/tokenizer.py` | 273-token byte vocab + atomic protocol markers |
| `agent/tool_schema.py` | name randomization, distractors, schema blocks |
| `agent/rich_dataset.py` | trace composition from the pool |
| `agent/dataset_builder.py` | parallel build + cache keyed by build params |
| `agent/agent_loop.py` | inference loop with tool execution |
| `agent/tools.py`, `repo_tools.py`, `sandbox_tools.py` | the three tiers |
| `training/trainer.py` | training, resume, LR schedule, early stopping |
| `scripts/run/` | the runnable pipeline |

## Pitfalls that have actually bitten

- **`$!` is not the trainer.** `setsid`/`env` fork; the captured pid exits
  immediately and a liveness check on it reports a healthy run as dead. Use
  `pgrep -f "train_split.py --mode instruct"`.
- **A stalled progress bar is usually a battery eval** (~2 min, no output), not
  a hang. Check `nvidia-smi`.
- **A log ending at exactly 4096 bytes** means the process was killed with one
  block buffered - historically systemd linger being off.
- **Dataset caches are keyed by a hash of build parameters.** If you do not know
  what built a cache, you can only reuse it with `--dataset-cache <path>`.
  `scripts/build_dataset.py` writes a sidecar so new caches avoid this.
- **Changing the pool invalidates every cache built from it.**
- **`--iters` is the cosine schedule's denominator.** Do not raise it between
  resumed segments.
- **Bigger batches are slower here** - `collate_pad` pads to the longest sequence
  in the batch and the length spread is wide.
- **Early stopping on the battery is off for a reason.** It killed a run at step
  3,000 of 12,000 and then selected worse weights. Validation loss picks weights.

## Conventions

- Comments explain *why*, not what. Match the surrounding density.
- New measurements go in HANDOFF.md, replacing the old claim rather than being
  appended beneath it. Append-only docs are how the last set went stale.
- A superseded conclusion moves to `archive/notes/` with a note saying what
  replaced it.
- Quote sample sizes and intervals. `35/100` not `35%`; the per-bucket numbers
  have wide, overlapping intervals.
