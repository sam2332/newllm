# Setting up a run

Every script here is idempotent, takes its settings from environment variables,
and can be run from anywhere. Numbers quoted are measured on this machine
(RTX 4090 + RTX 5090, 128-core EPYC-class, 499 GB RAM).

```bash
scripts/run/00_preflight.sh                  # check, fix anything it flags
scripts/run/02_build_dataset.sh              # ~20 min, ~6 GB   (skip if cached)
scripts/run/03_train.sh                      # ~89 min at 40k iters
scripts/run/04_eval.sh checkpoints_long_M/agent_best.pt
```

`01_gen_pool.sh` is optional and only needed when you want more language
diversity in the data. It changes the pool, which makes every existing dataset
cache stale, so it must be followed by `02_build_dataset.sh`.

## First-time setup

The server has no system PyTorch and system Python is 3.14, which has no torch
wheels. Use 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install tqdm numpy pytest requests
```

Use `.venv/bin/python` for everything.

### Two things that silently kill runs

**1. systemd stops your user slice at logout.** `Linger=no` means the user
service manager exits when your last login session ends and takes every process
with it. `nohup` and `setsid` do not save you. A run launched at 06:02 was gone
by 06:12. The signature is a log that stops at exactly 4096 bytes: one
filesystem block buffered and never flushed.

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger      # want Linger=yes
```

**2. Power.** Both GPUs at stock limits is 450 W + 575 W = 1025 W of GPU before
the CPU, and the machine hard-rebooted under sustained dual-GPU load. Cap them
(needs root, and does not persist across reboot):

```bash
sudo nvidia-smi -i 0 -pl 350
sudo nvidia-smi -i 1 -pl 450
```

A single capped GPU training run measures **~350 W** and has completed 89
minutes without incident. Ollama inference is much lighter — 113 W + 170 W with
both endpoints busy. **After any hard reboot, run `git fsck`**: a power cut has
corrupted this repo once, leaving a zero-byte object and an unreachable HEAD.
It was recoverable from the reflog.

## The scripts

| script | does | cost |
|---|---|---|
| `00_preflight.sh` | checks linger, power caps, venv, torch, disk, caches, Ollama | instant |
| `01_gen_pool.sh` | widens the pool with the Ollama teacher, merges into the existing one | ~9 min |
| `02_build_dataset.sh` | builds + caches the tokenized dataset (CPU only) | ~20 min, 6 GB |
| `03_train.sh` | trains on one GPU, detached | ~89 min / 40k iters |
| `03_train_segmented.sh` | same, in timed segments you can stop between | same + overhead |
| `04_eval.sh` | scores a checkpoint on held-out traces | ~10 min |

Override anything by environment variable:

```bash
ITERS=80000 OUT=checkpoints_xl GPU=1 scripts/run/03_train.sh
SEGMENT=15 scripts/run/03_train_segmented.sh
N=300 BATTERY=0 scripts/run/04_eval.sh checkpoints_long_M/agent_best.pt
```

## The retrain pipeline (ChatML + BPE + MoE, exports to GGUF)

The current checkpoints speak a bespoke byte-level protocol that only this
repo can serve. The retrain produces a model that loads natively in Ollama.
Every step below has been run end to end on a smoke model
(`scripts/test_export_gguf.py` proved the export is the same model).

```bash
scripts/run/01b_gen_library.sh                  # personas + chapter library (teacher, hours, both GPUs)
.venv/bin/python scripts/dump_corpus.py --samples 40000 --out data/corpus/traces.txt
.venv/bin/python scripts/train_tokenizer.py --corpus data/corpus/traces.txt --vocab 8192 16384
TOKENIZER=data/tokenizer/bpe8192.json PROJECT_FRACTION=0.25 PERSONA_FRACTION=0.2 \
  FORMAT_FRACTION=0.15 MAX_LEN=32768 SAMPLES=300000 scripts/run/02_build_dataset.sh
TOKENIZER=data/tokenizer/bpe8192.json SIZE=deep-moe-20 OUT=checkpoints_moe ITERS=120000 \
  scripts/run/03_train.sh
scripts/run/04_eval.sh checkpoints_moe/agent_best.pt        # held-out tool traces
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_system.py checkpoints_moe/agent_best.pt --judge
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_longhorizon.py checkpoints_moe/agent_best.pt
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/needle_test.py checkpoints_moe/agent_best.pt
scripts/run/05_export.sh checkpoints_moe/agent_best.pt      # GGUF -> docker Ollama, equivalence test
```

Rules that follow from how the pieces fit:

- **The tokenizer is part of the cache key.** A cache built with one tokenizer
  cannot be trained with another; `03_train.sh` must get the same `TOKENIZER`
  the build did (the cache's `.json` sidecar records it).
- **A BPE tokenizer implies the ChatML protocol** and `arch_version=3`; the
  server and evals pick the protocol from the checkpoint automatically.
- **The teacher owns both GPUs while it runs** (~30 GB each). Nothing else fits
  on the 5090 until `01b` finishes.
- **Two stories per outline request.** Five overflowed the token budget and
  the truncated JSON was silently dropped - it looked like the teacher was
  producing nothing.
- The MoE step time is dominated by the Python loop over experts (~3.4x a
  dense step at equal active parameters); `deep-moe-20` is ~516 ms/step at
  the mean trace length.

## Choosing settings

**Train longer than feels necessary.** At 12,000 iterations the model had seen
0.77 epochs — 449M of 584M tokens, about 6 tokens per parameter, where a 74.8M
model wants nearer 20. Validation loss was still falling monotonically when the
run ended. Going to 40,000 took val loss 0.0709 → 0.0434 and held-out accuracy
22/100 → 35/100. Only then does the curve start to flatten.

**Keep `--batch-size 2 --grad-accum 8`.** Bigger batches are *slower* here:
`collate_pad` pads to the longest sequence in the batch and the spread is wide
(p50 1624, p95 6761, max 14584), so one outlier drags the whole batch to its
length.

| config | opt-steps/hr | peak mem | power |
|---|---|---|---|
| bs=2 accum=8 | **3,541** | 2.7 GiB | 341 W |
| bs=8 accum=2 | 2,748 | 12.1 GiB | 450 W |
| bs=16 accum=1 | 2,315 | 17.7 GiB | 450 W |

**Leave `EVAL_EVERY=0`.** The 17-case battery is off as a *training* signal on
purpose. One case is worth 5.9 points, and it caused real damage twice: it
early-stopped a run at step 3,000 of 12,000 (val 0.239, where the full run
reached 0.0709), and then it selected step-7000 weights over better step-11499
ones. Validation loss picks the weights; score the finished checkpoint with
`04_eval.sh`. If you do turn it on, use `EVAL_EVERY=1000 EVAL_PATIENCE=10` — it
costs ~2 min per call, which at `eval-every 500` exceeds the training time.

**Validation is capped at 50 batches.** A full pass is 12,499 samples and ~4
minutes, run every 500 steps — longer than the training itself. `val_loader`
does not shuffle, so a fixed subset is still a fair comparison.

## Running in segments

`03_train_segmented.sh` stops each segment on the wall clock, writes resume
state and exits 0; the next segment continues from that step. **`ITERS` must
stay identical across segments** — it is the denominator of the cosine LR
schedule, so raising it between segments rewrites the schedule mid-run.

Resume state is kept when a run stops early or on time, and removed only when it
actually reaches `ITERS`. It survives a DDP ↔ single-GPU switch.

## Reading the results

Quote the held-out number, not the battery. The battery is 17 hand-written
cases, 13 of them single-hop toy questions; it cannot resolve finer than 5.9
points and is not a sample of the training distribution.

Current best (`checkpoints_long_M`, 40k iters, val 0.0434):

| tool calls in reference | score |
|---|---|
| 1 | 15/20 |
| 2 | 10/12 |
| 3 | 6/19 |
| 4+ | **4/49** |
| **overall** | **35/100** |

**Half the held-out distribution needs 4+ tool calls, and that bucket is the
whole result.** Diagnose it with:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/diag_deep_chains.py 40
```

Last time that ran, 35 of 40 deep traces diverged at **hop 0** — the failure is
grounding on the first call, not compounding with depth. One trace chained its
own wrong operand correctly for 24 hops.

## Gotchas

- **`$!` is not the trainer.** `setsid`/`env` fork, so the pid you capture exits
  immediately and a `kill -0` check on it reports a healthy run as dead. Find the
  real one with `pgrep -f "train_split.py --mode instruct"`.
- **Progress looks stalled during an eval.** A battery call takes ~2 minutes with
  no log output. Check `nvidia-smi` before concluding anything is hung.
- **Dataset caches are keyed by a hash of their build parameters.** A cache whose
  arguments you no longer know cannot be reused by parameter — pass
  `--dataset-cache <path>` instead. `02_build_dataset.sh` writes a `.json`
  sidecar recording them, so new caches do not have this problem.
- **Changing the pool invalidates every cache built from it.** Rebuild after
  `01_gen_pool.sh`.
- **The teacher never produces a tool result.** It supplies phrasings, reasoning
  sentences and factual key/value pairs only; every observation comes from the
  real `Toolbox` and is verified. If a teacher is ever allowed to state what a
  tool returned, the dataset stops being ground truth.
