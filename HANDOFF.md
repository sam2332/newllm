# Handoff

Current state of the project. **This file describes what is true now, not how it
got that way** - superseded notes are in [archive/notes/](archive/notes/).
How to actually run things: [scripts/run/README.md](scripts/run/README.md).
Traps that have cost real time: [docs/CODE_SMELLS.md](docs/CODE_SMELLS.md).

Last verified 2026-09-19.

## What this is

A transformer trained from scratch to use tools and chat, reading tool schemas
from its context the way real function-calling APIs pass them, so it can serve
tools it was never trained on. The end goal is **tools + chat + some roleplay**,
loaded natively in Ollama as a GGUF.

Two generations coexist. The legacy one is byte-level (273 tokens), 74.8M
dense parameters, speaking a bespoke `<assistant>{json}</assistant>` protocol.
The current one is BPE (8,192, Qwen2 pre-tokenizer), ChatML on the Qwen3
template, 561M total / 172M active MoE at `arch_version 3`.

## Where it stands

| capability | state |
|---|---|
| tool use, legacy | **41/100** held-out, reproducible. Strong at 1-2 calls (75-83%), weak at 4+ (20%) |
| tool use, MoE | 29/100 overall, **37% on grounded answers**. Better deep (57% at 4-7 hops), worse shallow (44%) |
| coherence, MoE | **broken.** "2 + 2" answers with gibberish; cannot recall a code from two turns back; 0/24 on the needle test at every length including 4k |
| coherence, L24 chat | **fixed.** Fluent English, recalls the planted code from two turns back, answers "capital of France" correctly. New weak spots are repetition and count-following, not coherence |
| knowledge | real where data was thick - correct on Python generators and `set -e`; incoherent on science (2,672 items vs python's 6,872) |
| chat | **L24 chat SFT done 2026-09-19**, val 0.5334. Coherence probe passes. Held-out grounded **6/20 (30%)**, token F1 0.51; free-text F1 0.44 |
| multi-turn, L24 chat | **4/10**. Coreference 3/3, but back-reference 0/2, ellipsis 0/2, topic-switch 0/2 |
| agentic depth, L24 chat | grounded by reference depth 0-1 **40%**, 2-3 **50%**, 4-7 **17%**, 8+ **0%**; `eval_agentic.py` 0/12 on generated tasks |
| roleplay | personas exist in data; no eval |
| Ollama | proxy works (`serve_ollama.py`); **GGUF exports cleanly** as `qwen3moe` with 65,536 context, not yet loaded into a container |

Checkpoints: `checkpoints_long_M/agent_best.pt` (legacy best, val 0.0434),
`checkpoints_moe_v2/agent_best.pt` (30,000 iters in 9.5h, best val 1.0324,
final train 0.618).

**The headline finding, and the pretrain that answered it.** The MoE model
recites its training set rather than generalising. `scripts/coherence_probe.py` shows it answering "Hello!" with a
sentence copied verbatim from `agent/direct_traces.py` and "List three fruits"
with "I can't - I have no access to your email". Results track data volume per
domain almost monotonically. The corpus explains it: 260k traces composed from
a few thousand library items, **78.8% of sentences are repeats** of another
sentence, distinct-8 of 0.288, roughly 1B tokens against 561M parameters where
Chinchilla wants 20:1. A bigger model on this corpus memorises harder.

**Pretraining fixed it.** The L24 chat model (FineWeb-Edu pretrain -> chat SFT)
passes the probe the MoE failed: "Hello!" gets "Hello. What are we working
on?" rather than a sentence copied out of `direct_traces.py`, "What is 2 + 2?"
reaches 4, and it **recalls the access code 7391 planted two turns earlier** -
the specific thing recorded above as impossible. So the recitation was a data
problem, not an architecture one, and the pretrain stage was the fix.

What is wrong now is different and worth naming precisely, because it is easy
to read the probe as a clean pass:

- **Repetition inside an answer.** Nearly every long reply restates its own
  clause - photosynthesis "provides the oxygen we breathe, produces oxygen",
  `set -e` ends on a verbatim repeat of its second sentence.
- **Counts are not followed.** "List three fruits" returns eight, with
  "Cherry" three times.
- **Right answer, invented reasoning.** 2 + 2 arrives at 4 through
  "subtracting 2 from both sides" and two irrelevant divisions. The number is
  correct and the derivation is nonsense, so arithmetic accuracy alone will
  overstate this model.
- **Thin topics still degrade.** The sky answer loops on "enters the Earth's
  atmosphere" and never reaches Rayleigh scattering; DRY is defined wrongly.

These read like an undertrained model at 1.23 epochs rather than a broken one,
and a repetition penalty at decode is untested here.

**It does not yet sustain a long tool chain.** A first pass of
`scripts/eval_agentic.py` (n=2 per cell, so indicative only, not a number to
quote) solved 1 of 12 generated tasks, and **9 of the 11 failures were
`early_stop`**: the model makes one or two tool calls and then answers with
whatever it has, rather than running the chain to the end. At depth 5 it spent
1-2 calls where 5 were needed. Depth is not degrading a working loop; the loop
stops early. The held-out eval agrees from the other direction - grounded
accuracy by reference depth runs 40%, 50%, 17%, **0% at 8+** - so two harnesses
built on different data say the same thing.

**Tool use regressed, and that was the trade.** 30% grounded against the MoE's
37% and the legacy byte model's 41/100. Not a regression to chase:
`09_sft_chat.sh` is deliberately 75% imported SmolTalk conversation with the
synthetic tool slices cut to a few percent, because this run was for the chat
half of "tools + chat". The multi-turn battery says the same thing precisely -
**coreference 3/3** ("what is 12 + 8" then "multiply that by 3" works) while
back-reference, ellipsis and topic-switch are all 0. It resolves against the
immediately preceding turn and cannot reach further back.

### Why it stops early: the imported corpus has no feedback loop in it

This is measured, not inferred. **Not one of the 60,000 `data/hf_xlam.json`
traces contains a single tool observation** - `<tool name=` appears 0 times.
The traces are a user turn, one or two assistant tool calls emitted
back-to-back, and then the trace ends; the model is never shown a result
coming back. `data/hf_openhermes.json` and `data/hf_smol_*.json` contain no
tool observations either (0 of 5,000 sampled in each).

So across the 75% of the chat SFT mix that is imported data, the demonstrated
behaviour is exactly *emit a call or two, then stop* - which is precisely what
`eval_agentic.py` measures it doing. Only the synthetic slices
(`project_fraction=0.02`, part of `knowledge`) carry genuine
call -> observation -> call arcs, and the chat mix cut them to a few percent.
**Early stopping is not a capability ceiling here; it is the behaviour the
data taught.** Depth in `hf_xlam.json` is shallow besides: of 4,000 sampled,
98.9% have 3 or fewer tool calls and 45 have 4+.

### What the literature says to do about both failures

- **The repetition is a data property, not a decoding accident.** "Repetition
  In Repetition Out" ([arXiv 2310.10226](https://arxiv.org/abs/2310.10226))
  finds a strong correlation between repetition in *training data* and
  degeneration at inference, shows the mechanism is self-reinforcing at the
  sentence level, and reports that penalising training-data repetition is the
  common factor behind several previously separate fixes - and that it still
  matters at larger model sizes and after instruction tuning. This corpus is
  **78.8% repeated sentences**, which predicts exactly the loop observed: in
  `eval_agentic.py` the model opened a `<tool_call>` and emitted the same
  `print(...)` line until the token budget ran out, at 2,048 and again at
  4,096 tokens (`kind=malformed`, never closing the tag). Raising the budget
  does not help; deduplicating the corpus is the lever.
- **The missing data type exists and is verified.** APIGen-MT
  ([arXiv 2504.03601](https://arxiv.org/abs/2504.03601)) generates multi-turn
  agent trajectories through simulated agent-human interplay with three-stage
  verification (format, real function execution, semantic), reporting 99%
  human-judged success over 200 sampled trajectories.
  [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
  is 5,000 of those trajectories, open, and is the subset used to train xLAM-2.
  It is the shape this corpus lacks: **tool results actually come back**.
  Importing it through `scripts/import_hf_dataset.py` is the cheapest
  intervention available, and it respects the teacher invariant, since the
  observations in it were produced by real function execution rather than
  written by a model.
- **Scale check before expecting much.** xLAM-2-1b-fc-r scores 43.12% on BFCL
  v3 ([xLAM](https://arxiv.org/pdf/2409.03215)); this model is 283M, roughly a
  quarter of that, so the target is a working loop, not a competitive score.
- **BFCL v3** ([leaderboard paper](https://openreview.net/forum?id=2GmDdhBdDk))
  is the standard for this and judges **by post-execution system state rather
  than by matching parameters**, which is the same choice `eval_agentic.py`'s
  `project` family makes by running the code. Its four categories - Base,
  Missing Functions, Missing Parameters, Long Context - are a ready-made list
  of what to add next; `eval_agentic.py` currently covers Base only.

## The data now on disk

| set | size | purpose |
|---|---|---|
| `data/pretrain/` | **9.5B tokens**, 19 uint16 shards, 18 GB | FineWeb-Edu. Language itself - the stage this project never had |
| `data/hf_openhermes.json` | 300,000 traces | general instruction-following, the corpus had none |
| `data/hf_xlam.json` | 60,000 traces, 3,605 distinct tool names | tool calling against schemas the model has never seen |
| `data/knowledge*.json` | 24,579 items, 6 domains | subject knowledge, teacher-written |
| `data/chapters.json`, `data/personas.json` | 894 chapters, 462 personas | long project arcs, persona voice |

The plan these support: pretrain on the shards for base competence, then SFT on
the instruct mix, with `coherence_probe.py` as the first gate rather than an
afterthought. **Stage 1 is done**: `scripts/run/06_pretrain.sh` ran 72,000
iters (**2.36B tokens** - an iter is a micro-batch; the run log claimed 9.44B, M preset, 86.9M params) in 5h43m to val loss 2.367 ->
`checkpoints_pretrain/pretrain_best.pt`. Stage 2 (`scripts/run/07_sft.sh`) had
never run - it fed the dataset builder on stdin, which a spawn process pool
cannot re-import - and was first launched 2026-09-17.

**A second, larger stage 1 is also done.** `scripts/run/10_chat_pipeline.sh`
ran the L24 preset (282,644,480 params) under `torchrun` on both GPUs:
200,000 iters per rank at batch 4 x 2,048, **3.28B tokens** in 10h45m to val
loss **2.1232** -> `checkpoints_pretrain_L24/pretrain_best.pt`. That is the
base the chat SFT warm-starts from.

**The chat SFT then OOMed at iter 1 and the pipeline stopped** (2026-09-18
04:38). L24 is 3.3x the M preset's parameters, and `09_sft_chat.sh` inherited
M's 16,384-token micro-batch with no gradient checkpointing: the backward
wanted 512 MiB more than the 5090 had, with 31.09 of 31.45 GiB already in use.
Validation at step 0 passed first, so the log looks like a healthy start.
`--grad-checkpoint` is now in the script and is **not optional at this size** -
measured, the same batch peaks at **10.7 GiB** and runs at **2.2 it/s**, which
puts the 40,000-iter run at ~5 h. The headroom matters as much as the fix: the
token-budget sampler draws longer batches later, and a run that merely fits at
iter 1 can still die at hour four.

## Environment

Two GPUs, and **since the PSU swap on 2026-09-17 both may train**. Before it the
machine hard-rebooted under sustained dual-GPU load (other loads shared the
circuit). After it a 20-minute dual load (4090 ~311 W + 5090 ~469 W) held, and
the L24 pretrain runs DDP on both at ~720 W, ~1.35x the 5090 alone. Caps need root and do not survive a reboot:

```bash
sudo nvidia-smi -i 0 -pl 350      # RTX 4090, 24 GB
sudo nvidia-smi -i 1 -pl 450      # RTX 5090, 32 GB  <- train here
loginctl enable-linger "$USER"    # or systemd kills runs at logout
```

Python 3.12 venv, torch 2.11.0+cu128, bf16 native on both cards. Use
`.venv/bin/python` for everything; system Python is 3.14 and has no torch
wheels. `scripts/run/00_preflight.sh` checks all of this.

**The machine has hard-cut five times, and not from GPU load.** Boot logs end
mid-line with no shutdown sequence and no error entries, then a 5-100 minute
gap before the next boot:

| when | what was running | draw |
|---|---|---|
| 2026-09-13 16:45 | `stress-ng --cpu 128 --vm-bytes 80%` - **no GPU load** | CPU/RAM only |
| 2026-09-14 02:46 | Ollama inference on **both** endpoints | ~250 W GPU |
| 2026-09-18 14:01 | **nothing** - the pipeline had already died at 04:38 | idle |
| 2026-09-18 16:52 | **nothing** | idle |
| 2026-09-18 17:58 | **nothing** | idle |

The three on 2026-09-18 happened with both GPUs at idle and no job running,
which rules out training load as the cause for those and weakens the
whole-machine-draw theory as a complete explanation. Whatever it is, it is
still live, so long runs need `--save-every` small enough that a cut is cheap
(the chat SFT saves every 1,000 iters, ~7.5 min).

Single-GPU training at 350 W has run 89 minutes without incident. So the
trigger is whole-machine draw, not the GPUs specifically - a 128-core EPYC at
full tilt is its own large load. Long teacher runs therefore use one endpoint
(`--endpoints 1`); `scripts/gen_*_ollama.py` save incrementally so a cut costs
minutes.

**The power caps are applied** (verified 2026-09-19: 350 W / 450 W, unit
`enabled` and `active`). `/etc/systemd/system/nvidia-power-limit.service`
carries the `-pl 350` / `-pl 450` lines and is enabled at boot.

They were *not* applied for part of 2026-09-18 and 09-19, after that day's
reboots left the unit disabled - both cards ran at their firmware maximum
(4090 450 W, 5090 575 W) and an L24 SFT step drew **510 W** on the 5090
against the 363 W measured under the cap. So check rather than assume:

```bash
nvidia-smi --query-gpu=power.limit --format=csv    # expect 350.00 W / 450.00 W
```

An earlier version of this note asserted the caps survived a reboot at a time
when they did not, which is the reason the check is written down instead of
the conclusion.

**Write logs to `logs/`, never `/tmp`** - `/tmp` is cleared on reboot, so the
evidence from a crash disappears exactly when it is wanted.

**After any hard reboot, run `git fsck`.** A power cut corrupted this repo once,
leaving a zero-byte object and an unreachable HEAD. It was recoverable from the
reflog.

## Measured performance

**The MoE run** (`deep-moe-20`, 5090, token-budget batching at 16,384 x 8
accumulation): 30,000 iters in **9h30m**, ~1.14 s/it, 363 W, 16 GB.
An iter is one micro-batch, so 30,000 iters over 66,782 micro-batches per
epoch is **~0.45 epochs** (the old figure of 3.6 multiplied by the 8
accumulation steps a second time). The LR schedule is cosine
across `--iters`, so that number has to be right at launch - a run cut short
never anneals.

**Batching reversed with MoE.** The dense byte model was fastest at batch 2
because `collate_pad` padded to the longest member. MoE amortises its expert
loop over the batch and wants the opposite:

| preset | batch x seq | tok/s |
|---|---|---|
| L-moe | 2 x 832 | 6,149 |
| L-moe | 16 x 1024 | **36,169** |
| deep-moe-20 | 2 x 832 | 3,395 |
| deep-moe-20 | 16 x 1024 | **27,108** |

`TokenBudgetSampler` groups by length to a token budget instead, measured at
0.1% padding waste. The validation loader must use it too: at a fixed batch of
32 a 19k-token trace wants ~20 GiB in cross-entropy alone, and validation runs
at step 0.

### Legacy figures (byte model, M preset, `--batch-size 2 --grad-accum 8`)

- **9.4 it/s**, 2.7 GiB peak, **341 W**. 40,000 iterations = **89 minutes**.
- Decode is **~80 tok/s and flat** with length (82 at 256 tokens, 78.5 at 3,000),
  so per-step overhead dominates, not compute. A 3,000-token response takes 38 s.
- Dataset cache load is ~96 s for 6.4 GB, paid once per run, per rank.

**Bigger batches are slower here.** `collate_pad` pads to the longest sequence in
the batch and the length spread is wide (p50 1624, p95 6761, max 14584), so one
outlier drags the whole batch to its length:

| config | opt-steps/hr | peak mem | power |
|---|---|---|---|
| bs=2 accum=8 | **3,541** | 2.7 GiB | 341 W |
| bs=8 accum=2 | 2,748 | 12.1 GiB | 450 W |
| bs=16 accum=1 | 2,315 | 17.7 GiB | 450 W |

## Architecture

`arch_version=1` reproduces the original code so old checkpoints still load.
`arch_version=2` is the default and fixes five defects that were in the original
(see `archive/notes/mistakes.md` for the full account):

1. `TransformerBlock.forward` was not pre-norm - it overwrote the residual
   stream. The single largest fix.
2. RoPE was applied once to the token embedding rather than per head to Q and K,
   so positional signal was destroyed by the first normalization. Exact span
   copying is a positional task, which is why tool arguments could not be copied.
3. `_init_weights` branched on `isinstance(p, nn.Linear)` while iterating
   `named_parameters()`, where `p` is a Tensor, so depth-scaled init never ran.
4. Inference re-ran a full forward pass over the whole sequence per token.
5. Training used fp16 + GradScaler although both GPUs do bf16 natively.

Current stack: RMSNorm, QK-Norm, SwiGLU, GQA, RoPE, fused SDPA, KV cache,
z-loss, decoupled weight decay, AdamW (0.9, 0.95), no biases, pre-norm. That is
close to Qwen3, which matters for the GGUF path below.

Verify with `.venv/bin/python scripts/debug_arch.py` (causality, KV-cache
equivalence, RoPE relative behaviour, gradient health, throughput).

## Protocol

Tool schemas go in the context, per request, like real function-calling APIs:

```
<system>{"tools":[{"name":...,"description":...,"parameters":{...}}]}</system>
<user>What is 12 + 8?</user>
<assistant>{"thought":"...","tool_call":{"name":"<a name from the schema>",...}}
```

This exists because tool names used to live only in the weights: renaming `calc`
to `compute_expression` took the old battery from 88.2% to 0.0%, while the model
still made 16 of 17 tool calls - it just called a tool that no longer existed.

Training randomizes the surface form on every trace (`agent/tool_schema.py`):
names drawn from synonymous, generic and deliberately opaque pools, paraphrased
descriptions, unused distractor tools so the model must select rather than copy,
and shuffled order. Opaque names like `xq7_eval` matter most - they leave the
description as the only signal.

`scripts/eval_renamed.py` scores the battery twice, once renamed. **Treat a large
gap as a blocking defect**: it means the model went back to memorizing names.

### Tokenizer

273 tokens: 256 byte values plus 17 atomic protocol markers. Byte-level so
arbitrary tool arguments can be copied exactly.

It has been **append-only** so far (271 -> 273 when `<system>` was added), which
is what lets `--init-checkpoint` grow embedding rows instead of refusing an older
checkpoint. A move to BPE deliberately breaks that and invalidates every
checkpoint - see "Open decisions".

## Data

`agent/rich_dataset.py` composes traces from a pool generated by a local Ollama
teacher (`scripts/gen_data_ollama.py`).

**The teacher never produces a tool result.** It supplies phrasings, reasoning
sentences and factual key/value pairs only. Every observation in every emitted
trace comes from the real `Toolbox` and is verified. **Preserve this invariant:**
if a teacher is ever allowed to state what a tool returned, the dataset stops
being ground truth.

Live pool (`data/ollama_pool.json`): 1,212 thoughts, 175 paraphrases, 682 facts.
An expanded merged pool is committed at `data/ollama_pool_merged.json` (**3,319
thoughts, 518 paraphrases, 1,678 facts**) but **not promoted**, because swapping
it invalidates the dataset cache and forces a 20-minute rebuild.

Current cache `data/cache/instruct_1d39780d71d5d7f5.pt`: 249,984 traces, 584M
tokens, mean 2,336. Roughly 47% need 4+ tool calls, tail past 40 hops.

Diversity, measured over 30,000 sampled traces - **questions were never the thin
axis**:

| axis | distinct | reuse |
|---|---|---|
| user questions | 22,718 | 1.3x |
| tool arguments | 144,342 | 1.9x |
| **thoughts** | **1,140** | **284x** |

Caches are keyed by a hash of their build parameters, so a cache whose arguments
you no longer know can only be reused with `--dataset-cache <path>`.
`scripts/build_dataset.py` writes a `.json` sidecar recording them, so new caches
do not have this problem.

## Tools

The model routes across 11 tools.

- `agent/tools.py` - `calc`, `now`, `search_memory`, `web_search`, `finish`.
- `agent/repo_tools.py` - **read-only** self-inspection: `list_files`,
  `read_file`, `search_code`, `describe_symbol`, `repo_stats`. Confined to the
  repo root by realpath containment; `..` and absolute paths rejected.
- `agent/sandbox_tools.py` - `run_bash`, `run_python` in a throwaway Docker
  container: `--network none`, `--read-only`, tmpfs /tmp, 512 MB, 1 CPU, 128
  pids, all capabilities dropped, `no-new-privileges`, uid 65534, wall-clock
  timeout. Verified unable to reach the host Ollama, resolve DNS, or see the
  repo; fork bomb contained; 2 GB allocation OOM-killed.

**Keep introspection and execution separate.** `repo_tools` reads code but runs
nothing; `sandbox_tools` runs code but cannot see the repo. Merging them would
let a confused model read a host path and act on it. A requested write-file tool
must therefore be a **third** module confined to its own workspace directory, not
an addition to `repo_tools`.

## Evaluation

**Quote held-out traces, not the battery.**

```bash
scripts/run/04_eval.sh checkpoints_long_M/agent_best.pt
```

- `scripts/eval_chatml_random.py` - the current one. Same split, same seed,
  replayed as a real agent loop against the trace's own schema. Reports
  grounded answers and free-text separately with token F1, plus buckets by
  tool-call depth, because exact match is right for "2336" and meaningless for
  a paragraph. **This is the number to report for a ChatML checkpoint.**
- `scripts/coherence_probe.py` - ten plain prompts, raw replies printed.
  **Run this first.** A checkpoint can post a respectable exact-match score
  while answering "What is 2 + 2?" with gibberish.
- `scripts/eval_random.py` - the legacy equivalent, byte tokenizer and JSON
  protocol only. It now refuses a ChatML checkpoint rather than printing 0/0.
- `scripts/eval_agent.py` - 17 hand-written cases. A smoke test. One case is 5.9
  points and 13 of the 17 are single-hop toy questions, so it cannot express
  "how many out of 100". It is **off by default as a training signal** because it
  did real damage twice: it early-stopped a run at step 3,000 of 12,000 (val
  0.239, where the full run reached 0.0709), then selected step-7000 weights over
  better step-11499 ones.
- `scripts/eval_chat.py` - scores coreference, ellipsis, back-reference, topic
  switch and no-tool-needed separately. **Never run against a trained model.**
- `scripts/diag_deep_chains.py` - classifies *why* deep chains fail.
- `scripts/eval_agentic.py` - **generated** multi-step tasks at a chosen depth
  (2-20+), not held-out traces, so the shapes are ones no generator trained on.
  Three families: `chain` (arithmetic where step k needs step k-1, the only
  true sequential dependency), `ledger` (write N files, then aggregate) and
  `lookup` (retrieve N keys, then combine). Ground truth is computed in Python
  and every observation comes from the real `Toolbox`/`VirtualWorkspace`.
  Reports solved, calls-used against calls-needed, and a first-failure
  taxonomy, because "40% at depth 10" is not actionable. `--rename` swaps in
  opaque surface names to separate chaining from name memorisation.

## Two caps that were hiding real accuracy

Both of these depressed measurements that were then reported as model quality.

**`max_steps`.** `run_agent` defaulted to 5, while **44.6% of traces need more
than 5 tool calls** and the deepest need 77. A trace needing more steps than the
cap scores 0 no matter how good the model is. Raising the eval from 20 to 64
took the score from 35/100 to **41/100**, with the 4+ bucket going 4/49 -> 10/49
and the shallow buckets byte-identical - the signature of a cap artifact rather
than noise.

| max_steps | unreachable |
|---|---|
| 5 (old default) | 44.6% |
| 20 | 14.7% |
| 32 | 7.9% |
| **64 (current)** | **0.2%** |

`run_agent` now defaults to 64 with a `max_total_new` budget of 16,384 tokens
across the whole run, because 64 steps x 4,096 tokens is otherwise ~55 minutes
of decode for one request that never emits a closing tag.

**Response length in the data.** The final `response` field averages **11
characters**, p99 is 132 and the longest in 20,000 traces is 313. **Not one
response exceeds 500 characters.** Raising `max_new` to 4,096 therefore does
nothing on its own - the model emits ~11 characters and stops, because that is
all it has ever seen. Long responses need generated data, not a bigger cap.

## What is actually wrong

`diag_deep_chains.py` on 40 traces needing 4+ calls: **35 diverged at hop 0**.
The failure is grounding on the *first* call, not compounding with depth. One
trace chained its own wrong operand correctly for 24 hops - the chaining
machinery works.

26 of 40 emitted a tool name absent from that trace's schema, picking the right
capability with the wrong surface name from the same pool (`read_store` ->
`memo_get`, `wz_query` -> `kb_query`).

Constrained decoding (`agent/constrained.py`) masks logits to the schema grammar
so an absent name is unrepresentable. Measured on the 12k checkpoint it is worth
**24/100 vs 22/100** and leaves the 4+ bucket unchanged at 1/49 - masking an
invalid name just makes the model pick a valid wrong one. **Not yet re-measured
on `checkpoints_long_M`.**

## What moved the needle

Training length, by a wide margin. 12,000 iterations was **0.77 epochs** - 449M
of 584M tokens, ~6 tokens per parameter where a 74.8M model wants nearer 20.
Validation loss was still falling monotonically when the run ended and train loss
sat level with val, so there was no overfitting to stop for.

| | 12k iters | 40k iters |
|---|---|---|
| best val | 0.0709 | **0.0434** |
| held-out | 22/100 | **41/100** |

Every bucket improved; the largest gain was single-call (7/20 -> 15/20), exactly
as the hop-0 diagnosis predicted. Validation loss is now flattening (0.0460 @
28k, 0.0434 @ 38.5k, 0.0435 @ 39.5k), so **further steps have reached diminishing
returns and data is the next lever.**

## Open decisions

### Vocabulary: byte-level vs BPE

Measured on this corpus, compression **saturates at ~4k vocab**:

| vocab | tokens/trace | vs bytes | embedding params | model total |
|---|---|---|---|---|
| **4,096** | 773 | **3.01x** | 6.3M | **80.7M** |
| 16,384 | 772 | 3.01x | 25.2M | 99.5M |
| 27,300 | 772 | 3.01x | 41.9M | 116.3M |
| 32,768 | 772 | 3.01x | 50.3M | 124.7M |

A 100x vocab buys **zero** extra compression over 15x and costs +35.6M
parameters, all in embeddings. The corpus is structured JSON with a bounded set
of protocol markers and facts, so BPE exhausts useful merges early. **This holds
only while the data stays protocol-heavy** - natural prose for roleplay would
change it, and 8k-16k becomes defensible then.

Split digits in the pre-tokenizer. It costs 10% compression (772 vs 700
tokens/trace) and prevents BPE merging `4710` into one token, which is the
classic reason subword models fail at arithmetic. Calc chains dominate the deep
traces this is meant to fix.

Arguments for switching: **3x fewer tokens** (mean trace 2,328 -> 772, `max_len`
16,384 -> ~6,144), long responses become practical, and deep chains stop
straining context. Argument against: it invalidates every checkpoint and breaks
the append-only tokenizer guarantee.

### Ollama: the server works today; native GGUF is the next track

`scripts/serve_ollama.py` is a real Ollama server, not a shim. Standard
semantics: the client defines tools, one `/api/chat` call yields one assistant
turn (`tool_calls` or content), the client executes and sends `role: "tool"`
back. Implements `/api/chat` (NDJSON streaming, `options`, `num_ctx`,
`num_predict`, `stop`, `seed`), `/api/generate`, `/api/tags`, `/api/show`,
`/api/ps`, `/api/version`, `/` health, and `/v1/chat/completions` + `/v1/models`
(OpenAI shape, SSE). A client system prompt is placed **after** the tool-schema
block, never in its place. Context truncation keeps the system head and drops
whole turns oldest-first.

The single-turn generator is `agent/turn.py:generate_turn` / `stream_turn`,
built on `agent/generate.py:generate_tokens` and `agent/ollama_context.py`.
Every future eval should go through it, so what is measured is what is served.

Verified with the official `ollama` Python package
(`scripts/test_ollama_client.py`, 12 hard checks): list/show parse, the full
tool loop with a name NOT in any training pool (`evaluate_sum`), streaming
chunks agree with non-streaming, `num_predict` exhaustion is a 200 with
`done_reason: "length"`, unknown model is a 404 `ResponseError`, OpenAI endpoint
returns `tool_calls`.

**Grammar-constrained decoding is on by default** (`--no-constrained` to turn
it off). On this checkpoint it is the difference between 12/12 and 11/12:
unconstrained, the model emits a memorized pool name (`evaluate_math`, `op_42`)
for a tool the schema calls `evaluate_sum` - the same hop-0 failure
`diag_deep_chains.py` found. On held-out traces it was only worth +2/100 because
those traces use pool names; for a client's *novel* names it is decisive.

Two model gaps the e2e exposed, both for Track B's data, not the server:
- **Parameter names were never randomized.** Every calc-like tool in training
  takes `expr`, so given `add_numbers(a, b)` the model emits
  `add_numbers({"expr": "12 + 8"})` - right name (forced), wrong argument shape.
  `agent/tool_schema.py` must randomize parameter names and descriptions the
  way it randomizes tool names.
- **No tools + a chatty prompt produces junk** (`Say hello.` -> garbage). Traces
  with no tool use are rare and chat has never been trained.

Native GGUF loading is Track C of the approved plan: retrain on the Qwen3 chat
template with a GPT-2-style BPE (Qwen2 pre-tokenizer regex) so the model loads
with a standard Modelfile. The architecture is already Qwen3-isomorphic except
four small deltas (attention biases, interleaved RoPE, embedding scale, MoE
router bias), which become `arch_version=3`.

### Context length is now dynamic

`max_len` is an allocation hint, not a limit. `RoPECache` grows its tables on
demand instead of raising, so a sequence longer than the declared `max_len`
produces exactly the logits it would have with a table preallocated to that size
(verified to 5.96e-07, float32 rounding). 131,072 tokens have been run through a
model declared `max_len=512`.

This works because there are **no learned position parameters** - RoPE is a pure
function of position and its buffers are non-persistent, so they never enter a
checkpoint. Memory scales linearly with length (0.06 GiB at 4k, 0.42 at 40k,
1.33 at 131k on a small model), which is flash SDPA doing its job.

**This is not length generalization.** The model runs at any length and is only
good near the lengths it trained on, which is ~14.5k - the longest trace in the
dataset. Past that, RoPE extrapolation degrades. Fixing that needs either RoPE
scaling at inference (NTK-aware, YaRN, linear position interpolation) or training
on genuinely long sequences. Enabling 32k costs almost nothing; *earning* 32k is
a data problem.

Offloading is not needed at this scale. The weights are 300 MB and a 32k training
step peaked at 3.6 GiB on a 32 GB card.

### Long responses

The target is responses up to ~3,000 tokens. At byte level that is 3,000
characters, about **484 words**. With BPE at 3.01x it is ~9,030 characters, about
**1,456 words**, in the same 38 seconds of decode.

Generation caps are now 4,096 everywhere (`run_agent`, `agent/chat.py`,
`serve_ollama.py`). `max_new` is a ceiling, not a target - generation
stops at `</assistant>`, so an ordinary tool call still costs the ~100 tokens it
needs. The tradeoff is the worst case: a model that never emits a closing tag
now burns 4,096 tokens, ~51 s at the measured ~80 tok/s, instead of 120.

At 51 s per long response, `torch.compile` or CUDA graphs on the single-token
step is the obvious lever, since decode is per-step-overhead bound and flat with
length.

## The retrain pipeline is proven end to end (2026-09-14)

Everything between "generate data" and "load it in Ollama" now exists and was
run as one chain on a throwaway 44M model (`/tmp/smoke_chain.sh` in the git
history of this note): corpus dump -> BPE tokenizer -> ChatML dataset ->
arch_version=3 MoE training -> the Python server (protocol picked from the
checkpoint) -> the official `ollama` client -> GGUF export -> `ollama create`
in the docker instance -> `ollama run`.

Measured on the smoke run:
- **BPE compression on ChatML traces: 4.14x** over byte-level (2,785 -> 672
  tokens per trace at 4k vocab), better than the 3.01x on the old format.
- **Export equivalence** (`scripts/test_export_gguf.py`): PyTorch and llama.cpp
  agree on the `<think>` prediction, and next-token logprobs agree within
  0.10 at 32 tokens and **0.027 at 287 tokens with 10/10 top-10 overlap**.
  The remaining difference is the f16 KV cache.
- MoE presets at the mean BPE trace length (batch 2, grad checkpointing):
  L-moe 464M/152M active 292 ms; deep-moe-20 561M/172M 516 ms;
  deep-moe-24 671M/204M 632 ms. The Python expert loop is ~3.4x a dense step
  at equal active params; worth a grouped-GEMM pass before a long run.

Ollama quirks that cost time and are now handled:
- A generated CONTROL token (`<think>`) is consumed by Ollama: `eval_count`
  is one more than the number of logprob entries and it never appears in the
  text. Distribution comparisons must sit *inside* the thought.
- `raw: true` bypasses the template AND the thinking parser (`thinking` is
  null). Fine for comparisons; not how clients will use the model.
- `$(...)` in bash strips a trailing newline, which silently changes a prompt
  that ends in `assistant\n`. Drive prompts from Python.

What the retrained model will see: Qwen3's exact Jinja rendering 60% of the
time, Ollama's Go-template rendering 30% (no newlines inside `<think>`, one
`<tool_call>` pair, unmerged tool results, blank line after `system`), a
compact serialization 10%. The `ollama` style is a best-effort reading of the
Go template; verify it against a live render (`OLLAMA_DEBUG=1`) before the
big run.

## Next

1. **Train the chat model.** Half the stated goal, pipeline ready, zero
   measurements. `--init-checkpoint` warm-starts it from the instruct model, and
   EOT semantics are already fixed so it yields to the user instead of writing
   their next turn.
2. **Size sweep.** "Smallest model that can do this" has never been asked - every
   number is the M preset. S is 22.8M against M's 74.8M, and a run is 89 minutes.
3. **Promote the expanded pool** and rebuild, now that steps have plateaued.
4. **Decide the tokenizer**, before spending more runs on the byte vocabulary.
5. **Write-file tool**, as a third confined module. Note the wrinkle: write tools
   have side effects and the dataset builder generates 3,000 traces/s with
   memoized outputs, so generation needs a virtual workspace that still yields
   genuine observations - otherwise the "teacher never fabricates a tool result"
   invariant quietly breaks.
6. **Re-baseline `agent/test_agent_regression.py`.** It has 5 pre-existing errors
   because it points at the legacy 256-vocabulary checkpoint. Keep the 3-of-5
   competition gate.

## Gotchas

- **`$!` is not the trainer.** `setsid`/`env` fork, so the pid you capture exits
  immediately and a liveness check on it reports a healthy run as dead. Use
  `pgrep -f "train_split.py --mode instruct"`.
- **...but that `pgrep -f` self-matches inside a wait loop.** The same pattern
  that makes `pkill -f` kill your own shell makes
  `until ! pgrep -f "train_split.py ..."; do sleep 5; done` never exit: the
  loop's own command line contains the pattern, so it matches itself and spins
  forever after the trainer is long gone. Break the self-match with a character
  class - `pgrep -f "train_spli[t].py"` - or check the GPU instead.
- **Progress looks stalled during a battery eval.** ~2 minutes with no log
  output. Check `nvidia-smi` before concluding anything hung.
- **A log stopping at exactly 4096 bytes** means the process was killed with one
  filesystem block buffered - historically, systemd linger.
- **Changing the pool invalidates every dataset cache built from it.**
- **`--iters` is the denominator of the cosine LR schedule.** Raising it between
  resumed segments rewrites the schedule mid-run.
- **Scale expectations.** This is a 25-75M parameter byte-level model. The
  architecture and data pipeline are sound and the numbers are real, but frontier
  behaviour is orders of magnitude away in both parameters and training tokens.
  Judge changes against held-out traces and the chat eval, not general capability.
