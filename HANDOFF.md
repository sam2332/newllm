# Handoff

Current state of the project. **This file describes what is true now, not how it
got that way** - superseded notes are in [archive/notes/](archive/notes/).
How to actually run things: [scripts/run/README.md](scripts/run/README.md).

Last verified 2026-09-14.

## What this is

A byte-level transformer (74.8M parameters) trained from scratch to use tools
through a JSON protocol. It reads tool schemas from its context the way real
function-calling APIs pass them, so it can serve tools it was never trained on.

The end goal is a small-to-medium model that does **tools + chat + some
roleplay**, usable from Ollama.

## Where it stands

| capability | state |
|---|---|
| tool use (instruct) | **41/100** on held-out traces. Works; not accurate yet |
| chat | **never trained.** Pipeline and eval are ready and unused |
| roleplay | **does not exist.** No data, no eval, no format decision |
| smallest viable size | **never tested.** Every number here is the M preset |
| Ollama | proxy works (`serve_ollama.py`). Native GGUF not attempted |

Best checkpoint: `checkpoints_long_M/agent_best.pt` (40,000 iters, val 0.0434).

```
41/100 correct, by tool calls the reference needed:
  1 call   15/20     <- fine
  2 calls  10/12     <- fine
  3 calls   6/19
  4+ calls 10/49     <- half the distribution lives here
```

**That last row is the whole result.** 49 of 100 held-out traces need four or
more tool calls and the model gets 10 of them. Everything else is in decent shape.

## Environment

Two GPUs, but **train on one**. Both at stock limits is 450 W + 575 W = 1025 W of
GPU before the CPU, and the machine hard-rebooted under sustained dual-GPU load.
Capped and single-GPU, a run sits at ~350 W and has completed 89 minutes without
incident. Caps need root and do not survive a reboot:

```bash
sudo nvidia-smi -i 0 -pl 350      # RTX 4090, 24 GB
sudo nvidia-smi -i 1 -pl 450      # RTX 5090, 32 GB  <- train here
loginctl enable-linger "$USER"    # or systemd kills runs at logout
```

Python 3.12 venv, torch 2.11.0+cu128, bf16 native on both cards. Use
`.venv/bin/python` for everything; system Python is 3.14 and has no torch
wheels. `scripts/run/00_preflight.sh` checks all of this.

**After any hard reboot, run `git fsck`.** A power cut corrupted this repo once,
leaving a zero-byte object and an unreachable HEAD. It was recoverable from the
reflog.

## Measured performance

Single GPU, M preset, `--batch-size 2 --grad-accum 8 --grad-checkpoint`:

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

- `scripts/eval_random.py` - N random held-out traces from the same split and
  seed the trainer used, with the toolbox renamed to match each trace's own
  randomized schema. **This is the number to report.**
- `scripts/eval_agent.py` - 17 hand-written cases. A smoke test. One case is 5.9
  points and 13 of the 17 are single-hop toy questions, so it cannot express
  "how many out of 100". It is **off by default as a training signal** because it
  did real damage twice: it early-stopped a run at step 3,000 of 12,000 (val
  0.239, where the full run reached 0.0709), then selected step-7000 weights over
  better step-11499 ones.
- `scripts/eval_chat.py` - scores coreference, ellipsis, back-reference, topic
  switch and no-tool-needed separately. **Never run against a trained model.**
- `scripts/diag_deep_chains.py` - classifies *why* deep chains fail.

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
