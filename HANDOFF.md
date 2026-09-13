# Handoff

## Environment

The server had no PyTorch. The project was developed on Windows against an
RTX 5060 Ti; this machine is Linux with two GPUs. Rebuilt as:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install tqdm numpy pytest requests
```

Use `.venv/bin/python` for everything. System Python is 3.14 and has no torch
wheels. Verified: torch 2.11.0+cu128, RTX 4090 (sm_89, 24GB) and RTX 5090
(sm_120, 32GB), bf16 supported on both.

Two Ollama instances run in Docker on ports 11434 and 11435
(containers `ollama-4090`, `ollama-3090`). `qwen3:30b-a3b-q8_0` supports tools.

## Architecture

`arch_version` selects the model generation. `arch_version=1` reproduces the
original code exactly, so old checkpoints still load. `arch_version=2` is the
default for new models and contains the fixes below.

Defects that were in the original code:

1. `TransformerBlock.forward` was not pre-norm. It did `x = self.norm1(x)` and
   then `x = x + attn(x)`, overwriting the residual stream rather than leaving
   it intact. This is the single largest fix.
2. RoPE was applied once to the token embedding rather than per head to Q and K
   inside attention, so positional signal was largely destroyed by the first
   normalization. Exact span copying is a positional task, which is why tool
   arguments could not be copied reliably.
3. `_init_weights` branched on `isinstance(p, nn.Linear)` while iterating
   `named_parameters()`, where `p` is a Tensor. Those branches never ran, so
   depth-scaled residual init was never applied.
4. Inference re-ran a full forward pass over the whole sequence per token and
   re-parsed the entire string each step.
5. Training used fp16 + GradScaler although both GPUs support bf16 natively.

Also added: RMSNorm, QK-Norm, SwiGLU, GQA, fused SDPA, KV cache, z-loss,
decoupled weight decay, AdamW betas (0.9, 0.95).

Run `.venv/bin/python scripts/debug_arch.py` to re-verify causality, KV-cache
equivalence, RoPE relative behaviour, gradient health and throughput.

## Results

Battery: `.venv/bin/python scripts/eval_agent.py <checkpoint> -v`

| model                              | overall | numeric | 2-hop | 3-hop | JSON |
|------------------------------------|---------|---------|-------|-------|------|
| v1 original, S, single-step data    | 47.1%   | 0%      | 0%    | 0%    | 85.7% |
| v2, S, single-step data             | 64.7%   | 43%     | 0%    | 0%    | 100%  |
| v2, M (74.8M), mixed data           | 88.2%   | 100%    | 100%  | 100%  | 100%  |

`checkpoints_v2_M/agent_best.pt` is the current best instruct checkpoint at
88.2% (15/17). It chains three tools correctly, e.g.
`search_memory -> web_search -> calc`.

This **supersedes the previous conclusion** recorded in `progress.md` that the
model "cannot reliably retain exact web/memory tool arguments and multi-step
arithmetic together" and that dataset work should stop. That conclusion was
reached against the defective architecture. With the fixes, multi-hop accuracy
went from 0% to 100% on the same synthetic distribution.

Remaining failures are both routing, on phrasings absent from the old templates:

- "Who is the president of the united states?" - emits no tool call
- "How many planets are there?" - routes to `search_memory`, returns `cavepeople`

## Data

The old generator was the real ceiling: 4000 traces contained 34 distinct
`thought` strings, 12 web facts and 7 memory keys. Validation loss reaches
0.0000 because the model memorizes 34 sentences.

`scripts/gen_data_ollama.py` drives both Ollama instances to produce language
diversity. Current pool (`data/ollama_pool.json`): 1212 thoughts across 16
situations, 175 paraphrases, 682 facts (171 numeric, usable for chains).

**The teacher never produces a tool result.** It supplies phrasings, reasoning
sentences and factual key/value pairs only. Every observation in every emitted
trace comes from the real `Toolbox` and is verified. A spot check of 559
observations found 0 mismatches. Preserve this invariant: if a teacher is ever
allowed to state what a tool returned, the dataset stops being ground truth.

`agent/rich_dataset.py` composes traces from the pool. Repo-tool outputs are
memoized during generation (3000 traces/s).

## Instruct / chat split

Two models, one architecture:

```bash
.venv/bin/python scripts/train_split.py --mode instruct --size M   # max_len 768
.venv/bin/python scripts/train_split.py --mode chat --size M       # max_len 2048
```

`--init-checkpoint` warm-starts one from the other and grows embedding rows
rather than refusing a smaller-vocabulary checkpoint.

EOT semantics changed. It previously appeared once per trace, at the very end,
so the model learned end-of-episode and never end-of-turn; in conversation it
would run past its own answer and write the user's next message. Every
assistant turn emitting a final response now ends with a supervised EOT.

The tokenizer grew 271 -> 273 (`<system>`, `</system>`). The extension is
append-only: ids 0-270 keep their meaning.

Evaluate chat with `scripts/eval_chat.py`, which scores coreference, ellipsis,
back-reference, topic switch and no-tool-needed separately.

## Tool schemas are in the context (important)

The model used to see only `<user>question</user>`. Tool names lived in its
weights, never in its context. Measured: renaming `calc` to
`compute_expression` took the battery from **88.2% to 0.0%**. It still made 16
of 17 tool calls; it just called `calc`, which no longer existed.

The context now carries the schema the way real function-calling APIs do:

```text
<system>{"tools":[{"name":...,"description":...,"parameters":{...}}]}</system>
<user>What is 12 + 8?</user>
<assistant>{"thought":"...","tool_call":{"name":"<a name from the schema>",...}}
```

Training randomizes the surface form on every trace (`agent/tool_schema.py`):
names drawn from synonymous, generic and deliberately opaque pools, paraphrased
descriptions, unused distractor tools so the model must select rather than copy
the only option, and shuffled order. The opaque names matter most - they leave
the description as the only signal.

`scripts/eval_renamed.py` scores the battery twice, once with the original names
and once renamed. **Treat a large gap as a blocking defect**: it means the model
has gone back to memorizing names and cannot serve a user's own tools.

Checkpoints trained before this change (including `checkpoints_v2_M`, the 88.2%
baseline) do NOT read schemas and score 0% under renaming. Keep the baseline for
comparison, but it is not shippable as a general tool-calling model.

## Tools

The model routes across 11 tools.

- `agent/tools.py` - `calc`, `now`, `search_memory`, `web_search`, `finish`.
  `Toolbox(memory=..., web_kb=...)` now accepts injected data.
- `agent/repo_tools.py` - read-only self-inspection: `list_files`, `read_file`,
  `search_code`, `describe_symbol`, `repo_stats`. Confined to the repo root;
  `..` and absolute paths rejected.
- `agent/sandbox_tools.py` - `run_bash`, `run_python` in a throwaway Docker
  container: `--network none`, `--read-only`, tmpfs /tmp, 512MB, 1 CPU, 128
  pids, all capabilities dropped, `no-new-privileges`, uid 65534, wall-clock
  timeout. Verified unable to reach the host Ollama, resolve DNS, or see the
  repo; fork bomb contained; 2GB allocation OOM-killed.

**Keep introspection and execution separate.** `repo_tools` reads code but runs
nothing; `sandbox_tools` runs code but cannot see the repo. Merging them would
let a confused model read a host path and act on it.

`scripts/gen_sandbox_pool.py` runs a task list once in real containers and keeps
only what succeeds, so dataset generation never starts a container.

## Ollama compatibility

`agent/ollama_format.py`, verified against a live instance. Two deviations from
a naive OpenAI assumption:

- `tool_calls` is a list of `{"id", "function": {"index", "name", "arguments"}}`
- `function.arguments` is a real JSON object, not a JSON-encoded string

Tool results return as `{"role": "tool", "tool_name": ..., "content": ...}`.
The internal protocol stays compact because a byte-level model pays per
character; `thought` maps to Ollama's `thinking` field.

## Constrained decoding

`agent/constrained.py` masks logits to the assistant-JSON grammar, making
invalid JSON and non-existent tool names unrepresentable. Enable with
`run_agent(..., constrained=True)`.

It does not currently improve accuracy, because `arch_version=2` already emits
100% valid JSON greedily. It is a guarantee, not a quality gain. It will matter
for smaller models, higher temperatures, or a larger tool set.

## Tests

```bash
.venv/bin/python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -q
.venv/bin/python smoke_test.py
.venv/bin/python test_modern_features.py
```

27 passed; smoke and modern-feature tests pass.

`agent/test_agent_regression.py` has 5 errors. **Pre-existing**, not caused by
these changes: it points at `checkpoints/agent_best.pt`, the legacy
256-vocabulary checkpoint. Re-baseline it against `checkpoints_v2_M/` or the
newer instruct checkpoint, keeping the 3-of-5 competition gate.

## Next

1. Re-baseline `test_agent_regression.py` and `agent/promote.py` against a
   current checkpoint. This is the main piece of unfinished work.
2. Train and evaluate the chat model; `scripts/eval_chat.py` is ready.
3. The two routing failures should be re-checked after training on the diverse
   pool - both look like template-coverage gaps rather than model limits.
4. Generation is bound by Python and kernel-launch overhead at this size, not
   compute. If decode speed matters, CUDA graphs or `torch.compile` on the
   single-token step is the lever; the KV cache alone does not help here.

## Scale expectations

This is a 25-75M parameter byte-level model. The architecture and data pipeline
are now sound and the numbers above are real, but frontier-model behaviour is
several orders of magnitude away in both parameters and training tokens. Judge
changes against the battery and the chat eval, not against general capability.

## Background jobs kept dying: two separate causes

Several long runs were lost. They had two unrelated causes, and conflating them
cost most of a night.

### 1. systemd user manager was not lingering

`Linger=no` means systemd terminates the user's service manager when the last
login session ends, taking every `--user` unit with it. A run launched at
06:02:31 was gone by 06:12:55, which is when the user manager itself restarted.
`nohup` and `setsid` do not help: the whole user slice goes.

Fixed with `loginctl enable-linger lmeadows` (now `Linger=yes`). Verify with:

```bash
loginctl show-user lmeadows -p Linger
```

A run's log stopping at exactly 4096 bytes is the signature - one filesystem
block buffered and never flushed because the process was killed.

### 2. Suspected power loss under dual-GPU load

The machine hard-rebooted while both GPUs were at ~85%. Combined draw at stock
limits:

| GPU | default limit | max |
|-----|---------------|-----|
| RTX 4090 | 450 W | 600 W |
| RTX 5090 | 575 W | 600 W |

That is 1025 W of GPU before the CPU (128-core EPYC-class) and the rest of the
system. If the PSU or circuit cannot sustain it, the machine browns out under
sustained load - which matches training dying rather than idling dying.

**This is a hypothesis, not a confirmed diagnosis.** To isolate it, run on GPU 1
only (`CUDA_VISIBLE_DEVICES=1`, single process, no torchrun) and see whether a
long run survives. If it does, the dual-GPU power draw is implicated.

Capping power would be the mitigation, but it needs root:

```bash
sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 400
```

Unprivileged attempts fail with "Insufficient Permissions".

### What survives a power cut

- Training checkpoints every `--save-every` steps, written atomically, resumed
  with `--resume`. Losing a run costs about a minute, not hours.
- The dataset cache under `data/cache/` rebuilds in ~95 s and is reused across
  runs, so a restart does not repeat it.
- Git has been corrupted once by a power cut (a zero-byte object left HEAD
  unreachable). It was recoverable from the reflog. Run `git fsck` after any
  hard reboot before trusting the repo.

## Short segmented runs (power-limited operation)

The GPUs are now capped below stock because sustained draw was tripping a fuse:

| GPU | cap | stock |
|-----|-----|-------|
| RTX 4090 (0) | 350 W | 450 W |
| RTX 5090 (1) | 450 W | 575 W |

**Measured, single GPU, the M instruct config (`--batch-size 2 --grad-accum 8
--grad-checkpoint`, `max_len` 16384):**

- 127 ms per micro-step, 2.7 GiB peak, **341 W** - under the 350 W cap already.
- 12,000 iterations is **~25 minutes of compute**, not hours.

The "466 hours" in the old log was step 0 only, and step 0 includes a full
validation pass. The wall clock was never dominated by training:

- validation is 12,499 samples at batch 2, ~4 min, and ran every 500 steps -
  roughly an hour per run, more than the training itself. Use `--val-batches`.
- the dataset cache is 6.4 GB and takes ~96 s to load, per rank.

Larger batches are *slower* here, because `collate_pad` pads to the longest
sequence in the batch and the length distribution is very wide (p50 1624,
p95 6761, p100 14584). One outlier drags the whole batch to its length:

| config | opt-steps/hr | peak mem | power |
|--------|--------------|----------|-------|
| bs=2 accum=8  | 3,541 | 2.7 GiB | 341 W |
| bs=8 accum=2  | 2,748 | 12.1 GiB | 450 W |
| bs=16 accum=1 | 2,315 | 17.7 GiB | 450 W |

Keep `--batch-size 2 --grad-accum 8`.

### Running in segments

`--max-minutes N` stops a segment cleanly on the wall clock: it writes resume
state, prints the step it reached and exits 0. **Keep `--iters` identical across
segments** - it is the denominator of the cosine LR schedule, so raising it
between segments rewrites the schedule mid-run. A segment that stopped on time
does not overwrite `agent_best.pt` or `run.json`; the resume state is the
artifact.

```bash
while ! grep -q "saved ->" logs/seg.log 2>/dev/null; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_split.py \
    --mode instruct --size M --batch-size 2 --grad-accum 8 \
    --grad-checkpoint --iters 12000 --val-batches 50 \
    --save-every 250 --max-minutes 20 --resume \
    --out checkpoints_schema_M >> logs/seg.log 2>&1 || break
done
```

Three bugs made segmented runs unsafe before this; all three are fixed:

1. `_save_resume_state` wrote `self.model.state_dict()`, which under DDP carries
   a `module.` prefix. A run started with `torchrun` could not be resumed on one
   GPU - exactly the move the power problem forces. It now saves from the
   unwrapped module and strips the prefix from older files on load.
2. Accuracy state (`best_acc`, `best_acc_step`, `eval_history`,
   `_best_acc_state`) was neither saved nor restored. Every segment restarted at
   `best_acc = -1`, so its first evaluation always counted as an improvement and
   overwrote `agent_best_bestacc.pt` with worse weights, and early-stopping
   patience reset to zero each segment.
3. `train_split.py` deleted the `.resume` file whenever `train()` returned. With
   a time-budget stop that would have deleted the state the next segment needs.

## Where the schema-trained model actually stands

Single-GPU run on the 5090, 12,000 iterations, 41.5 minutes, no power incident.
`checkpoints_schema_M/agent_best.pt`, best val 0.0709, final train loss 0.054.

The 17-case battery reports 17.6%. That is **3 cases out of 17** - the battery
cannot resolve anything finer than 5.9 points, and 13 of its 17 cases are
single-hop toy questions, so it is not a sample of the training distribution.
Use `scripts/eval_random.py`, which scores held-out traces from the same split
and seed the trainer used:

```bash
.venv/bin/python scripts/eval_random.py checkpoints_schema_M/agent_best.pt -n 100
```

**22/100 on random held-out traces (95% CI 15-31%).** By chain depth:

| tool calls in reference | score | 95% CI |
|-------------------------|-------|--------|
| 1    |  7/20 (35%) | 18-57% |
| 2    |  9/12 (75%) | 47-91% |
| 3    |  5/19 (26%) | 12-49% |
| 4+   |  1/49 ( 2%) |  0-11% |

**Half the held-out distribution needs 4 or more tool calls, and the model gets
1 of 49.** That single bucket is the entire result: bring 4+ chains to the level
of the 2-call bucket and the overall number triples. Nothing else on the list
matters as much.

The per-bucket intervals are wide and overlap heavily - 1-call scoring below
2-call is not a real inversion, just n=20 and n=12. Do not read depth-by-depth
ordering from a 100-trace run; raise `-n` before drawing conclusions.

Two harness defects were found writing this, both of which understated the
score. They are fixed, and are worth knowing about before trusting any new eval:

- `max_steps` defaulted to 8 while deep chains run past 12 hops, so those traces
  scored 0 by construction. Now 20.
- Scoring compared `run_agent`'s `final_answer` - which is the raw tool result,
  because `_select_final_answer` prefers `steps[-1]["result"]` - against the
  reference `"response"` sentence. A correct "20" was marked wrong against "The
  answer is 20". It now checks the model's own response text as well.

Both together were worth 2 points (20 -> 22), so the 4+ failures are real.

### Early stopping is too aggressive for this battery

The first run died at step 3,000 of 12,000: `--eval-patience 4` at
`--eval-every 500` on a 17-case battery, where one case is 5.9 points and the
early curve is pure noise. It restored step-1000 weights at val loss 0.239.

The full run reached val 0.0709 - **3.4x better** - and the battery's first real
signal did not appear until **step 7,000**, long past where patience-4 had
already given up. Use `--eval-every 1000 --eval-patience 10`; it also halves the
eval overhead, which is ~2 minutes per call and otherwise exceeds the training
time.

## Running one: scripts/run/

The operational knowledge in this file is executable now. See
`scripts/run/README.md`.

```bash
scripts/run/00_preflight.sh      # linger, power caps, venv, disk, caches, ollama
scripts/run/02_build_dataset.sh  # ~20 min, ~6 GB
scripts/run/03_train.sh          # ~89 min at 40k iters, one GPU, detached
scripts/run/04_eval.sh checkpoints_long_M/agent_best.pt
```

`01_gen_pool.sh` widens the pool with the Ollama teacher and merges rather than
overwriting; it invalidates every dataset cache, so rebuild after it.
`03_train_segmented.sh` runs in timed segments. Everything is overridable by
environment variable.

`scripts/build_dataset.py` builds a cache without training and writes a `.json`
sidecar recording the parameters, so no future cache repeats the situation where
`instruct_1d39780d71d5d7f5.pt` could only be reused by path.

## Longer training was the lever, not more data

12,000 iterations was 0.77 epochs: 449M of 584M tokens, ~6 tokens per parameter
where a 74.8M model wants nearer 20. Validation loss was still falling
monotonically when the run ended, and train 0.0747 sat level with val 0.0709 -
no generalization gap at all.

40,000 iterations (89 min, one GPU):

| | 12k | 40k |
|---|---|---|
| best val | 0.0709 | **0.0434** |
| held-out | 22/100 | **35/100** |
| 1 call | 7/20 | 15/20 |
| 2 calls | 9/12 | 10/12 |
| 3 calls | 5/19 | 6/19 |
| 4+ calls | 1/49 | 4/49 |

Every bucket improved. The largest gain is single-call (7/20 -> 15/20), which is
what `diag_deep_chains.py` predicted: the failure was grounding on the *first*
call, not compounding with depth.

Validation loss is now flattening (0.0460 @ 28k, 0.0434 @ 38.5k, 0.0435 @ 39.5k),
so further steps have reached diminishing returns and **data is the next lever**.

Constrained decoding was tested against this and is not the answer: 24/100 vs
22/100, with the 4+ bucket unchanged at 1/49. Masking an invalid tool name just
makes the model pick a valid wrong one.

## Pool expansion

`data/ollama_pool_merged.json` is ready but **not yet promoted**, because
swapping it invalidates the dataset cache. Generated in 9.2 min with zero errors:

| axis | before | generated | merged |
|---|---|---|---|
| thoughts | 1,212 | 2,120 | **3,319** |
| paraphrases | 175 | 376 | **518** |
| facts | 682 | 1,260 | **1,678** |

Thoughts were the axis worth widening: 1,140 distinct across 323,715 uses in the
dataset (284x reuse), against 22,718 distinct user questions per 30,000 traces
(1.3x) and 144,342 distinct tool arguments. Questions and arguments were never
the thin part.

To adopt it:

```bash
cp data/ollama_pool_merged.json data/ollama_pool.json
scripts/run/02_build_dataset.sh && scripts/run/03_train.sh
```
