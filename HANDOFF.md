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
