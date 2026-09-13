# Progress Log

## 2026-08-17

- Created initial transformer in single file.
- Split project into `model/` modules: attention, MLA, sparse, MoE, feed-forward, SSM, RoPE, text encoder/decoder.
- Added `reasoning.py`, `sampling.py`, `training/` package.
- Implemented all modern feature tests; `python test_modern_features.py` passes.
- Built `live_training_test.py` and trained 6 variants on sample data using GPU.
- Installed CUDA 12.8 torch for RTX 5060 Ti.
- Added `big_training_test.py` and `training_8k_big.py` for 8K context experiments.
- Created sequence-level MoE with topic clustering and load-balancing auxiliary loss.
- Added sampler with temperature, top-k, top-p, min-p, repetition penalty.
- Created agent docs: `AGENTS.md`, `goals.md`, `progress.md`, `mistakes.md`.

## Result logging

- Added `results/` directory and `results_logger.py`.
- `train_sequence_moe.py` now writes timestamped JSON with loss curves, topic routing, and sample generations.
- First logged run: `results/sequence_moe_20260817_204127.json` — final_loss=0.9178, time=139.8s, generations still degenerate.

## Agentic tool-use experiments

- Added `agent/` package: `tools.py`, `agent_dataset.py`, `agent_loop.py`, `train_agent.py`.
- Defined tools: `calc`, `now`, `search_memory`, `finish`.
- Training traces use ReAct format with explicit `BEGIN_THINK`/`END_THINK` internal reasoning block.
- Scaled to 100k traces and a 12.8M-parameter transformer (d_model=512, 6 layers, 8 heads).
- In-progress run: `results/agent_*.json` — first large-scale run currently training.

## Agentic tool-use update

- Fixed numerical stability (untied embeddings, fp32 decoder logits, `torch.amp`, scaled init, causal mask) so a 151M-parameter agent trains without loss explosion.
- Added causal mask to `model/transformer.py` after discovering the model attended to future tokens and generated gibberish despite low loss.
- Rewrote `agent/agent_loop.py` for true interactive multi-step ReAct execution: it pauses on each `Action:`, runs the real tool, appends the observation, retokenises context, and continues until `END_THINK`.
- Fixed `_safe_eval` fallback for malformed expressions and improved multi-hop dataset templates (`agent/agent_dataset.py`).
- Added checkpoint saving/loading and `agent/chat.py` CLI for chat/test modes.
- Trained an S-size (25M) curriculum model on 20k mixed single + multi-step traces for 4k iterations; it achieves 8/8 on the test battery, including multi-hop questions.
- Promoted `agent_s_multi_4k_v2.pt` to `agent_best.pt`; chat CLI now answers 9/9 test questions correctly.
- Launched an L-size (151M) curriculum run (`agent_run_v3f.log`) with proper file logging; it is currently training (loss dropping from ~5.7).

## Project cleanup

- Created `archive/` with `checkpoints/`, `results/`, and `logs/` subdirectories.
- Moved all older checkpoints, results JSONs, and training logs into `archive/`.
- Kept only the current best checkpoint (`agent_best.pt` / `agent_s_multi_4k_v2.pt`), the two most recent result JSONs, and the active `agent_run_v3f.log` in the working directories.
- Added `scripts/` containing PowerShell helpers: `train_best_agent_s.ps1`, `train_best_agent_l.ps1`, `watch_agent.ps1`, `chat_agent.ps1`, `test_agent.ps1`.
- Added `CLEANUP.md` documenting what is kept vs archived.
- Updated `AGENTS.md` quick commands to use the new helper scripts and file-logging patterns.
- Updated `.gitignore` so `results/` are preserved; only logs, pyc files, and `.venv/` are ignored.

## JSON tool calls + web_search

- Migrated agent tool calls from bracket format to JSON:
  - `agent/tools.py` now declares JSON schemas for `calc`, `now`, `search_memory`, `web_search`, `finish`.
  - `agent/agent_dataset.py` emits `Action: {"tool": ..., "args": ...}` and includes web + multi-hop web/math traces.
  - `agent/agent_loop.py` parses JSON calls first, falls back to legacy bracket format.
- Added a fake `web_search` tool with a static knowledge base (`WEB_KB`) so the LLM must look up facts not in parametric memory.
- Added web_search + web-then-math evaluation cases in `agent/train_agent.py`.
- Created `scripts/train_web_agent_s.ps1` to launch a fresh S-size JSON+web run into a separate `checkpoints_web/` directory so `agent_best.pt` is preserved.
- Updated [AGENTS.md](AGENTS.md) with JSON/web chat and test commands.

## L-size curriculum stall

- `agent_run_v3f.log` ended at iter 5122/10000 (~51%); loss was healthy (~0.05) but throughput collapsed from 3.2 it/s to 6-7 s/it before the process disappeared. Recorded in [mistakes.md](mistakes.md) as a suspected OOM/thermal or Windows background-kill event.

## Regression gate + trainer best-val update

- Fixed an fp16 overflow in `model/mla_attention.py`: `masked_fill(mask == 0, -1e9)` cannot be cast to `Half`; changed to `-1e4` to match `model/attention.py` and `model/sparse_attention.py`.
- Updated `training/trainer.py` to snapshot the best-validation-loss weights during training and restore them at the end, with a new `save_best_val()` helper.
- Updated `agent/train_agent.py` so the saved `agent_best.pt` is always the (restored) best-validation checkpoint.
- Created a regression gate in `agent/test_agent_regression.py` with configurable accuracy thresholds and a core-required-question subset so the restored legacy checkpoint passes the default base gate.
- Verified `smoke_test.py`, `live_training_test.py`, and the regression suite against `checkpoints/agent_best.pt` all pass.

## Web agent curriculum attempt and regression gate result

- First full S-size JSON+web run (`scripts/train_web_agent_s.ps1`, 4k iters) completed with healthy loss (best train 0.0935, best val 0.1182) but the checkpoint only scored 23.53% on the eval battery and failed basic arithmetic.
- Root cause: the 25M model learned the format but could not reliably copy question numbers into `calc` arguments when trained directly on the full mixed distribution.
- Implemented a true two-stage curriculum:
  - Stage 1: single-step traces (math, memory, date, web facts) with higher LR (`3e-4`).
  - Stage 2: resume the same checkpoint on multi-step + web-math traces with lower LR (`1e-4`).
  - New script: `scripts/train_web_agent_s_curriculum.ps1`.
- Added `--lr` argument to `agent/train_agent.py` for curriculum-stage LR control.
- The two-stage curriculum still did not reach the gate; arithmetic copying remained poor (~23.53%).
- Implemented **digit-word expansion** as a new experiment:
  - Math questions, expressions, and answers are spelled out (`12 + 8` → `one two + eight`).
  - `agent/tools.py` collapses the words back to digits before `_safe_eval`.
  - `agent/agent_loop.py` also collapses calc arguments and final answers.
  - New script: `scripts/train_web_agent_s_curriculum_words.ps1`.
  - Sidetrack ideas and future experiments collected in [sidetrack_ideas.md](sidetrack_ideas.md).
- Added math-only diagnostic mode to `agent/train_agent.py` and ran an S-size math-only experiment.
  - Result: 5.88% accuracy; the 25M model cannot learn arithmetic copying even in isolation.
  - This shifts the hypothesis from "task mixture" to "capacity or representation mismatch".
- Launched M-size math-only diagnostic (`scripts/train_web_agent_m_math_only.ps1`); result was 5.88% accuracy — same as S-size. Capacity is not the bottleneck.
- Added single-digit math-only mode (`--single-digit-math`) to `agent/train_agent.py` and `agent/agent_dataset.py`, plus script `scripts/train_web_agent_s_math_digits.ps1` to test whether the model can copy single-token numbers.
- Updated `agent/promote.py` to implement a 5-check competition against the current best checkpoint:
  - Checks: overall accuracy, numeric accuracy, exact accuracy, required-question pass rate, robustness composite.
  - Candidate must pass absolute thresholds AND win ≥3/5 checks to promote.
- Updated `AGENTS.md`, `mistakes.md`, `sidetrack_ideas.md` with each iteration.

## Next expected steps

- Single-digit diagnostic showed the model can generate valid expressions but does not copy them from the question. Next experiment: compact math format `calc(expr) = ?` so the expression is a contiguous marked substring.
- If compact format succeeds, gradually reintroduce memory/date/web and move to a full M-size run.
- If compact format fails, add a pointer/constrained decoding mechanism or restructure the trace to reprint numbers near the expression slot.
- Promote the new JSON+web model to `checkpoints/agent_best.pt` once it wins the 5-check competition.
- Decide whether to resume/restart the L-size curriculum run with lower effective batch size or gradient checkpointing.
- Scale to XL/1B parameters once L-size multi-step behaviour is solid.
- Fix sequence MoE generation collapse.
- Add KV-cache, real corpus loading, and validation metrics.
- Run full 8K big training to convergence.

## JSON message-agent rewrite and handoff status

- The legacy `BEGIN_THINK`/`END_THINK` ReAct protocol was replaced in the working tree with a JSON-message protocol.
- The intended contract is:
  - User message: `<user>question</user>`.
  - Assistant tool turn: `<assistant>{"thought":"...","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>`.
  - Tool result: `<tool name=calc>20</tool>`.
  - Assistant final turn: `<assistant>{"thought":"...","response":"20"}</assistant>`.
- `thought` is required on every assistant turn. Each assistant object must contain exactly one of `tool_call` or `response`.
- Updated implementation files: `agent/agent_dataset.py`, `agent/agent_loop.py`, `agent/train_agent.py`, `agent/chat.py`, `agent/test_agent_loop.py`, and new `agent/ollama_cot.py`.
- `python smoke_test.py` passed after the rewrite.
- `python -m pytest agent/test_agent_loop.py -v` passed: 15 tests.
- Fixed an inference/training prompt mismatch: training traces begin directly with `<user>`, so the JSON loop now uses the same context rather than adding an unseen system prompt.
- Tightened parsing so an assistant object is rejected unless `thought` is a string and exactly one valid `tool_call` or string `response` is present.
- Full JSON multi-hop traces can reach 702 bytes. The JSON-agent trainer and model now default to `--max-len 768`; a unit test verifies sampled full traces fit that budget.
- A fast diagnostic run (`5000` samples, `1000` iterations, S-size, math-only, single-digit) reached low training loss (~0.11) but returned no answers in evaluation. This is not a model-quality result and must not be promoted.
- `checkpoints/agent_best.pt` was overwritten by that diagnostic run. It is an incompatible JSON-protocol checkpoint with 0% evaluation accuracy and is not a usable best model. Recover the previous promoted legacy checkpoint from `checkpoints/agent_best_prev.pt` or archive before running any legacy tests, but do not use it with the JSON loop.
- The legacy regression suite currently fails against the diagnostic checkpoint by design; it cannot be used as a JSON-protocol quality gate until a new checkpoint is trained.
- Handoff priority: make training and inference context serialization identical, retrain a JSON checkpoint, then re-baseline the regression/promotion gate. See `HANDOFF.md`.

## Assistant-only JSON agent redesign

- Added `agent/tokenizer.py`: shared byte tokenizer with atomic role tags, tool tags, and JSON keys (271 vocabulary entries).
- Added `agent/dataset.py`: loss applies only to assistant message tokens and the trailing EOT; user/tool messages are context only.
- Updated training and inference to use the same structural tokenizer. Chat and resume now reject legacy 256-vocabulary checkpoints explicitly.
- Added tokenizer, loss-mask, context-format, and legacy-checkpoint guard tests. `python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -v` passes 19 tests.

## Masked JSON math baseline

- Ran a fresh S-size masked JSON math-only training job: 20,000 traces, 3,000 iterations, `max_len=768`, output in `checkpoints_json_math/`.
- Training converged at best validation loss `0.0032`, but initial inference returned no answers despite low loss.
- Diagnosed and fixed two inference defects:
  - Atomic-token generation was sliced using a token index as a character index, which stripped the leading assistant JSON structure.
  - Inference omitted the newline delimiter that the training serializer places between `<user>` and `<assistant>` messages.
- After the fixes, the checkpoint emits valid JSON, invokes `calc`, and completes the tool loop. A held-out arithmetic slice scored 5/5: `12 + 8`, `15 * 4`, `7 * 6`, `91 - 37`, and `144 / 12`.
- Updated final-answer handling to use real tool output as the source of truth while retaining the model's final JSON string as `model_response`. `python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -v` now passes 20 tests.

## Grounded complex scenarios

- Replaced the old unsupported "string length" multi-hop template with chains that use actual numeric observations.
- Added three-tool scenario families: dependent calculator stages, web fact -> two calculations, calculator -> memory -> calculation, and memory + web -> calculation.
- Each generated tool observation is validated against `Toolbox`; a 10,000-trace sample had 3,344 three-tool traces, maximum length 746, and no trace above the 768-token context limit.
- `python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -v` passes 24 tests.

## Mixed-task train/test loop

- Full mixed training from random initialization (`checkpoints_json_full`, 4,000 iterations) reached 47.06% on the legacy built-in battery: memory/date and several web facts worked, but numeric grounding failed.
- Added `--init-checkpoint` so a fresh output directory can start from the validated JSON math checkpoint, and corrected the retired string-length evaluation case to use grounded chains.
- Curriculum from the math checkpoint (`checkpoints_json_curriculum`, 2,500 iterations, `lr=1e-4`) reached 100% numeric and date accuracy on the corrected battery but only 22.22% exact lookup accuracy. Conservative tool/key typo recovery raised total corrected-battery accuracy to 58.82%.
- Simple-task replay from that checkpoint (`checkpoints_json_replay`, 1,500 iterations, `lr=5e-5`) raised simple exact behavior but reduced multi-step arithmetic; it reached 52.94% on its built-in battery.
- Added `--math-replay-fraction` to mix math-only samples into curriculum data. A 50% replay run (`checkpoints_json_balanced`, 3,000 iterations) still scored only 35.29% on the corrected battery (71.43% numeric, 0% exact).
- Conclusion: the 25M S-size model cannot reliably retain exact web/memory tool arguments and multi-step arithmetic together with the current synthetic distribution. None of the mixed checkpoints are promotion candidates. Keep `checkpoints_json_math/agent_best.pt` as the validated arithmetic baseline.
- Focused JSON-agent suite now passes 27 tests.

## Current Decision

- Preserve `checkpoints_json_math/agent_best.pt` as the verified JSON arithmetic specialist.
- Do not promote `checkpoints_json_full`, `checkpoints_json_curriculum`, `checkpoints_json_replay`, or `checkpoints_json_balanced`; all fail the mixed-task standard.
- Stop dataset-only reweighting experiments for the S-size model. The next change should be architectural: constrained decoding for tool names/argument JSON, a copy/pointer path for user spans and tool values, or a larger model trained on the same masked JSON objective.

## Architecture rewrite (arch_version=2)

The Linux server had no PyTorch at all (the project was Windows-native), so the
environment was rebuilt first: `.venv` on Python 3.12 with torch 2.11.0+cu128.
Python 3.14 is the system default and has no torch wheels. Both GPUs verified:
RTX 4090 (sm_89, 24GB) and RTX 5090 (sm_120, 32GB), bf16 supported on both.

### Defects found in the original model code

1. `model/transformer_block.py` did not implement pre-norm. It ran
   `x = self.norm1(x)` and then `x = x + attn(x)`, which overwrites the residual
   stream instead of leaving it untouched. Correct pre-LN is `x = x + attn(norm1(x))`.
   Measured effect: not gradient starvation at init (layer0/layer11 gradient-norm
   ratio was 0.994 before and 1.011 after, essentially unchanged) but a large
   difference in what the model can learn - see the A/B below.
2. RoPE was applied once to the token embedding in `TextEncoder`, not per head to
   Q and K inside attention. The first normalization then largely removes it.
   Measured: total variation between the next-token distribution for the same
   bigram at two different absolute positions was 0.109 (v1) vs 0.021 (v2), so v2
   is ~5x more position-invariant, which is the signature of true relative
   encoding. This is the defect most likely behind the documented failure to copy
   exact tool arguments.
3. `Transformer._init_weights` branched on `isinstance(p, nn.Linear)` while
   iterating `named_parameters()`, where `p` is always a Tensor. Both the Linear
   and LayerNorm branches were unreachable, so no depth-scaled residual init was
   ever applied.
4. `agent/agent_loop.py` ran a full forward pass over the entire sequence for
   every generated token, and re-decoded and re-parsed the whole string each
   step - O(n^2) in GPU work and in Python.
5. Training used fp16 + GradScaler. Both GPUs support bf16 natively, which needs
   no loss scaling and cannot overflow the same way.

### Changes

`arch_version` selects the generation; `arch_version=1` reproduces the original
code path exactly so existing checkpoints stay loadable.

- true pre-norm residual stream (Xiong et al. 2020)
- RMSNorm (Zhang & Sennrich 2019), `model/norm.py`
- RoPE applied per head to Q/K (Su et al. 2021), `model/rope.py:apply_rope`
- QK-Norm (Chameleon 2024, Gemma-3, OLMo-2)
- SwiGLU feed-forward (Shazeer 2020), hidden width scaled 2/3 to hold parameters
- Grouped-Query Attention (Ainslie et al. 2023), default `n_kv_heads = n_heads/4`
- fused `F.scaled_dot_product_attention` plus an incremental KV cache
- depth-scaled residual init, `std = 0.02 / sqrt(2 * n_layers)`
- bf16 autocast, AdamW betas (0.9, 0.95), decay excluded from norms/biases/
  embeddings, and output z-loss 1e-4

### Measured A/B (identical data, seed, and token budget)

S-size, 20k single-step traces, 1200 iterations, effective batch 128:

| variant            | params | best val | battery | valid JSON | single-step |
|--------------------|--------|----------|---------|-----------|-------------|
| v1 original (fp16) | 25.5M  | 0.0910   | 47.1%   | 85.7%     | 62%         |
| v2 modern (bf16)   | 26.0M  | 0.0131   | 64.7%   | 100%      | 85%         |
| v2 + GQA (bf16)    | 22.8M  | 0.0137   | 64.7%   | 96.3%     | 85%         |

Validation loss fell 7x and battery accuracy rose from 47.1% to 64.7%, which
also exceeds the best previously recorded mixed-task result (58.82%).

Training throughput and memory, batch 16 x seq 768 on the 4090:

| variant              | tokens/s | ms/step | peak VRAM |
|----------------------|----------|---------|-----------|
| v1 original + fp16   | 134,152  | 91.6    | 7.34 GB   |
| v2 + bf16            | 221,837  | 55.4    | 4.30 GB   |
| v2 + GQA + bf16      | 259,507  | 47.4    | 4.03 GB   |

Note on generation speed: at this model size single-token decoding is bound by
Python and kernel-launch overhead, not by compute - throughput is flat across
context lengths 128 to 640. The KV cache therefore does not speed up decoding
here, and v2's extra per-layer operations make it slightly slower per step. The
cache matters for correctness and for larger models; the real wins at this scale
are the training numbers above.

### Correctness harness

`scripts/debug_arch.py` checks causality, KV-cache equivalence against a full
forward pass, RoPE relative-position behaviour, residual gradient health, and
throughput. KV-cache output matches a full forward pass to 1.3e-07.

## Constrained decoding

`agent/constrained.py` implements the state machine HANDOFF.md asked for. Logits
are masked to the tokens the assistant-JSON grammar allows, so structurally
invalid output and non-existent tool names are unrepresentable. Verified to
accept all real training message shapes and to reject an invalid tool name at
the first diverging character.

It did not change battery accuracy, because v2 already emits 100% valid JSON
greedily. It is a guarantee, not a quality gain, and is off by default
(`run_agent(..., constrained=True)`).

## Instruct / chat split

Two models from one architecture, differing in data distribution and context:

- instruct: single-turn task execution, `max_len=768`
- chat: multi-turn conversation, `max_len=2048`

`scripts/train_split.py --mode instruct|chat` trains either. `--init-checkpoint`
warm-starts one from the other and grows embedding rows rather than refusing a
checkpoint with a smaller vocabulary.

A blocker was fixed first: EOT appeared only once per trace, at the very end, so
the model learned end-of-episode and never end-of-turn - in a conversation it
would continue past its own answer and write the user's next message. Every
assistant turn that emits a final response now ends with a supervised EOT.

`agent/chat_dataset.py` generates conversations whose later turns are only
answerable from earlier ones: pronoun coreference, ellipsis, back-reference past
the most recent turn, topic switches, and turns needing no tool at all.
`agent/agent_loop.py:run_chat` carries context across turns at inference.

The tokenizer grew from 271 to 273 tokens (`<system>`, `</system>`). The
extension is append-only: ids 0-270 keep their meaning, so older checkpoints can
be grown instead of discarded.

## Ollama compatibility

`agent/ollama_format.py` converts between the internal protocol and Ollama's
chat API, verified against a live qwen3:30b-a3b-q8_0 instance. Two deviations
from a naive OpenAI assumption matter:

- `tool_calls` is a list of `{"id", "function": {"index", "name", "arguments"}}`,
  not a single `tool_call`
- `function.arguments` is a real JSON object, not a JSON-encoded string

Tool results return as `{"role": "tool", "tool_name": ..., "content": ...}`. The
internal format stays compact because a byte-level model pays per character;
`thought` maps to Ollama's `thinking` field so nothing is lost in a round trip.

## Data diversity

The existing generator was the real ceiling, not the architecture: 4000 traces
contain only 34 distinct `thought` strings, 12 web facts and 7 memory keys. That
is why validation loss reaches 0.0000 while held-out accuracy stalls - the model
memorizes 34 sentences.

`scripts/gen_data_ollama.py` drives both Ollama instances concurrently to
generate paraphrases, reasoning sentences and factual key/value pairs. The
teacher supplies language only. It is never asked to compute a result or to say
what a tool returned; every observation in every emitted trace comes from the
real `Toolbox`. `agent/rich_dataset.py` composes traces and verifies them - a
spot check of 559 observations found 0 mismatches.

## Self-inspection and sandboxed execution

`agent/repo_tools.py` (read-only): `list_files`, `read_file`, `search_code`,
`describe_symbol`, `repo_stats`. Confined to the repository root; path traversal
via `..` and absolute paths are both rejected.

`agent/sandbox_tools.py` (execution): `run_bash`, `run_python`, each in a fresh
throwaway Docker container with `--network none`, `--read-only`, tmpfs /tmp,
512MB, 1 CPU, 128 pids, all capabilities dropped, `no-new-privileges`, uid 65534,
and a wall-clock timeout. Verified: cannot reach the host Ollama, cannot resolve
DNS, cannot see the repo or checkpoints, fork bomb contained, 2GB allocation
OOM-killed, `sleep 300` killed at 20s.

Introspection and execution are deliberately separate tool sets. `repo_tools`
can read the code but not run anything; `sandbox_tools` can run code but cannot
see the code.

`scripts/gen_sandbox_pool.py` executes a task list once in real containers and
keeps only what succeeds (26/27; the `bc` task was correctly rejected as absent
from the slim image), so dataset generation never starts a container.

The model now routes across 11 tools.

## Evaluation

- `scripts/eval_agent.py` scores the instruct battery and reports parsed-JSON
  validity, tool-call rate, per-kind accuracy and, separately, single-step vs
  multi-hop accuracy, as HANDOFF.md asked.
- `scripts/eval_chat.py` scores multi-turn skills separately: coreference,
  ellipsis, back-reference, topic switch, and no-tool-needed.

The earlier "valid JSON" metric was wrong - it counted task completion, not JSON
validity. It now parses every `<assistant>` block in the trace and validates it
against the contract.

## Test status

`agent/test_agent_dataset.py` and `agent/test_agent_loop.py`: 27 passed.
`smoke_test.py` and `test_modern_features.py` (standard, MLA, sparse, MoE,
hybrid, multi-token, reasoning): all pass.

`agent/test_agent_regression.py`: 5 errors, pre-existing. It points at
`checkpoints/agent_best.pt`, the legacy 256-vocabulary checkpoint HANDOFF.md
already documents as unusable. It needs re-baselining against a current
checkpoint.
