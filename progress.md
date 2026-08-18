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
- Launched M-size math-only diagnostic (`scripts/train_web_agent_m_math_only.ps1`) to test whether ~60M params can learn arithmetic copying.
- Updated `agent/promote.py` to implement a 5-check competition against the current best checkpoint:
  - Checks: overall accuracy, numeric accuracy, exact accuracy, required-question pass rate, robustness composite.
  - Candidate must pass absolute thresholds AND win ≥3/5 checks to promote.
- Updated `AGENTS.md`, `mistakes.md`, `sidetrack_ideas.md` with each iteration.

## Next expected steps

- Wait for M-size math-only diagnostic to finish and evaluate whether the larger model learns arithmetic.
- If M-size succeeds, build a full M-size JSON+web curriculum run on top of the math foundation.
- If M-size fails, try a structural fix for number copying (pointer/constrained decoding, separate number encoder, or BPE tokenizer).
- Promote the new JSON+web model to `checkpoints/agent_best.pt` once it wins the 5-check competition.
- Decide whether to resume/restart the L-size curriculum run with lower effective batch size or gradient checkpointing.
- Scale to XL/1B parameters once L-size multi-step behaviour is solid.
- Fix sequence MoE generation collapse.
- Add KV-cache, real corpus loading, and validation metrics.
- Run full 8K big training to convergence.
