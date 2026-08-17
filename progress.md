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

## Next expected steps

- Finish current agent training run and check tool-use accuracy.
- Iterate data/model until agent achieves >70% accuracy on held-out questions.
- Fix sequence MoE generation collapse.
- Add checkpointing.
- Run full 8K big training to convergence.
