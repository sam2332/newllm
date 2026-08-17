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

## Next expected steps

- Stabilize sequence MoE generation.
- Add checkpointing.
- Run full 8K big training to convergence.
