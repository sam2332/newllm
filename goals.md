# Project Goals

## Current goals

1. Build a minimal but correct transformer from scratch in PyTorch.
2. Implement modern LLM ideas: MLA, sparse attention, MoE, RoPE, SSM, hybrid layers, multi-token prediction, reasoning-time compute.
3. Add a working training + evaluation pipeline with synthetic data.
4. Scale to 8K token context on an RTX 5060 Ti.
5. Make sequence-level Mixture-of-Experts actually route to different "mini-minds" per story type.
6. Provide modern sampling (temperature, top-k, top-p, min-p, repetition penalty).

## Completed

- [x] Transformer from scratch
- [x] Modular file structure under `model/`
- [x] Modern feature modules (MLA, sparse, MoE, SSM, multi-token, RoPE, reasoning)
- [x] Smoke + feature tests
- [x] Basic training pipeline on sample stories
- [x] Big dataset + larger-context training scripts
- [x] CUDA 12.8 GPU setup for RTX 5060 Ti
- [x] Sampler with temperature/top-k/top-p/min-p/repetition penalty
- [x] Sequence-level MoE with load-balancing loss

## In progress

- [x] Persist all sequence MoE experiment results to `results/`
- [x] Build a tiny ReAct agent with tools (`calc`, `now`, `search_memory`)
- [x] Add explicit `BEGIN_THINK`/`END_THINK` internal reasoning to agent traces
- [x] Add checkpoint saving/loading and a chat/test CLI (`agent/chat.py`)
- [x] Train agentic transformer to basic tool-usage competence (accuracy > 70%)
- [x] Add curriculum: single-step → multi-step agent traces
- [x] Persist timestamped snapshots so no experiment is lost
- [x] Migrate agent tool calls to standard JSON format
- [x] Add fake `web_search` tool so the LLM must look up facts
- [ ] Make sequence MoE generate coherent continuations per topic
- [ ] Train 8K big model to convergence
- [ ] Add real validation metrics (perplexity, accuracy)
- [ ] Scale agent model toward 1B params with MoE/MLA/sparse attention

## Future goals

- [ ] Add KV-cache inference
- [ ] Quantization / 4-bit training support
- [ ] Load real text corpus
- [ ] Distributed training across multiple GPUs
- [ ] Export model to safetensors / ONNX
