# AI Agent Guide for newllm

This is a from-scratch transformer/LLM experimentation project in Python/PyTorch.

## Quick commands

Run tests after any model or training change:

```powershell
python smoke_test.py
python test_modern_features.py
python live_training_test.py
python train_sequence_moe.py
```

Long-context benchmark:

```powershell
python training_8k_big.py
```

Result logging:

```powershell
python train_sequence_moe.py
python -m agent.train_agent
```

Every run writes a timestamped JSON file to `results/`. Keep these files — they are the permanent record of experiments.

Agentic tool-use experiments:

```powershell
python -m agent.train_agent
```

- Defines `calc`, `now`, `search_memory`, `finish` tools in `agent/tools.py`.
- Synthetic ReAct training data lives in `agent/agent_dataset.py`.
- Inference loop with tool execution lives in `agent/agent_loop.py`.
- Trained model learns to emit `BEGIN_THINK` ... `END_THINK` internal reasoning plus tool calls.

## Project layout

- `model/` — modular transformer components.
  - `transformer.py` — main model; supports MLA, sparse attention, MoE, hybrid SSM/attention, multi-token prediction, RoPE.
  - `mla_attention.py`, `sparse_attention.py`, `moe_layer.py`, `ssm_layer.py`, `multi_token_head.py`, `rope.py`, `self_check_rnn.py`
  - `text_encoder.py`, `text_decoder.py`
- `training/` — datasets, trainer, device selection.
  - `dataset.py`, `big_dataset.py` — synthetic story generation
  - `trainer.py` — training loop with AdamW, gradient clipping
  - `device_utils.py` — auto-picks `cuda`, `mps`, or `cpu`
- `reasoning.py` — inference-time reasoning helper.
- `sampling.py` — temperature, top-k, top-p, min-p, repetition penalty.
- `requirements.txt` — pinned for CUDA 12.8 / RTX 5060 Ti.

## Conventions

- Keep caveman-style comments inside code (`# grug: ...`) when explaining big ideas.
- Add a new module under `model/` and wire it through `Transformer` constructor in `model/transformer.py`.
- Maintain backward-compatible defaults so `smoke_test.py` keeps passing.
- New training scripts live in root and accept `max_iters`, `batch_size`, `device` overrides.
- Model outputs may be `logits` or `(logits, aux_loss)` tuples; trainer handles both.
- **Always update [mistakes.md](mistakes.md)** after fixing any non-trivial bug or hitting a pitfall. Future agents (including this one) must read it before making changes.

## Common pitfalls

- `torch` must be CUDA-enabled. If `torch.cuda.is_available()` is `False`, reinstall with the index from `requirements.txt`.
- Long-context jobs (8K+) need small batch size (1-2) and sparse/MLA attention on a 16GB GPU.
- `Subset` datasets do not expose `collate_pad`; always read it from the underlying `StoryDataset`.
- Modern MoE routing needs a load-balancing/auxiliary loss or all sequences collapse to one expert.

## See also

- [requirements.txt](requirements.txt) for environment setup
- [smoke_test.py](smoke_test.py) for the simplest forward-pass check
- [test_modern_features.py](test_modern_features.py) for feature coverage
