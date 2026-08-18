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
# Watch the latest training run
python watch_agent.py --log agent_run_web_s.log

# Train a fresh S-size agent with JSON + web_search (writes to checkpoints_web/)
python -u -m agent.train_agent --samples 20000 --iters 4000 --size S --curriculum --no-resume --checkpoint-dir checkpoints_web > agent_run_web_s.log 2>&1

# Train the same model with a real two-stage curriculum
#   (single-step foundation, then multi-step + web-math)
scripts\train_web_agent_s_curriculum.ps1

# Same two-stage curriculum but with digit-word expansion for arithmetic
# copying (experimental, see sidetrack_ideas.md)
scripts\train_web_agent_s_curriculum_words.ps1

# Train our current best small agentic model (25M params, curriculum, multi-step)
python -u -m agent.train_agent --samples 20000 --iters 4000 --size S --curriculum --no-resume > agent_run_best_s.log 2>&1

# Train the larger 151M agentic model (takes ~50 min on RTX 5060 Ti)
python -u -m agent.train_agent --samples 100000 --iters 10000 --size L --curriculum --no-resume > agent_run_best_l.log 2>&1

# Chat with the best checkpoint
python -m agent.chat --mode chat

# Chat with the web-search variant
python -m agent.chat --mode chat --checkpoint checkpoints_web/agent_best.pt

# Run the tool-use test battery
python -m agent.chat --mode test

# Run the web-search test battery
python -m agent.chat --mode test --checkpoint checkpoints_web/agent_best.pt

# Regression gate: fail promotion if the candidate does not pass thresholds
# AND win at least 3 out of 5 metric comparisons against the current best.
python -m agent.promote --dry-run checkpoints_web/agent_best.pt --best checkpoints/agent_best.pt --min-accuracy 0.85 --min-numeric 0.75 --min-exact 0.85

# Promote a passing checkpoint to the workspace best
python -m agent.promote checkpoints_web/agent_best.pt --best checkpoints/agent_best.pt --min-accuracy 0.85 --min-numeric 0.75 --min-exact 0.85

# PyTest regression suite (base gate used by CI/default tests)
python -m pytest agent/test_agent_regression.py -v

# PyTest regression suite including web_search cases
python -m pytest agent/test_agent_regression.py -v --web

# Unit tests for parsing/tool execution (no model needed)
python -m pytest agent/test_agent_loop.py -v
```

## JSON Agent Protocol (Current)

The agent is being migrated from legacy ReAct text to tagged JSON messages.
This is a new model family; legacy checkpoints do not work with the new loop.

```text
<user>What is 12 + 8?</user>
<assistant>{"thought":"I need the calculator.","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>
<tool name=calc>20</tool>
<assistant>{"thought":"The calculation is complete.","response":"20"}</assistant>
```

- Every assistant turn must be valid JSON.
- Every assistant object has a required `thought` string.
- It has exactly one of `tool_call` or `response`.
- `tool_call` uses `{ "name": string, "arguments": object }`.
- `agent/ollama_cot.py` can collect short reasoning examples through Ollama at `http://localhost:11434/api/chat`; default model name is `kimi2.7-cloud`.

Current JSON migration validation:

```powershell
python smoke_test.py
python -m pytest agent/test_agent_loop.py -v
```

Do not run JSON diagnostic training into `checkpoints/agent_best.pt`. Use a dedicated directory, for example:

```powershell
python -u -m agent.train_agent --samples 20000 --iters 4000 --size S --math-only --no-resume --checkpoint-dir checkpoints_json --max-len 768
```

Read [HANDOFF.md](HANDOFF.md) before continuing the migration.

- Defines `calc`, `now`, `search_memory`, `web_search`, `finish` tools in `agent/tools.py`.
- Synthetic JSON-message training data lives in `agent/agent_dataset.py`.
- Inference loop with tool execution lives in `agent/agent_loop.py`.
- Trained model is now expected to emit JSON `thought` + `tool_call`/`response` objects.
- Best checkpoints are promoted to `checkpoints/agent_best.pt`; old artifacts live in `archive/`.
- `agent/promote.py` runs a regression gate before promotion. A candidate must pass absolute thresholds AND win at least 3 out of 5 metric comparisons against the current best (`checkpoints/agent_best.pt`). The old best is kept as `*_prev.pt`.
- `agent/test_agent_regression.py` is the matching PyTest suite.
- `training/trainer.py` now saves/restores the best-validation-loss checkpoint instead of the final-iteration weights.

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
