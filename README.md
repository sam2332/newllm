# newllm

A from-scratch transformer/LLM experimentation project in Python/PyTorch, built to test modern ideas like RoPE, MLA, sparse attention, MoE, SSM, multi-token prediction, and small agentic tool use.

## Quick start

1. **Install dependencies** (CUDA 12.8 for an RTX 5060 Ti):

   ```powershell
   pip install -r requirements.txt
   ```

2. **Run the smoke tests** to verify the install and model code:

   ```powershell
   python smoke_test.py
   python test_modern_features.py
   ```

3. **Chat with the current best agent** (or run the test battery):

   ```powershell
   python -m agent.chat --mode chat
   python -m agent.chat --mode test
   ```

## What this project includes

- `model/` — modular transformer components:
  - `transformer.py` — main model with optional MLA, sparse attention, MoE, hybrid SSM/attention, multi-token prediction, and RoPE.
  - `attention.py`, `mla_attention.py`, `sparse_attention.py`, `moe_layer.py`, `ssm_layer.py`, `multi_token_head.py`, `rope.py`, `self_check_rnn.py`
  - `text_encoder.py`, `text_decoder.py`
- `training/` — data loaders, trainer, and device selection.
  - `dataset.py`, `big_dataset.py` — synthetic story generation
  - `trainer.py` — training loop with AdamW, gradient clipping, cosine LR, AMP, validation
  - `device_utils.py` — auto-picks `cuda`, `mps`, or `cpu`
- `agent/` — a tiny ReAct-style agent that uses tools.
  - `tools.py` — `calc`, `now`, `search_memory`, `finish`
  - `agent_dataset.py` — synthetic ReAct traces with `BEGIN_THINK`/`END_THINK` blocks
  - `agent_loop.py` — interactive multi-step inference loop
  - `train_agent.py` — training script with size presets S/M/L/XL
  - `chat.py` — chat and test CLI
- `sampling.py` — temperature, top-k, top-p, min-p, repetition penalty.
- `reasoning.py` — inference-time reasoning helper.
- `watch_agent.py` — tail the latest agent training log.
- `scripts/` — ready-to-run PowerShell helpers for training and testing the agent.

## Reproducing the best agent

The current best checkpoint is `checkpoints/agent_best.pt` (a 25M-parameter S-size model trained on a curriculum of single-step and multi-step ReAct traces). To reproduce it:

```powershell
# Small model (~13 minutes on RTX 5060 Ti)
.\scripts\train_best_agent_s.ps1

# Larger 151M model (~50 minutes)
.\scripts\train_best_agent_l.ps1
```

Or run directly:

```powershell
python -u -m agent.train_agent --samples 20000 --iters 4000 --size S --curriculum --no-resume > agent_run_best_s.log 2>&1
```

## Project layout conventions

- Keep caveman-style comments (`# grug: ...`) when explaining big ideas.
- Add new modules under `model/` and wire them through `Transformer.__init__`.
- Maintain backward-compatible defaults so `smoke_test.py` keeps passing.
- Every training run writes a timestamped JSON file to `results/` — keep these files; they are the permanent record of experiments.
- Old checkpoints, results, and logs are moved to `archive/` once they are no longer the active experiment.

## Common pitfalls

- `torch` must be CUDA-enabled. If `torch.cuda.is_available()` is `False`, reinstall with the index from `requirements.txt`.
- Long-context jobs (8K+ tokens) need a small batch size (1-2) and sparse/MLA attention on a 16 GB GPU.
- `Subset` datasets do not expose `collate_pad`; the trainer reads it from the underlying `StoryDataset`.
- Modern MoE routing needs a load-balancing/auxiliary loss or all sequences collapse to one expert.

## Useful commands

| Task | Command |
|------|---------|
| Smoke test | `python smoke_test.py` |
| Modern feature tests | `python test_modern_features.py` |
| Train best small agent | `.\scripts\train_best_agent_s.ps1` |
| Train best large agent | `.\scripts\train_best_agent_l.ps1` |
| Watch training log | `python watch_agent.py --log agent_run_best_l.log` |
| Chat with agent | `python -m agent.chat --mode chat` |
| Test agent tool use | `python -m agent.chat --mode test` |

## Documentation

- [AGENTS.md](AGENTS.md) — detailed guide for the agentic experiments.
- [mistakes.md](mistakes.md) — lessons learned and pitfalls.
- [goals.md](goals.md) — project goals and current progress.
- [progress.md](progress.md) — day-by-day progress log.
- [CLEANUP.md](CLEANUP.md) — what files are kept vs archived.
