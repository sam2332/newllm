# newllm

A from-scratch transformer/LLM experimentation project in Python/PyTorch, built to test modern ideas like RoPE, MLA, sparse attention, MoE, SSM, multi-token prediction, and small agentic tool use.

## Quick start

### Linux (current server: RTX 4090 + RTX 5090)

PyTorch has no wheels for Python 3.14, so use 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install tqdm numpy pytest requests
```

Use `.venv/bin/python` for everything below.

### Windows

```powershell
pip install -r requirements.txt
```

### Verify

```bash
.venv/bin/python smoke_test.py
.venv/bin/python test_modern_features.py
.venv/bin/python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -q
.venv/bin/python scripts/debug_arch.py     # architecture correctness + speed
```

### Use the agent

```bash
.venv/bin/python -m agent.chat --mode chat
.venv/bin/python -m agent.chat --mode test
```

## Architecture versions

`arch_version` selects the model generation. **New models default to 2.**

- **1** — the original code path, kept so existing checkpoints still load.
- **2** — true pre-norm residuals, RMSNorm, per-head RoPE on Q/K, QK-Norm,
  SwiGLU, grouped-query attention, fused Flash attention with a KV cache,
  depth-scaled init, bf16, and output z-loss.

On identical data, seed and token budget, v2 cut validation loss ~7x and raised
battery accuracy from 47.1% to 64.7% at S size. At M size on mixed data the
model reaches 88.2%, with 100% on 2-hop and 3-hop tool chains.
See [progress.md](progress.md) for the measurements and the defects they fixed.

## Training the two models

The agent is split into an instruct model (single-turn task execution) and a
chat model (multi-turn conversation with coreference).

```bash
.venv/bin/python scripts/train_split.py --mode instruct --size M   # max_len 768
.venv/bin/python scripts/train_split.py --mode chat --size M       # max_len 2048
```

Evaluate them separately:

```bash
.venv/bin/python scripts/eval_agent.py checkpoints_instruct_M/agent_best.pt -v
.venv/bin/python scripts/eval_chat.py  checkpoints_chat_M/agent_best.pt -v
```

## Training data

The hand-written generator produces only 34 distinct reasoning sentences, 12
web facts and 7 memory keys, so a model trained on it memorizes rather than
generalizes. `scripts/gen_data_ollama.py` uses the local Ollama instances to
generate language diversity instead:

```bash
.venv/bin/python scripts/gen_data_ollama.py --out data/ollama_pool.json
.venv/bin/python scripts/gen_sandbox_pool.py        # verified sandbox outputs
```

**The teacher model never produces a tool result.** It supplies question
phrasings, reasoning sentences and factual key/value pairs only; every
observation in every trace comes from the real `Toolbox` and is verified.
Preserve this invariant or the dataset stops being ground truth.

## Tools

The agent routes across 11 tools:

| module | tools | notes |
|--------|-------|-------|
| `agent/tools.py` | `calc`, `now`, `search_memory`, `web_search`, `finish` | accepts injected memory/KB |
| `agent/repo_tools.py` | `list_files`, `read_file`, `search_code`, `describe_symbol`, `repo_stats` | read-only, confined to the repo root |
| `agent/sandbox_tools.py` | `run_bash`, `run_python` | isolated Docker container |

Introspection and execution are deliberately separate: `repo_tools` reads code
but runs nothing, `sandbox_tools` runs code but cannot see the repo.

Sandbox containers run with `--network none`, `--read-only`, tmpfs `/tmp`,
512MB, 1 CPU, 128 pids, all capabilities dropped, `no-new-privileges`, uid
65534, and a wall-clock timeout.

## Ollama-compatible serving

```bash
.venv/bin/python scripts/serve_ollama_compat.py \
    --checkpoint checkpoints_instruct_M/agent_best.pt --port 11500
```

Any client that already speaks to Ollama can point at that port.
`agent/ollama_format.py` handles the conversion and is verified against a live
Ollama instance.

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
- [LEADERBOARD.md](LEADERBOARD.md) is refreshed on each saved result and records model parameters, architecture, training budget, loss, evaluation splits, runtime, checkpoints, and rank.
- Old checkpoints, results, and logs are moved to `archive/` once they are no longer the active experiment.

## Common pitfalls

- `torch` must be CUDA-enabled. If `torch.cuda.is_available()` is `False`, reinstall with the index from `requirements.txt`.
- Python 3.14 has no torch wheels. Use 3.12.
- The tokenizer's special-token list is **append-only**. Ids 0-270 must keep their meaning so older checkpoints stay loadable. When loading a checkpoint whose vocabulary is smaller than the current one, build the tokenizer with `AgentTokenizer(max_vocab=<checkpoint vocab>)` (`agent.chat.load_tokenizer_for` does this for you); otherwise the tokenizer can emit ids the model has no embedding row for, which surfaces as a CUDA device-side assert that poisons the process.
- Generation at this model size is bound by Python and kernel-launch overhead, not compute. Throughput is flat across context lengths, so the KV cache does not speed up decoding here.
- Long-context jobs (8K+ tokens) need a small batch size (1-2) and sparse/MLA attention on a 16 GB GPU.
- `Subset` datasets do not expose `collate_pad`; the trainer reads it from the underlying `StoryDataset`.
- Modern MoE routing needs a load-balancing/auxiliary loss or all sequences collapse to one expert.

## Useful commands

| Task | Command |
|------|---------|
| Smoke test | `.venv/bin/python smoke_test.py` |
| Modern feature tests | `.venv/bin/python test_modern_features.py` |
| Architecture correctness + speed | `.venv/bin/python scripts/debug_arch.py` |
| Unit tests | `.venv/bin/python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -q` |
| Regression gate | `.venv/bin/python -m pytest agent/test_agent_regression.py -q` |
| Stricter web gate | `.venv/bin/python -m pytest agent/test_agent_regression.py -q --web` |
| Train instruct model | `.venv/bin/python scripts/train_split.py --mode instruct --size M` |
| Train chat model | `.venv/bin/python scripts/train_split.py --mode chat --size M` |
| Evaluate instruct | `.venv/bin/python scripts/eval_agent.py <ckpt> -v` |
| Evaluate chat | `.venv/bin/python scripts/eval_chat.py <ckpt> -v` |
| Generate diverse data | `.venv/bin/python scripts/gen_data_ollama.py` |
| Serve Ollama-compatible API | `.venv/bin/python scripts/serve_ollama_compat.py --checkpoint <ckpt>` |
| Chat with agent | `.venv/bin/python -m agent.chat --mode chat` |
| Test agent tool use | `.venv/bin/python -m agent.chat --mode test` |

## Documentation

- [AGENTS.md](AGENTS.md) — detailed guide for the agentic experiments.
- [mistakes.md](mistakes.md) — lessons learned and pitfalls.
- [goals.md](goals.md) — project goals and current progress.
- [progress.md](progress.md) — day-by-day progress log.
- [LEADERBOARD.md](LEADERBOARD.md) — newest-first ranking of saved experiment results.
- [CLEANUP.md](CLEANUP.md) — what files are kept vs archived.
