# newllm

A from-scratch transformer in Python/PyTorch, trained to use tools through a JSON
protocol. Byte-level, 74.8M parameters, no pretrained weights.

The goal is **the smallest model that can accurately do tools and chat, with some
roleplay, usable from Ollama** - see [goals.md](goals.md).

Current state, honestly: tool use works and scores **35/100** on held-out traces;
chat is trained-but-never-run; roleplay does not exist yet. Full detail in
[HANDOFF.md](HANDOFF.md).

## Quick start

```bash
python3.12 -m venv .venv       # torch has no 3.14 wheels; system python is 3.14
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install tqdm numpy pytest requests

scripts/run/00_preflight.sh    # checks GPU caps, systemd linger, venv, caches
```

Use `.venv/bin/python` for everything.

### Train and evaluate

```bash
scripts/run/02_build_dataset.sh                            # ~20 min, ~6 GB
scripts/run/03_train.sh                                    # ~89 min, one GPU
scripts/run/04_eval.sh checkpoints_long_M/agent_best.pt    # ~10 min
```

Everything is overridable by environment variable
(`ITERS=80000 SIZE=S scripts/run/03_train.sh`). Read
[scripts/run/README.md](scripts/run/README.md) before a first run - it documents
two failure modes that silently killed runs for a night, and why the defaults
are what they are.

### Talk to it

```bash
.venv/bin/python -m agent.chat --mode chat
.venv/bin/python scripts/serve_ollama_compat.py \
    --checkpoint checkpoints_long_M/agent_best.pt --port 11500
```

The second serves `/api/chat`, `/api/tags` and `/api/show`, so any Ollama client
can point at it. Native GGUF loading is a stretch goal, not done.

### Verify the build

```bash
.venv/bin/python smoke_test.py
.venv/bin/python test_modern_features.py
.venv/bin/python scripts/debug_arch.py    # causality, KV cache, RoPE, throughput
.venv/bin/python -m pytest agent/test_agent_dataset.py agent/test_agent_loop.py -q
```

## How it works

Tool schemas are passed in the context per request, the way real function-calling
APIs do it:

```
<system>{"tools":[{"name":"...","description":"...","parameters":{...}}]}</system>
<user>What is 12 + 8?</user>
<assistant>{"thought":"...","tool_call":{"name":"<a name from the schema>",...}}
<tool name=...>20</tool>
<assistant>{"thought":"...","response":"20"}
```

Training randomizes tool names, descriptions, ordering and adds distractor tools
on every trace, so reading the schema is the only strategy that works. Memorizing
names is useless by construction - and that was a real failure: renaming `calc`
once took the old model from 88.2% to 0.0%.

Loss applies only to assistant spans. Every observation comes from the real
toolbox, never from a teacher model.

## Layout

| path | what |
|---|---|
| `model/` | transformer and its modules (RoPE, MLA, MoE, SSM, sparse attn) |
| `agent/` | protocol, tokenizer, tools, dataset generation, agent loop |
| `training/` | trainer, resume, LR schedule |
| `scripts/` | training, evaluation, diagnostics, data generation |
| `scripts/run/` | **runnable pipeline + setup guide** |
| `data/` | tool pools (tracked); `data/cache/` holds built datasets (ignored) |
| `archive/notes/` | superseded working notes, kept for the reasoning |

## Evaluation

Report `scripts/eval_random.py` - random held-out traces from the training
distribution. The 17-case battery in `scripts/eval_agent.py` is a smoke test:
one case is worth 5.9 points and 13 of 17 are single-hop toy questions. It is
deliberately **off** as a training signal, having twice caused real damage.

```
35/100 held-out, by tool calls needed:
  1 call   15/20      2 calls  10/12
  3 calls   6/19      4+ calls  4/49   <- half the distribution
```

## Tools

11 tools in three isolation tiers, deliberately kept apart:

- **plain** - `calc`, `now`, `search_memory`, `web_search`, `finish`
- **read-only repo** - `list_files`, `read_file`, `search_code`,
  `describe_symbol`, `repo_stats`. Confined to the repo root.
- **sandboxed execution** - `run_bash`, `run_python` in a throwaway container
  with no network, read-only rootfs, dropped capabilities, and a timeout. It
  cannot see the repo.

Introspection reads but never runs; execution runs but never sees. Do not merge
them.
