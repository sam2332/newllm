# newllm

A from-scratch transformer in Python/PyTorch, trained to use tools through a JSON
protocol. Byte-level, 74.8M parameters, no pretrained weights.

The goal is **the smallest model that can accurately do tools and chat, with some
roleplay, usable from Ollama** - see [goals.md](goals.md).

Current state, honestly: tool use works and scores **41/100** on held-out traces;
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
.venv/bin/python scripts/serve_ollama.py \
    --checkpoint checkpoints_long_M/agent_best.pt --port 11500
```

The second is an Ollama server with standard semantics: you define the tools,
the model returns `tool_calls`, you execute them and send back `role: "tool"`.
The official client works unmodified:

```python
import ollama
client = ollama.Client(host="http://127.0.0.1:11500")
tools = [{"type": "function", "function": {
    "name": "add_numbers", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
messages = [{"role": "system", "content": "You are Zed."},
            {"role": "user", "content": "What is 12 + 8?"}]
r = client.chat(model="newllm-agent", messages=messages, tools=tools)
for call in r.message.tool_calls or []:
    args = call.function.arguments
    messages += [r.message, {"role": "tool", "tool_name": call.function.name,
                             "content": str(args["a"] + args["b"])}]
    r = client.chat(model="newllm-agent", messages=messages, tools=tools)
print(r.message.content)
```

`OLLAMA_HOST=http://127.0.0.1:11500 ollama run newllm-agent` and OpenAI clients
pointed at `http://127.0.0.1:11500/v1` work too. `stream=True` returns NDJSON
deltas. Verify all of it with `scripts/test_ollama_client.py`. Native GGUF
loading is the next track, not done.

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

## Progress notifications (Discord)

Generation, training and evaluation all run detached for hours, often across
both GPUs and more than one machine. Point them at a Discord webhook and the
milestones arrive in a chat room instead of a log you have to ssh in and tail.

```bash
cp config.example.json config.json     # then paste your webhook URL into it
.venv/bin/python scripts/notify_test.py
```

```json
{
  "discord": {
    "webhook_url": "https://discord.com/api/webhooks/...",
    "username": "newllm",
    "enabled": true,
    "min_seconds_between": 2.0
  }
}
```

`config.json` is gitignored, because a webhook URL is a credential: anyone
holding it can post to the channel. `$DISCORD_WEBHOOK_URL` overrides the
file, and `$NEWLLM_CONFIG` points at a different config path. Set
`"enabled": false` to mute without deleting anything.

What reports, and when:

| source | messages |
|---|---|
| training | start; progress every 30 min with loss, val and ETA; time-budget stop; finish with best val and checkpoint path |
| teacher generators | start; every 200 requests with items/min and ETA; finish with totals |
| `supervise_teachers.sh` | a warning whenever it restarts a dead generator |
| evals | final score, grounded subset and mean F1 |
| GGUF export | architecture, layers, vocab and file size |

Every message carries the hostname, so several machines can share one
channel. Nothing here can break or slow a run: posts go out on a daemon
thread, every failure is swallowed after one warning to stderr, and training
that cannot reach Discord simply carries on. Set `notify_every_min=0` on the
`Trainer` to silence progress posts while keeping start and finish.

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
41/100 held-out, by tool calls needed:
  1 call   15/20      2 calls  10/12
  3 calls   6/19      4+ calls 10/49   <- half the distribution
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
