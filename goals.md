# Project goals

Updated 2026-09-14. Historical goal lists are in
[archive/notes/](archive/notes/); most of the original list (build a transformer
from scratch, modern feature modules, sampling, MoE routing, an 8K-context run
on an RTX 5060 Ti) is done or superseded by the current machine.

## The goal

**The smallest model that can accurately do tools and chat, with some roleplay,
usable from Ollama.**

Four things have to be true at once, and only the first has been measured.

### 1. Tools - partly there

Read tool schemas from context and call them correctly, including tools never
seen in training.

- now: **41/100** on held-out traces
- blocker: chains of 4+ calls score **10/49**, and that is half the distribution
- the failure is grounding on the *first* call, not depth

### 2. Chat - not started

Multi-turn conversation: coreference, ellipsis, back-reference, topic switch, and
knowing when no tool is needed. Yield to the user at end of turn instead of
writing their next message.

`scripts/eval_chat.py` scores all five separately and has never been run against
a trained model. `train_split.py --mode chat` works. This is the largest
unmeasured gap.

### 3. Roleplay - not started

No data, no eval, no format decision. Needs a judgement on tokenizer first: at
byte level a 3,000-token reply is ~484 words, which is probably not what
"roleplay" means here.

### 4. Small - never tested

Every number in this project is the **M** preset (74.8M). S is 22.8M. The
question "what is the smallest model that can do this" has not been asked once.
A run is 89 minutes, so this is cheap to answer.

## Constraints that shape the work

- **Ollama-usable.** A proxy is acceptable (`serve_ollama.py` already
  works), so native GGUF is a stretch goal rather than a requirement. This is why
  the tokenizer is still an open decision instead of a forced one.
- **Responses up to ~3,000 tokens.** Current generation caps are 120-400.
- **A write-file tool is wanted.** It must not be bolted onto `repo_tools`;
  read-the-repo and write-files stay separate capabilities.
- **One GPU at a time.** Dual-GPU load browns the machine out.
- **The teacher never produces a tool result.** Every observation comes from the
  real toolbox. This is what makes the dataset ground truth.

## Done

- Transformer from scratch, modular under `model/`
- `arch_version=2`: pre-norm, per-head RoPE, working init, KV cache, bf16
- Modern modules: MLA, sparse attention, MoE, SSM, multi-token, reasoning-time
- Sampling: temperature, top-k, top-p, min-p, repetition penalty
- JSON tool protocol with assistant-only loss masking
- Tool schemas in context with randomized names, so tools are read not memorized
- 11 tools across three isolation levels (plain, read-only repo, sandboxed exec)
- Ollama-teacher data generation that never fabricates tool results
- Parallel cached dataset build (3,000 traces/s)
- Held-out evaluation that reflects the training distribution
- Reproducible run scripts and a preflight check
- Diagnosed and survived the power/stability problems that were killing runs

## Not doing

- Competing with frontier models. This is 25-75M parameters.
- Training on data whose tool outputs were written by a teacher.
- Two-GPU training, until the power situation is properly understood.
