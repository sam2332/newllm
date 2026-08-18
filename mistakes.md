# Mistakes & Lessons

> **Rule:** Always update this file after fixing a non-trivial bug or discovering a pitfall.
> Future agents must read it before making changes to the codebase.

## MoE load balancing

**Mistake:** First MoE routed every sequence to expert 3.
**Why:** No auxiliary loss. Router converged to exploit one expert because it was easiest.
**Fix:** Added `_router_loss()` in `model/sequence_moe.py` and added `aux_loss` to training loss with weight 1.0. Also added Gumbel noise during training to force exploration. Now hunt/fire/river cluster to expert 0 and sky to expert 2.

## Dataset collate with `torch.utils.data.Subset`

**Mistake:** `Trainer` failed with `AttributeError: 'Subset' object has no attribute 'collate_pad'`.
**Why:** `random_split` wraps `StoryDataset` in `Subset`, which does not forward custom attributes.
**Fix:** Read `collate_pad` from the underlying dataset object when a `Subset` is passed.

## CPU-only PyTorch

**Mistake:** `torch.cuda.is_available()` returned `False` despite having an RTX 5060 Ti.
**Why:** Installed `torch 2.11.0+cpu` wheel.
**Fix:** Reinstalled with CUDA 12.8 index:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## RoPE shape mismatch

**Mistake:** `RuntimeError: tensor a (128) must match tensor b (64)` when applying RoPE.
**Why:** `cos`/`sin` buffers had shape `(seq_len, d_model/2)` but `rotate_half` returned full `d_model` size.
**Fix:** Use `repeat_interleave(2, dim=-1)` on cos/sin to expand each pair.

## Generation defaults

**Mistake:** Greedy argmax generation always collapsed to repetition (`t t t ...`) on tiny models.
**Why:** Argmax is brittle with low-capacity models and small vocab.
**Fix:** Added `sampling.py` with temperature, top-k, top-p, min-p, repetition penalty, and wired it into `Trainer.generate()`.

## 8K context under-training

**Mistake:** First 8K run produced `t t t` because final loss stayed ~2.5.
**Why:** Model was too small (d_model=128, 4 layers) and trained only 1000 steps on limited data.
**Fix:** Started `training_8k_big.py` with d_model=512, 8 layers, gradient accumulation, LR schedule, validation split, and 5000 steps.

## Sequence MoE generation quality

**Status:** Ongoing. Routing is balanced but generated text still degrades into repetition.
**Hypothesis:** Each mini-mind expert is only trained on ~25% of data; 3K steps is too few. Need longer training or shared expert layers.

## Agent loop context length

**Mistake:** `RuntimeError: tensor a (306) must match tensor b (256)` when running the tiny agent.
**Why:** ReAct loop appends observations, so context can exceed `max_len`. RoPE buffers are sized for `max_len`.
**Fix:** Truncate `input_ids` to the model's `max_len` before every forward pass in `agent/agent_loop.py`.

## Agent trace formatting

**Mistake:** Trained agent produced 14.29% accuracy and regurgitated the prompt (`Question: ... BEGIN_THINK`) instead of generating a proper ReAct chain.
**Why:** Training traces were joined with single spaces in `agent/agent_dataset.py`, so `BEGIN_THINK`/`END_THINK`/actions/observations ran together without line delimiters. The model could not learn structure. Inference also seeded with no newline separators and used `TOOL_RE.search(text)` (first action) instead of the last generated action.
**Fix:** Changed `_format_trace` to use newline-delimited lines. Updated `agent_loop.py` to seed with `Question: ...\nBEGIN_THINK\n`, append `\nObservation: ...\n` after each tool result, and parse the *last* `Action:` inside the think block. Also expanded `Toolbox.memory` to match evaluation keys and fixed answer extraction to stop at newlines.

## Large agent model loss explosion

**Mistake:** 140M-parameter agent model started training with `loss=344` and stayed above 200.
**Why:** Tied input/output embeddings combined with AMP fp16 caused the final 1024→256 logit projection to overflow. The byte-level cross-entropy random baseline is ~5.5, so any value >100 means NaN/overflow.
**Fix:**
- Untied embeddings by default (`tie_weights=False`) in `model/transformer.py`.
- Compute final decoder logits in fp32 by casting the last hidden state before `TextDecoder`.
- Switched trainer AMP from deprecated `torch.cuda.amp` to `torch.amp` API.
- Guarded `TextDecoder` dropout so it only applies during training.
- Added scaled Xavier-style weight init (`_init_weights`) for deep/wide stability.
- Added `make_model` size presets so scaling experiments stay reproducible.

Result: v3b started at `loss≈5.7` and dropped below 0.01 within 2000 steps.

## No causal mask in Transformer

**Mistake:** After the loss explosion was fixed, the 140M agent reached very low training loss but generated complete gibberish during inference (e.g. `Action: calc[10 17]` for `12 + 8`).
**Why:** `model/transformer.py` had no causal mask, so every position attended to future tokens during training. The model learned to cheat by looking ahead and never learned proper left-to-right generation.
**Fix:** Added `_causal_mask()` to `Transformer.forward()` so the model only attends to previous positions when no external mask is supplied. Also clipped attention masked values to `-1e4` instead of `-1e9` to avoid fp16 `-inf` saturation in `model/attention.py` and `model/sparse_attention.py`.

Result: after retraining with the causal mask, the agent produced correct single-step ReAct traces and coherent multi-step chains.

## Agent answer extraction

**Mistake:** Agent loop stopped at `END_THINK` and expected an `Answer:` line afterwards, but the model often emitted `END_THINK\nAnswer:` with nothing following, yielding an empty final answer.
**Why:** The loop returned `text[:end]` up to `END_THINK`, and then tried to extract `Answer:` from a fragment that contained only the header.
**Fix:** In `agent/agent_loop.py`, when the think block closes after one or more tool actions, return the result of the last executed tool as the answer. This matches how the synthetic training traces are structured and makes tool use the source of truth.

## Multi-step inference required in-loop tool execution

**Mistake:** Multi-step ReAct traces in the dataset were never actually executed at inference time. The loop generated the whole trace at once, hallucinated observations, and stopped at the first `Action:`.
**Why:** `run_agent` generated a fixed number of tokens, matched actions once, and did not feed real observations back into the model.
**Fix:** Rewrote `agent/agent_loop.py` to generate incrementally. Whenever a new `Action: tool[arg]` line appears, execution pauses, the real tool is run, the observation is appended to the context, and generation continues from the updated context. The context is retokenised after each observation so the model sees the actual result. `max_new` in `agent/chat.py` was raised to 400 to fit 3-step traces.

Result: the S-size curriculum model now correctly answers single-step and multi-hop questions such as "Add 5 to the version." (→ 5.1) and "Multiply 3 and 4, then add the length of the leader." (→ 16).

## JSON tool-call parser could not handle nested objects

**Mistake:** After migrating to JSON tool calls, the S-size agent generated well-formed `Action: {"tool":"calc","args":{"expr":"..."}}` lines during inference, but `run_agent` never paused to execute them. The eval showed `final_answer` as the raw context and tool `steps` stayed empty.
**Why:** `_find_unexecuted_action` used the regex `Action:\s*(\{.*?\})`. The lazy `.*?` stops at the first `}`, so for nested objects it captured an unbalanced fragment like `{"tool":"calc","args":{"expr":"..."}` and `json.loads` failed silently.
**Fix:** Replaced the regex with a balanced-brace scanner (`_find_json_end`) that tracks brace depth while respecting quoted strings. JSON calls now parse correctly and the loop executes tools.

## Agent model underfits JSON + web details on the first full-dataset run

**Status:** Observed on `checkpoints_web/agent_best.pt` after 4k S-size iters.
**Symptom:** Tools execute, but the model emits wrong calc operands (e.g. `18 + 11` for `12 + 8`) and wrong web queries for some facts. Final accuracy on the 17-question eval battery was 0/17.
**Why:** A 25M model trained directly on the full mixed dataset (math, memory, date, web, multi-hop web+math) did not converge enough to copy question numbers into the JSON action arguments.
**Fix plan:**
- Use a true curriculum: first train on single-step traces only, then resume on multi-step + web-math traces.
- Add a `--resume-from` argument to `agent/train_agent.py` so resuming can start from a user-chosen checkpoint path.
- Save the best validation-loss checkpoint instead of the final checkpoint.

## L-size curriculum run stalled/died mid-training

**Status:** Observed, not yet root-caused.
**Symptom:** `agent_run_v3f.log` stopped at iter 5122/10000 (~51%). Loss was healthy (~0.05) and validation loss was good, but throughput collapsed from ~3.2 it/s to 6-7 s/it right before the process disappeared.
**Hypotheses:**
- OOM event during validation/gradient accumulation on the 151M model with batch 16 × accum 8 = effective 128 on a 16GB RTX 5060 Ti.
- Power/thermal throttling or Windows terminating the background process after sustained load.
- Checkpoint save at iter 5100 may have coincided with a transient disk/VRAM issue.
**Next steps:**
- Restart the L run with a smaller effective batch (e.g. batch 8 × accum 8) or gradient checkpointing.
- Monitor `nvidia-smi` memory and clock throttling if it stalls again.
- Save checkpoints more frequently and resume rather than running uninterrupted.
