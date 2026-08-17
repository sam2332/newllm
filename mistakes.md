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
