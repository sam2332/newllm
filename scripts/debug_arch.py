"""Correctness + speed harness for the arch_version=2 rewrite.

Checks, in order:
  1. causality       - token t's logits must not depend on tokens > t
  2. kv cache        - incremental decoding must match a full forward pass
  3. rope relativity - attention scores must depend on relative offset only
  4. residual health - gradient norm at layer 0 vs layer N-1 (the pre-LN bug)
  5. throughput      - tokens/sec, v1 vs v2, with and without the KV cache
"""

import sys, time, math
sys.path.insert(0, "/home/lmeadows/llm")
import torch
from model.transformer import Transformer

DEV = "cuda"
VOCAB = 271
torch.manual_seed(0)


def build(arch, **kw):
    return Transformer(vocab_size=VOCAB, d_model=256, n_layers=6, n_heads=8,
                       d_ff=1024, max_len=768, dropout=0.0,
                       arch_version=arch, **kw).to(DEV).eval()


def check_causality(model, name):
    x = torch.randint(0, VOCAB, (1, 64), device=DEV)
    with torch.no_grad():
        base = model(x)
        base = base[0] if isinstance(base, tuple) else base
        x2 = x.clone()
        x2[0, 40:] = torch.randint(0, VOCAB, (24,), device=DEV)
        alt = model(x2)
        alt = alt[0] if isinstance(alt, tuple) else alt
    # positions 0..39 must be identical
    delta = (base[0, :40] - alt[0, :40]).abs().max().item()
    print(f"  [{name}] causality max-delta on untouched prefix: {delta:.2e} "
          f"{'PASS' if delta < 1e-4 else 'FAIL'}")
    return delta < 1e-4


def check_kv_cache(model, name):
    x = torch.randint(0, VOCAB, (1, 48), device=DEV)
    with torch.no_grad():
        full = model(x)
        full = full[0] if isinstance(full, tuple) else full
        caches = model.new_caches()
        # prefill all but last, then feed last token alone
        model(x[:, :-1], caches=caches)
        step = model(x[:, -1:], caches=caches)
        step = step[0] if isinstance(step, tuple) else step
    delta = (full[0, -1] - step[0, -1]).abs().max().item()
    scale = full[0, -1].abs().max().item()
    print(f"  [{name}] kv-cache vs full-forward last-token delta: {delta:.2e} "
          f"(logit scale {scale:.2f}) {'PASS' if delta < 2e-2 else 'FAIL'}")
    return delta < 2e-2


def check_rope_relative(model, name):
    """Same bigram at two absolute positions should give similar next-token
    distributions if position is encoded relatively."""
    a, b = 65, 66
    pad = torch.randint(0, VOCAB, (1, 30), device=DEV)
    s1 = torch.cat([torch.tensor([[a, b]], device=DEV), pad], 1)
    s2 = torch.cat([pad, torch.tensor([[a, b]], device=DEV)], 1)
    with torch.no_grad():
        o1 = model(s1); o1 = o1[0] if isinstance(o1, tuple) else o1
        o2 = model(s2); o2 = o2[0] if isinstance(o2, tuple) else o2
    p1 = torch.softmax(o1[0, 1], -1)
    p2 = torch.softmax(o2[0, -1], -1)
    # not a pass/fail, just a diagnostic number
    tv = 0.5 * (p1 - p2).abs().sum().item()
    print(f"  [{name}] total-variation between same-bigram-different-position: {tv:.4f}")
    return tv


def check_residual_gradients(arch, name):
    """The pre-LN bug shows up as gradient starvation in early layers."""
    m = Transformer(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=8,
                    d_ff=1024, max_len=768, dropout=0.0,
                    arch_version=arch).to(DEV).train()
    x = torch.randint(0, VOCAB, (4, 128), device=DEV)
    y = torch.randint(0, VOCAB, (4, 128), device=DEV)
    out = m(x); out = out[0] if isinstance(out, tuple) else out
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()

    def gnorm(i):
        tot = 0.0
        for p in m.layers[i].parameters():
            if p.grad is not None:
                tot += p.grad.float().pow(2).sum().item()
        return math.sqrt(tot)

    first, last = gnorm(0), gnorm(len(m.layers) - 1)
    ratio = first / max(last, 1e-12)
    print(f"  [{name}] grad-norm layer0={first:.3e} layer11={last:.3e} "
          f"ratio={ratio:.4f}  (closer to 1.0 = healthier residual stream)")
    del m
    torch.cuda.empty_cache()
    return ratio


def bench(model, name, use_cache, n_new=100):
    x = torch.randint(0, VOCAB, (1, 256), device=DEV)
    torch.cuda.synchronize()
    with torch.no_grad():
        if use_cache:
            caches = model.new_caches()
            model(x, caches=caches)
            cur = x[:, -1:]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n_new):
                o = model(cur, caches=caches)
                o = o[0] if isinstance(o, tuple) else o
                cur = o[:, -1:].argmax(-1)
        else:
            seq = x
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n_new):
                o = model(seq)
                o = o[0] if isinstance(o, tuple) else o
                seq = torch.cat([seq, o[:, -1:].argmax(-1)], 1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  [{name}] {'kv-cache' if use_cache else 'no-cache '}: "
          f"{n_new/dt:7.1f} tok/s  ({dt*1000:.0f} ms for {n_new})")
    return n_new / dt


print("=" * 68)
print("ARCHITECTURE CORRECTNESS + SPEED HARNESS")
print("=" * 68)

print("\n1) arch_version=1 (original code)")
m1 = build(1)
print(f"  params: {m1.count_parameters():,}")
check_causality(m1, "v1")
check_rope_relative(m1, "v1")

print("\n2) arch_version=2 (rewritten)")
m2 = build(2)
print(f"  params: {m2.count_parameters():,}")
check_causality(m2, "v2")
check_kv_cache(m2, "v2")
check_rope_relative(m2, "v2")

print("\n3) GQA variant (n_kv_heads=2)")
m3 = build(2, n_kv_heads=2)
print(f"  params: {m3.count_parameters():,}")
check_causality(m3, "gqa")
check_kv_cache(m3, "gqa")

print("\n4) Residual-stream gradient health (12 layers)")
r1 = check_residual_gradients(1, "v1")
r2 = check_residual_gradients(2, "v2")
print(f"  -> v2/v1 improvement in layer0 gradient reach: {r2/max(r1,1e-12):.2f}x")

print("\n5) Generation throughput")
bench(m1, "v1", False)
bench(m2, "v2", False)
bench(m2, "v2", True)
bench(m3, "gqa", True)
print("\ndone.")
