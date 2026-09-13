"""Head-to-head: arch_version=1 vs 2 on the real agent data.

Same seed, same traces, same token budget, same LR schedule. The only
difference is the architecture, so any gap is attributable to the rewrite.
"""

import sys, time, argparse, json
sys.path.insert(0, "/home/lmeadows/llm")
import torch
from torch.utils.data import random_split

from agent.agent_dataset import generate_simple_agent_dataset, generate_agent_dataset
from agent.dataset import AgentDataset
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from model.transformer import Transformer
from training.trainer import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=20000)
ap.add_argument("--iters", type=int, default=1200)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--mixed", action="store_true",
                help="use the hard mixed/multi-hop distribution")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
args = ap.parse_args()

MAX_LEN = 768
torch.manual_seed(args.seed)

print(f"generating {args.samples} traces (mixed={args.mixed})...")
if args.mixed:
    traces = generate_agent_dataset(num_samples=args.samples, max_len=MAX_LEN,
                                    seed=args.seed)
else:
    traces = generate_simple_agent_dataset(num_samples=args.samples,
                                           max_len=MAX_LEN, seed=args.seed)
full = AgentDataset(traces, max_len=MAX_LEN, tokenizer=TOK)
val_n = int(len(full) * 0.05)
train_set, val_set = random_split(
    full, [len(full) - val_n, val_n],
    generator=torch.Generator().manual_seed(args.seed))
print(f"dataset: {len(train_set)} train / {len(val_set)} val\n")


def build(arch, **kw):
    torch.manual_seed(args.seed)
    return Transformer(vocab_size=TOK.vocab_size, d_model=512, n_layers=8,
                       n_heads=8, d_ff=2048, max_len=MAX_LEN, dropout=0.1,
                       use_rope=True, tie_weights=False,
                       arch_version=arch, **kw)


def run(label, arch, amp_dtype, **kw):
    torch.manual_seed(args.seed)
    model = build(arch, **kw)
    n = model.count_parameters()
    tr = Trainer(model, train_set, batch_size=args.batch, lr=args.lr,
                 max_iters=args.iters, val_dataset=val_set,
                 grad_accum_steps=4, warmup_steps=args.iters // 10,
                 use_amp=True, amp_dtype=amp_dtype,
                 z_loss=1e-4 if arch >= 2 else 0.0,
                 weight_decay=0.1 if arch >= 2 else 0.0,
                 betas=(0.9, 0.95) if arch >= 2 else (0.9, 0.999),
                 num_workers=4)
    t0 = time.time()
    hist = tr.train()
    dt = time.time() - t0
    final_val = tr.val_loss()
    print(f"\n{label}")
    print(f"  params        {n:,}")
    print(f"  best val loss {tr.best_val_loss:.4f}  (final {final_val:.4f})")
    print(f"  train loss    last50 avg {sum(hist[-50:])/50:.4f}")
    print(f"  wall time     {dt:.0f}s  ({args.iters/dt:.1f} it/s)")
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": model._get_config()},
               f"/tmp/claude-1001/-home-lmeadows-llm/c01a16c4-6b98-42c5-90cc-1de1d8bb39a7/scratchpad/ab_{label.replace(' ','_').replace('/','')}.pt")
    out = {"label": label, "params": n, "best_val": tr.best_val_loss,
           "final_val": final_val, "seconds": dt,
           "last50": sum(hist[-50:]) / 50}
    del model, tr
    torch.cuda.empty_cache()
    return out


results = []
results.append(run("v1 original (fp16)", 1, "fp16"))
results.append(run("v2 modern (bf16)", 2, "bf16"))
results.append(run("v2 modern + GQA (bf16)", 2, "bf16", n_kv_heads=2))

print("\n" + "=" * 70)
print(f"{'variant':26s} {'params':>12s} {'best val':>10s} {'time':>8s}")
print("-" * 70)
base = results[0]["best_val"]
for r in results:
    print(f"{r['label']:26s} {r['params']:12,} {r['best_val']:10.4f} "
          f"{r['seconds']:7.0f}s   {'baseline' if r is base else f'{base/r[chr(39)+chr(39)] if False else 0:.0f}'}"
          if False else
          f"{r['label']:26s} {r['params']:12,} {r['best_val']:10.4f} {r['seconds']:7.0f}s")
print("-" * 70)
for r in results[1:]:
    print(f"  {r['label']}: val loss {base:.4f} -> {r['best_val']:.4f} "
          f"({(1 - r['best_val']/base)*100:+.1f}%), "
          f"{results[0]['seconds']/r['seconds']:.2f}x faster")
json.dump(results, open("/tmp/claude-1001/-home-lmeadows-llm/c01a16c4-6b98-42c5-90cc-1de1d8bb39a7/scratchpad/ab_results.json","w"), indent=2)
