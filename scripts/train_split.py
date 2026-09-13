"""Train the instruct model or the chat model from the shared architecture.

Two models, one architecture, two data distributions and two context budgets:

  instruct : single-turn task execution. One user request in, a verified tool
             plan and a final answer out. Short context (768).
  chat     : multi-turn conversation. Coreference across turns, topic switches,
             back-references, and an explicit end-of-turn after every answer so
             the model yields to the user instead of writing their next message.

Both consume the Ollama diversity pool when one is available, falling back to
the hand-written generators otherwise.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/lmeadows/llm")
import torch
from torch.utils.data import random_split

from agent.dataset import AgentDataset
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from model.transformer import Transformer
from training.trainer import Trainer
from training.device_utils import get_best_device

PRESETS = {
    "S": dict(d_model=512, n_layers=8, n_heads=8, d_ff=2048),
    "M": dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072),
    "L": dict(d_model=1024, n_layers=12, n_heads=16, d_ff=4096),
    "XL": dict(d_model=2048, n_layers=20, n_heads=16, d_ff=8192),
}
DEFAULT_MAX_LEN = {"instruct": 768, "chat": 2048}


def build_traces(mode, pool_path, samples, max_len, seed):
    """Prefer the generated diversity pool; fall back to the built-ins."""
    if pool_path and os.path.exists(pool_path):
        from agent.rich_dataset import load_pool, generate
        pool = load_pool(pool_path)
        traces, stats = generate(pool, samples, mode=mode,
                                 max_len=max_len, seed=seed)
        print(f"source: ollama pool {pool_path}")
        print(f"  facts={stats['facts']} numeric_facts={stats['numeric_facts']} "
              f"dropped={stats['dropped_invalid']} too_long={stats['too_long']}")
        return traces
    print(f"source: built-in generator (no pool at {pool_path})")
    if mode == "chat":
        from agent.chat_dataset import generate_chat_dataset
        return generate_chat_dataset(num_samples=samples, max_len=max_len,
                                     seed=seed)
    from agent.agent_dataset import generate_agent_dataset
    return generate_agent_dataset(num_samples=samples, max_len=max_len,
                                  seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["instruct", "chat"])
    ap.add_argument("--pool", default="data/ollama_pool.json")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--size", default="M", choices=list(PRESETS))
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-every", type=int, default=250,
                    help="write resumable state every N iterations (0 = off)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/agent_best.pt.resume if present")
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start from another checkpoint (e.g. train chat "
                         "from the finished instruct model)")
    args = ap.parse_args()

    max_len = args.max_len or DEFAULT_MAX_LEN[args.mode]
    out_dir = args.out or f"checkpoints_{args.mode}_{args.size}"
    os.makedirs(out_dir, exist_ok=True)
    device = get_best_device()
    torch.manual_seed(args.seed)

    print(f"=== training {args.mode} model ({args.size}) on {device} ===")
    traces = build_traces(args.mode, args.pool, args.samples, max_len, args.seed)
    ds = AgentDataset(traces, max_len=max_len, tokenizer=TOK)
    val_n = max(1, int(len(ds) * 0.05))
    train_set, val_set = random_split(
        ds, [len(ds) - val_n, val_n],
        generator=torch.Generator().manual_seed(args.seed))
    print(f"dataset: {len(train_set)} train / {len(val_set)} val, "
          f"max_len={max_len}")

    cfg = PRESETS[args.size]
    model = Transformer(vocab_size=TOK.vocab_size, max_len=max_len,
                        dropout=0.1, use_rope=True, tie_weights=False,
                        arch_version=2,
                        n_kv_heads=max(1, cfg["n_heads"] // 4),
                        qk_norm=True, **cfg)
    print(f"model: {model.count_parameters():,} parameters, "
          f"vocab={TOK.vocab_size}")

    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location="cpu",
                        weights_only=False)
        state = ck["model"]
        own = model.state_dict()
        # Grow the embedding / output rows if the checkpoint predates the
        # chat tokens, instead of refusing to load it.
        loaded, grown, skipped = 0, 0, 0
        for k, v in state.items():
            if k not in own:
                skipped += 1
                continue
            if own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            elif v.dim() == own[k].dim() and v.shape[1:] == own[k].shape[1:] \
                    and v.shape[0] < own[k].shape[0]:
                own[k][:v.shape[0]].copy_(v)
                grown += 1
            else:
                skipped += 1
        model.load_state_dict(own)
        print(f"warm start from {args.init_checkpoint}: "
              f"{loaded} tensors loaded, {grown} grown, {skipped} skipped")

    path = os.path.join(out_dir, "agent_best.pt")
    trainer = Trainer(model, train_set, batch_size=args.batch_size, lr=args.lr,
                      max_iters=args.iters, device=device, val_dataset=val_set,
                      grad_accum_steps=args.grad_accum,
                      warmup_steps=max(50, args.iters // 20),
                      use_amp=True, amp_dtype="bf16", z_loss=1e-4,
                      weight_decay=0.1, betas=(0.9, 0.95), num_workers=4,
                      checkpoint_path=path, save_every=args.save_every,
                      resume=args.resume)
    print(f"batch={args.batch_size} x {args.grad_accum} "
          f"(effective {args.batch_size * args.grad_accum}) lr={args.lr}")

    t0 = time.time()
    hist = trainer.train()
    dt = time.time() - t0

    model.save(path)
    # The run completed, so the resume state is no longer needed.
    resume_file = path + ".resume"
    if os.path.exists(resume_file):
        os.remove(resume_file)
    meta = {
        "mode": args.mode, "size": args.size, "params": model.count_parameters(),
        "max_len": max_len, "samples": args.samples, "iters": args.iters,
        "lr": args.lr, "seed": args.seed,
        "effective_batch": args.batch_size * args.grad_accum,
        "best_val_loss": trainer.best_val_loss,
        "final_train_loss": sum(hist[-50:]) / 50,
        "minutes": dt / 60, "pool": args.pool,
        "vocab_size": TOK.vocab_size,
    }
    json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
    print(f"\nbest_val={trainer.best_val_loss:.4f} "
          f"train_last50={meta['final_train_loss']:.4f} "
          f"time={dt/60:.1f}min")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
