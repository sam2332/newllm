"""Stage 1: pretrain on packed FineWeb-Edu shards.

The instruct-only model recites. It answers "Hello!" with a sentence copied
verbatim from its training data, "What is 2 + 2?" with gibberish, and cannot
repeat a four-digit code from two turns earlier - because a corpus composed
from a few thousand library items is 78.8% repeated sentences, and roughly 1B
tokens against 561M parameters is an order of magnitude short of what the
model needs to learn language rather than memorise phrasing.

This is the stage that was missing. Same Transformer, same Trainer; the only
differences are the data (fixed-length windows over packed shards) and the
mask (all ones - there is no prompt to exclude, every position is supervised).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/pretrain.py \\
        --size M --seq-len 2048 --batch-size 16 --grad-accum 4 --iters 60000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/lmeadows/llm")

import torch

from model.transformer import Transformer
from scripts.train_split import PRESETS
from training.packed_dataset import PackedShardDataset
from training.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/pretrain")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe8192_v2.json")
    ap.add_argument("--size", default="M", choices=list(PRESETS))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--val-windows", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--notify-every-min", type=float, default=10.0,
                    help="minutes between progress posts to the webhook (0 disables the timer; posts at 25/50/75%% still go out)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--arch-version", type=int, default=3)
    ap.add_argument("--rope-base", type=float, default=1e6)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile each layer in place (state_dict keys unchanged)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/pretrain_best.pt.resume")
    ap.add_argument("--out", default="checkpoints_pretrain")
    args = ap.parse_args()

    from agent.notify import notify as _notify
    from agent.tokenizer_registry import load_tokenizer, spec_for

    # torchrun sets LOCAL_RANK; one process per GPU, each on a disjoint shard.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp = local_rank >= 0
    rank0 = True
    device = None
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank0 = dist.get_rank() == 0
    notify = _notify if rank0 else (lambda *a, **k: False)
    if not rank0:
        import builtins
        builtins.print = lambda *a, **k: None

    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)
    spec = spec_for(tok)

    ds = PackedShardDataset(args.shards, seq_len=args.seq_len,
                            limit_tokens=args.limit_tokens)
    print(f"corpus: {ds.describe()}")
    # Held out from the END of the corpus, so a resumed or shortened run never
    # trains on its own validation windows.
    n_val = min(args.val_windows, max(1, len(ds) // 100))
    train_set = torch.utils.data.Subset(ds, range(0, len(ds) - n_val))
    val_set = torch.utils.data.Subset(ds, range(len(ds) - n_val, len(ds)))

    cfg = PRESETS[args.size]
    model = Transformer(vocab_size=tok.vocab_size, max_len=args.seq_len,
                        dropout=0.0, use_rope=True, tie_weights=False,
                        arch_version=args.arch_version, rope_base=args.rope_base,
                        n_kv_heads=max(1, cfg["n_heads"] // 4), qk_norm=True,
                        grad_checkpoint=args.grad_checkpoint, **cfg)
    model.tokenizer_spec = spec
    if args.compile:
        # Module.compile() compiles in place, so checkpoints keep plain keys
        # and load without a torch.compile wrapper.
        for layer in model.layers:
            layer.compile()
    world = dist.get_world_size() if ddp else 1
    n_params = model.count_parameters()
    # An iter is one micro-batch: Trainer steps the optimizer every
    # grad_accum iters. Counting accum here once reported the M run as 9.44B
    # tokens when it had seen 2.36B.
    tokens_per_step = args.batch_size * args.seq_len * world
    planned = args.iters * tokens_per_step
    print(f"model: {n_params:,} parameters, vocab {tok.vocab_size}, "
          f"seq_len {args.seq_len}")
    amount = (f"{planned/1e9:.2f}B" if planned >= 1e9 else f"{planned/1e6:.0f}M")
    print(f"{tokens_per_step:,} tokens/iter x {args.iters:,} iters "
          f"({args.iters // args.grad_accum:,} optimizer steps) = "
          f"{amount} tokens ({planned/ds.tokens:.2f} epochs, "
          f"{planned/n_params:.0f} tokens/param)")
    # Dropout is 0 here on purpose: with 9.5B tokens against 561M parameters
    # the corpus is the regulariser, and dropout would only slow convergence.

    import shutil
    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    if ddp:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model.to(device), device_ids=[local_rank])

    ckpt_path = os.path.join(args.out, "pretrain_best.pt")
    trainer = Trainer(model, train_set, val_dataset=val_set,
                      batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
                      max_iters=args.iters, lr=args.lr, warmup_steps=args.warmup,
                      save_every=args.save_every, val_batches=args.val_batches,
                      notify_every_min=args.notify_every_min,
                      checkpoint_path=ckpt_path, resume=args.resume,
                      max_minutes=args.max_minutes, ddp=ddp, device=device)
    notify(f"pretrain started: {args.size} {n_params/1e6:.0f}M, "
           f"{planned/1e9:.2f}B tokens over {args.iters:,} iters "
           f"({planned/n_params:.0f} tok/param)", tag="pretrain")
    t0 = time.time()
    hist = trainer.train()
    dt = time.time() - t0

    trainer.save_best_val(ckpt_path)
    if not rank0:
        dist.destroy_process_group()
        return
    meta = {"size": args.size, "params": n_params, "seq_len": args.seq_len,
            "iters": args.iters, "lr": args.lr, "tokens": int(planned),
            "corpus_tokens": int(ds.tokens), "tokenizer": spec,
            "best_val_loss": trainer.best_val_loss, "minutes": dt / 60,
            "arch_version": args.arch_version, "rope_base": args.rope_base}
    json.dump(meta, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    print(f"\nbest_val={trainer.best_val_loss:.4f} time={dt/60:.1f}min "
          f"-> {ckpt_path}")
    notify(f":white_check_mark: **pretrain finished** {args.size}, "
           f"best val {trainer.best_val_loss:.4f} in {dt/3600:.1f}h -> "
           f"`{args.out}/pretrain_best.pt`", tag="pretrain", blocking=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
