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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import random_split

from agent.dataset import AgentDataset
from agent.dataset_builder import build as build_dataset, PrebuiltDataset
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
# instruct was 768 when the fact KB held 12 short toy facts. With 682 generated
# facts the keys are much longer, and three-tool chains overflowed 768 - which
# silently dropped 94% of the cross-source pattern. 1024 keeps 100%.
DEFAULT_MAX_LEN = {"instruct": 1024, "chat": 2048}


def build_traces(mode, pool_path, samples, max_len, seed,
                 deep_fraction=0.0, deep_min=4, deep_max=12, quiet=False,
                 scenarios_path=None, scenario_fraction=0.0):
    """Prefer the generated diversity pool; fall back to the built-ins."""
    if pool_path and os.path.exists(pool_path):
        from agent.rich_dataset import load_pool, generate
        pool = load_pool(pool_path)
        traces, stats = generate(pool, samples, mode=mode,
                                 max_len=max_len, seed=seed,
                                 deep_fraction=deep_fraction,
                                 deep_min=deep_min, deep_max=deep_max)
        scenario_traces = []
        if scenarios_path and scenario_fraction and os.path.exists(scenarios_path):
            import json as _json, random as _random
            from agent.scenario_traces import generate_from_scenarios
            from agent.tools import Toolbox
            plans = _json.load(open(scenarios_path))
            want = int(samples * scenario_fraction)
            _rng = _random.Random(seed)

            def _thought(situation, fallback):
                opts = pool["thoughts"].get(situation)
                return _rng.choice(opts) if opts else fallback

            scenario_traces, s_stats = generate_from_scenarios(
                plans, pool["facts"], Toolbox.DEFAULT_MEMORY, want,
                rng_seed=seed, max_len=max_len, thought_fn=_thought,
                sandbox_pool=pool.get("sandbox"))
            samples = max(0, samples - len(scenario_traces))
            if not quiet:
                print(f"scenarios: {len(plans):,} plans -> "
                      f"{len(scenario_traces):,} traces  {s_stats}")
        if not quiet:
            print(f"source: ollama pool {pool_path}")
            print(f"  facts={stats['facts']} numeric={stats['numeric_facts']} "
                  f"too_long={stats['too_long']} "
                  f"ungrounded_rejected={stats['ungrounded_rejected']}")
            if deep_fraction:
                import collections
                hops = collections.Counter(t.count('"tool_call"') for t in traces)
                deep = sum(v for k, v in hops.items() if k >= 10)
                print(f"  deep chains: max {max(hops)} hops, "
                      f"{deep} traces with >=10 hops")
        return traces + scenario_traces
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
    ap.add_argument("--build-workers", type=int, default=48,
                    help="processes used to build the dataset")
    ap.add_argument("--scenarios", default="data/scenarios.json",
                    help="teacher-written scenario plans")
    ap.add_argument("--scenario-fraction", type=float, default=0.0,
                    help="fraction of the dataset built from scenario plans")
    ap.add_argument("--deep-fraction", type=float, default=0.0,
                    help="fraction of traces that are deep N-hop chains")
    ap.add_argument("--deep-min", type=int, default=4)
    ap.add_argument("--deep-max", type=int, default=12)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in backward; ~30%% slower, "
                         "needed for long context")
    ap.add_argument("--eval-every", type=int, default=500,
                    help="score the task battery every N steps (0 = off)")
    ap.add_argument("--eval-patience", type=int, default=4,
                    help="stop after this many evals with no improvement")
    ap.add_argument("--save-every", type=int, default=250,
                    help="write resumable state every N iterations (0 = off)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/agent_best.pt.resume if present")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop this segment after N minutes, write resume "
                         "state and exit 0 (0 = run to --iters)")
    ap.add_argument("--val-batches", type=int, default=0,
                    help="limit the validation pass to N batches (0 = all); "
                         "a full pass over 12.5k samples costs minutes")
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start from another checkpoint (e.g. train chat "
                         "from the finished instruct model)")
    args = ap.parse_args()

    # torchrun sets these; absent means single-GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = local_rank >= 0 and world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = get_best_device()
    is_main = (not ddp) or dist.get_rank() == 0

    max_len = args.max_len or DEFAULT_MAX_LEN[args.mode]
    out_dir = args.out or f"checkpoints_{args.mode}_{args.size}"
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    if is_main:
        print(f"=== training {args.mode} model ({args.size}) on {device} ===")
        if ddp:
            print(f"    DDP across {world_size} GPUs "
                  f"(effective batch x{world_size})")
    # Build once, in parallel, and cache. The single-threaded path spent ~44
    # minutes here and DDP made each rank repeat it; rank 0 now builds while
    # the others wait on the barrier, then everyone loads the same cache.
    build_kwargs = dict(
        samples=args.samples, mode=args.mode, pool_path=args.pool,
        scenarios_path=args.scenarios if args.scenario_fraction else None,
        scenario_fraction=args.scenario_fraction,
        deep_fraction=args.deep_fraction, deep_min=args.deep_min,
        deep_max=args.deep_max, max_len=max_len, seed=args.seed,
        workers=args.build_workers)
    if ddp:
        if is_main:
            samples_data = build_dataset(verbose=True, **build_kwargs)
            dist.barrier()
        else:
            dist.barrier()
            samples_data = build_dataset(verbose=False, **build_kwargs)
    else:
        samples_data = build_dataset(verbose=True, **build_kwargs)

    ds = PrebuiltDataset(samples_data)
    val_n = max(1, int(len(ds) * 0.05))
    train_set, val_set = random_split(
        ds, [len(ds) - val_n, val_n],
        generator=torch.Generator().manual_seed(args.seed))
    if is_main:
        hops = None
        print(f"dataset: {len(train_set):,} train / {len(val_set):,} val, "
              f"max_len={max_len}")

    cfg = PRESETS[args.size]
    model = Transformer(vocab_size=TOK.vocab_size, max_len=max_len,
                        dropout=0.1, use_rope=True, tie_weights=False,
                        arch_version=2,
                        n_kv_heads=max(1, cfg["n_heads"] // 4),
                        qk_norm=True,
                        grad_checkpoint=args.grad_checkpoint, **cfg)
    if is_main:
        print(f"model: {model.count_parameters():,} parameters, "
              f"vocab={TOK.vocab_size}, "
              f"grad_checkpoint={args.grad_checkpoint}")

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

    # Accuracy on a held-out battery, scored during training. Only rank 0 runs
    # it; the other ranks would duplicate the work and interleave output.
    eval_fn = None
    if args.eval_every and is_main:
        from scripts.eval_agent import CASES, score as _score
        import contextlib, io

        def eval_fn(m):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                acc, _ = _score(m, device=device, verbose=False)
            return acc

    if ddp:
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        find_unused_parameters=False)
    trainer = Trainer(model, train_set, batch_size=args.batch_size, lr=args.lr,
                      max_iters=args.iters, device=device, val_dataset=val_set,
                      grad_accum_steps=args.grad_accum,
                      warmup_steps=max(50, args.iters // 20),
                      use_amp=True, amp_dtype="bf16", z_loss=1e-4,
                      weight_decay=0.1, betas=(0.9, 0.95), num_workers=4,
                      checkpoint_path=path if is_main else None,
                      save_every=args.save_every if is_main else 0,
                      resume=args.resume, ddp=ddp,
                      eval_fn=eval_fn, eval_every=args.eval_every,
                      eval_patience=args.eval_patience,
                      max_minutes=args.max_minutes,
                      val_batches=args.val_batches)
    print(f"batch={args.batch_size} x {args.grad_accum} "
          f"(effective {args.batch_size * args.grad_accum}) lr={args.lr}")

    t0 = time.time()
    hist = trainer.train()
    dt = time.time() - t0

    if ddp:
        dist.barrier()
    if not is_main:
        dist.destroy_process_group()
        return
    if trainer.stopped_on_time:
        # Not finished. The resume state is the artifact; leave agent_best.pt
        # and run.json alone so they keep describing the last complete run.
        print(f"segment stopped on time after {dt/60:.1f} min; "
              f"resume with the same command to continue "
              f"(--iters {args.iters} unchanged)")
        if ddp:
            dist.destroy_process_group()
        return
    # Unwrap DDP before saving so the checkpoint has plain parameter names.
    to_save = model.module if isinstance(model, DistributedDataParallel) else model
    to_save.save(path)
    # The run completed, so the resume state is no longer needed.
    resume_file = path + ".resume"
    if os.path.exists(resume_file):
        os.remove(resume_file)
    meta = {
        "mode": args.mode, "size": args.size,
        "params": to_save.count_parameters(),
        "deep_fraction": args.deep_fraction,
        "deep_range": [args.deep_min, args.deep_max],
        "grad_checkpoint": args.grad_checkpoint,
        "world_size": world_size,
        "max_len": max_len, "samples": args.samples, "iters": args.iters,
        "lr": args.lr, "seed": args.seed,
        "effective_batch": args.batch_size * args.grad_accum,
        "best_val_loss": trainer.best_val_loss,
        "best_accuracy": trainer.best_acc,
        "best_accuracy_step": trainer.best_acc_step,
        "eval_history": trainer.eval_history,
        "stopped_early": trainer.stopped_early,
        "final_train_loss": sum(hist[-50:]) / 50,
        "minutes": dt / 60, "pool": args.pool,
        "vocab_size": TOK.vocab_size,
    }
    json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
    if trainer.eval_history:
        print("\naccuracy curve:")
        for st, a in trainer.eval_history:
            print(f"    step {st:6d}  {a:6.1%}"
                  f"{'   <- best' if st == trainer.best_acc_step else ''}")
    print(f"\nbest_val={trainer.best_val_loss:.4f} "
          f"train_last50={meta['final_train_loss']:.4f} "
          f"time={dt/60:.1f}min")
    print(f"saved -> {path}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
