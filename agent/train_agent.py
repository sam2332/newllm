"""Train a tiny agentic transformer on synthetic ReAct traces.

The model learns to map questions to chains of Thought/Action/Observation/Answer
with an explicit BEGIN_THINK/END_THINK internal reasoning block.
"""

import os
import torch
import random
import time
import argparse
from torch.utils.data import random_split
from training.dataset import StoryDataset
from training.trainer import Trainer
from training.device_utils import get_best_device
from model.transformer import Transformer
from agent.agent_dataset import generate_simple_agent_dataset, generate_agent_dataset
from agent.agent_loop import run_agent
from agent.tools import Toolbox
from results_logger import save_result
from sampling import Sampler


def make_agent_dataset(num_samples=100000, max_len=512, val_frac=0.05,
                       simple=True, seed=42, math_only=False):
    if math_only:
        traces = generate_simple_agent_dataset(num_samples=num_samples,
                                               max_len=max_len, seed=seed,
                                               use_json_tools=True,
                                               math_only=True)
    elif simple:
        traces = generate_simple_agent_dataset(num_samples=num_samples,
                                               max_len=max_len, seed=seed,
                                               use_json_tools=True)
    else:
        traces = generate_agent_dataset(num_samples=num_samples,
                                        max_len=max_len, seed=seed,
                                        use_json_tools=True)
    full = StoryDataset(traces, max_len=max_len, max_vocab=256)
    if val_frac <= 0:
        return full, None
    val_size = int(len(full) * val_frac)
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(seed))
    return train_set, val_set


def make_model(max_len=512, size="L", attention_type="standard", use_moe=False,
               num_experts=4, top_k=2):
    """
    Create the agentic transformer.

    Sizes:
        S  ~ 16M  (d_model=512,  8 layers, 8 heads,  d_ff=2048)
        M  ~ 60M  (d_model=768,  12 layers, 12 heads, d_ff=3072)
        L  ~ 140M (d_model=1024, 12 layers, 16 heads, d_ff=4096)
        XL ~ 1B   (d_model=2048, 20 layers, 16 heads, d_ff=8192)

    Optional MoE/MLA/sparse attention can be enabled by name, e.g.:
        attention_type='mla' or use_moe=True.
    """
    presets = {
        "S":  {"d_model": 512,  "n_layers": 8,  "n_heads": 8,  "d_ff": 2048, "dropout": 0.1},
        "M":  {"d_model": 768,  "n_layers": 12, "n_heads": 12, "d_ff": 3072, "dropout": 0.1},
        "L":  {"d_model": 1024, "n_layers": 12, "n_heads": 16, "d_ff": 4096, "dropout": 0.1},
        "XL": {"d_model": 2048, "n_layers": 20, "n_heads": 16, "d_ff": 8192, "dropout": 0.1},
    }
    cfg = presets[size]
    model = Transformer(
        vocab_size=256,
        max_len=max_len,
        attention_type=attention_type,
        use_rope=True,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
        tie_weights=False,  # untied embeddings prevent output-logit overflow in fp16
        **cfg,
    )
    print(f"Agent model size={size} parameters: {model.count_parameters():,}")
    return model


def evaluate_agent(model, cases: list, device: str, toolbox=None):
    """cases is list of {question, expected, kind}."""
    if toolbox is None:
        toolbox = Toolbox()
    correct = 0
    by_kind = {}
    details = []
    for case in cases:
        q = case["question"]
        expected = case.get("expected")
        kind = case.get("kind", "exact")
        result = run_agent(model, q, toolbox, device=device, greedy=True)
        got = result["final_answer"]
        think = result.get("thinking", "")
        ok = False
        if kind == "date":
            ok = bool(got) and any(c.isdigit() or c == "-" for c in got)
        elif kind == "numeric":
            try:
                ok = abs(float(got) - float(expected)) < 1e-3
            except Exception:
                ok = got.strip().lower() == str(expected).strip().lower()
        else:
            ok = got.strip().lower() == str(expected).strip().lower()
        if ok:
            correct += 1
        by_kind.setdefault(kind, {"total": 0, "correct": 0})
        by_kind[kind]["total"] += 1
        if ok:
            by_kind[kind]["correct"] += 1
        details.append({"question": q, "expected": expected,
                        "got": got, "thinking": think, "ok": ok,
                        "kind": kind})
    per_kind = {k: v["correct"] / max(1, v["total"]) for k, v in by_kind.items()}
    return correct / max(1, len(cases)), per_kind, details


def train_agent(num_samples=100000, iters=10000, size="L",
                attention_type="standard", use_moe=False,
                checkpoint_dir="checkpoints", resume=True,
                curriculum=False, save_every=1000, lr=3e-4, math_only=False):
    device = get_best_device()
    print(f"\n=== Agent training on {device} ===")
    max_len = 512

    # Real two-stage curriculum:
    #   stage 1 (simple=True)  -> single-step math/memory/date/web traces
    #   stage 2 (simple=False) -> add multi-hop, web+math, memory+math traces
    train_set, val_set = make_agent_dataset(num_samples=num_samples,
                                            max_len=max_len, val_frac=0.05,
                                            simple=not curriculum,
                                            math_only=math_only)
    print(f"dataset: {len(train_set)} train, {len(val_set or [])} val traces")
    if math_only:
        print("math-only diagnostic mode")
    elif curriculum:
        print("curriculum stage 2: multi-step + web-math traces")
    else:
        print("curriculum stage 1: single-step traces only")

    model = make_model(max_len=max_len, size=size,
                       attention_type=attention_type, use_moe=use_moe)

    # Gradient accum to effective batch ~128 while keeping VRAM low on 16GB.
    batch_size = 16
    grad_accum = 8
    trainer = Trainer(model, train_set, batch_size=batch_size, lr=lr,
                      max_iters=iters, device=device, val_dataset=val_set,
                      grad_accum_steps=grad_accum, warmup_steps=500,
                      use_amp=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "agent_best.pt")
    if resume and os.path.exists(checkpoint_path):
        print(f"resuming from {checkpoint_path}")
        model.load(checkpoint_path)

    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start
    print(f"\n=== training finished in {elapsed/60:.1f} minutes ===")

    final_loss = history[-1]
    best_loss = min(history)
    avg_last_50 = sum(history[-50:]) / 50
    best_val = trainer.best_val_loss
    print(f"final={final_loss:.4f} best={best_loss:.4f} "
          f"avg_last_50={avg_last_50:.4f} best_val={best_val:.4f} "
          f"time={elapsed:.1f}s")

    # Save best checkpoint by validation loss.
    # The trainer already restored the best-validation weights, so saving
    # model.state_dict() here captures the optimal run.
    if val_set is not None:
        model.save(checkpoint_path)
        print(f"checkpoint saved to {checkpoint_path}")
    else:
        # No validation set: keep the final weights.
        model.save(checkpoint_path)
        print(f"checkpoint saved to {checkpoint_path}")

    # Save a timestamped snapshot too so we never lose an experiment.
    snapshot_path = os.path.join(
        checkpoint_dir,
        f"agent_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    )
    model.save(snapshot_path)
    print(f"snapshot saved to {snapshot_path}")

    eval_cases = [
        {"question": "What is 12 + 8?", "expected": "20", "kind": "numeric"},
        {"question": "Calculate 15 * 4.", "expected": "60", "kind": "numeric"},
        {"question": "What is the project?", "expected": "newllm", "kind": "exact"},
        {"question": "Retrieve the leader.", "expected": "grug", "kind": "exact"},
        {"question": "What is the current date and time?", "kind": "date"},
        {"question": "What is 7 * 6?", "expected": "42", "kind": "numeric"},
        {"question": "Look up the version.", "expected": "0.1", "kind": "exact"},
        {"question": "Add 5 to the version.", "expected": "5.1", "kind": "numeric"},
        {"question": "Multiply 3 and 4, then add the length of the leader.", "expected": "16", "kind": "numeric"},
        # web_search eval: model must look up facts not in parametric memory.
        {"question": "What is the capital of france?", "expected": "Paris", "kind": "exact"},
        {"question": "Who is the president of the united states?", "expected": "Alice Johnson", "kind": "exact"},
        {"question": "How many planets are there?", "expected": "8", "kind": "exact"},
        {"question": "What is the speed of light?", "expected": "299792458 m/s", "kind": "exact"},
        {"question": "Look up the boiling point of water.", "expected": "100 degrees Celsius", "kind": "exact"},
        {"question": "What is the largest planet?", "expected": "Jupiter", "kind": "exact"},
        # multi-hop web + math
        {"question": "What is the capital of japan plus 5?", "expected": "10", "kind": "numeric"},
        {"question": "How many planets are there times 2?", "expected": "16", "kind": "numeric"},
    ]
    acc, per_kind, details = evaluate_agent(model, eval_cases, device)
    print(f"agent accuracy: {acc:.2%}")
    for k, v in per_kind.items():
        print(f"  {k}: {v:.2%}")
    for d in details:
        print(f"  {d['question'][:45]:45} expected={d['expected']} got={d['got']} ok={d['ok']}")
        if d["thinking"]:
            print(f"      thinking: {d['thinking'][:120]}")

    sampler = Sampler(temperature=0.6, top_k=20, top_p=0.9,
                      repetition_penalty=1.0)
    sample_q = "What is 7 * 6?"
    sample_result = run_agent(model, sample_q, Toolbox(), device=device,
                              sampler=sampler)
    print(f"sample: {sample_q}")
    print(f"thinking: {sample_result.get('thinking', '')[:200]}")
    print(f"answer: {sample_result['final_answer']}")

    result = {
        "num_samples": num_samples,
        "iters": iters,
        "model_size": size,
        "attention_type": attention_type,
        "use_moe": use_moe,
        "curriculum": curriculum,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "avg_last_50": avg_last_50,
        "best_val_loss": best_val,
        "elapsed_seconds": elapsed,
        "accuracy": acc,
        "per_kind_accuracy": per_kind,
        "eval_details": details,
        "sample": {
            "question": sample_q,
            "answer": sample_result["final_answer"],
            "thinking": sample_result.get("thinking", ""),
            "trace": sample_result["trace"],
            "steps": sample_result.get("steps", []),
        },
        "history": history,
        "val_history": trainer.val_history,
        "checkpoint": checkpoint_path,
        "snapshot": snapshot_path if val_set else None,
    }
    path = save_result("agent", result)
    print(f"results saved to {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the agentic LLM.")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--size", default="L", choices=["S", "M", "L", "XL"])
    parser.add_argument("--attention", default="standard",
                        choices=["standard", "mla", "sparse"])
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="peak learning rate (default 3e-4)")
    parser.add_argument("--math-only", action="store_true",
                        help="diagnostic: train only on single-step math traces")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    train_agent(
        num_samples=args.samples,
        iters=args.iters,
        size=args.size,
        attention_type=args.attention,
        use_moe=args.moe,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        curriculum=args.curriculum,
        lr=args.lr,
        math_only=args.math_only,
    )
