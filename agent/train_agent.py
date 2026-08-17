"""Train a tiny agentic transformer on synthetic ReAct traces.

The model learns to map questions to chains of Thought/Action/Observation/Answer
with an explicit BEGIN_THINK/END_THINK internal reasoning block.
"""

import torch
import random
import time
from training.dataset import StoryDataset
from training.trainer import Trainer
from training.device_utils import get_best_device
from model.transformer import Transformer
from agent.agent_dataset import generate_agent_dataset, THINK_START
from agent.agent_loop import run_agent
from agent.tools import Toolbox
from results_logger import save_result
from sampling import Sampler


def make_agent_dataset(num_samples=100000, max_len=256):
    traces = generate_agent_dataset(num_samples=num_samples)
    return StoryDataset(traces, max_len=max_len, max_vocab=256)


def evaluate_agent(model, cases: list, device: str):
    """cases is list of {question, expected, kind}."""
    toolbox = Toolbox()
    correct = 0
    details = []
    for case in cases:
        q = case["question"]
        expected = case.get("expected")
        kind = case.get("kind", "exact")
        result = run_agent(model, q, toolbox, device=device)
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
        details.append({"question": q, "expected": expected,
                        "got": got, "thinking": think, "ok": ok})
    return correct / max(1, len(cases)), details


def train_agent(num_samples=100000, iters=10000):
    device = get_best_device()
    print(f"\n=== Agent training on {device} ===")
    dataset = make_agent_dataset(num_samples=num_samples)
    print(f"dataset: {len(dataset)} traces")

    model = Transformer(
        vocab_size=256,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_len=256,
        attention_type="standard",
        use_rope=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Agent model parameters: {total:,}")

    trainer = Trainer(model, dataset, batch_size=32, lr=3e-4,
                      max_iters=iters, device=device)
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start
    print(f"\n=== training finished in {elapsed/60:.1f} minutes ===")

    final_loss = history[-1]
    best_loss = min(history)
    avg_last_50 = sum(history[-50:]) / 50
    print(f"final={final_loss:.4f} best={best_loss:.4f} "
          f"avg_last_50={avg_last_50:.4f} time={elapsed:.1f}s")

    eval_cases = [
        {"question": "What is 12 + 8?", "expected": "20", "kind": "numeric"},
        {"question": "Calculate 15 * 4.", "expected": "60", "kind": "numeric"},
        {"question": "What is the project?", "expected": "newllm", "kind": "exact"},
        {"question": "Retrieve the leader.", "expected": "grug", "kind": "exact"},
        {"question": "What is the current date and time?", "kind": "date"},
        {"question": "Multiply 3 and 4, then add the length of the leader.", "expected": "17", "kind": "numeric"},
        {"question": "Add 10 to the version.", "expected": "10.1", "kind": "numeric"},
    ]
    acc, details = evaluate_agent(model, eval_cases, device)
    print(f"agent accuracy: {acc:.2%}")
    for d in details:
        print(f"  {d['question'][:45]:45} expected={d['expected']} got={d['got']} ok={d['ok']}")
        if d["thinking"]:
            print(f"      thinking: {d['thinking'][:80]}")

    sampler = Sampler(temperature=0.6, top_k=20, top_p=0.9,
                      repetition_penalty=1.05)
    sample_q = "What is 7 * 6?"
    sample_result = run_agent(model, sample_q, Toolbox(), device=device,
                              sampler=sampler)
    print(f"sample: {sample_q}")
    print(f"thinking: {sample_result.get('thinking', '')[:120]}")
    print(f"answer: {sample_result['final_answer']}")

    result = {
        "num_samples": num_samples,
        "iters": iters,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "avg_last_50": avg_last_50,
        "elapsed_seconds": elapsed,
        "accuracy": acc,
        "eval_details": details,
        "sample": {"question": sample_q,
                   "answer": sample_result["final_answer"],
                   "thinking": sample_result.get("thinking", ""),
                   "trace": sample_result["trace"]},
        "history": history,
    }
    path = save_result("agent", result)
    print(f"results saved to {path}")
    return result


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    train_agent(num_samples=100000, iters=10000)
