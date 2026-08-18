"""Promote a trained agent checkpoint only if it passes the regression gate.

The gate runs a fixed test battery through the real ReAct tool loop and checks
that accuracy, per-kind accuracy, and individual required questions all meet
thresholds.  If the gate fails, the candidate is rejected and the current best
checkpoint is left untouched.
"""

import argparse
import os
import shutil
import sys
import torch

from agent.chat import load_checkpoint
from agent.train_agent import evaluate_agent
from agent.tools import Toolbox


# Default regression battery. Keep this in sync with the eval cases used in
# agent/train_agent.py so a promotion decision matches the training report.
DEFAULT_CASES = [
    {"question": "What is 12 + 8?", "expected": "20", "kind": "numeric"},
    {"question": "Calculate 15 * 4.", "expected": "60", "kind": "numeric"},
    {"question": "What is the project?", "expected": "newllm", "kind": "exact"},
    {"question": "Retrieve the leader.", "expected": "grug", "kind": "exact"},
    {"question": "What is the current date and time?", "kind": "date"},
    {"question": "What is 7 * 6?", "expected": "42", "kind": "numeric"},
    {"question": "Look up the version.", "expected": "0.1", "kind": "exact"},
    {"question": "Add 5 to the version.", "expected": "5.1", "kind": "numeric"},
    {"question": "Multiply 3 and 4, then add the length of the leader.", "expected": "16", "kind": "numeric"},
    # web_search facts are not in parametric memory.
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


class Gate:
    """Define pass/fail criteria for a candidate checkpoint."""

    def __init__(
        self,
        min_accuracy: float = 0.75,
        min_numeric_accuracy: float = 0.60,
        min_exact_accuracy: float = 0.80,
        required_cases: list = None,
    ):
        self.min_accuracy = min_accuracy
        self.min_numeric_accuracy = min_numeric_accuracy
        self.min_exact_accuracy = min_exact_accuracy
        self.required_cases = required_cases or []

    def passes(self, overall: float, per_kind: dict, details: list) -> tuple:
        reasons = []
        if overall < self.min_accuracy:
            reasons.append(
                f"overall accuracy {overall:.2%} < {self.min_accuracy:.0%}"
            )
        numeric = per_kind.get("numeric", 0.0)
        if numeric < self.min_numeric_accuracy:
            reasons.append(
                f"numeric accuracy {numeric:.2%} < {self.min_numeric_accuracy:.0%}"
            )
        exact = per_kind.get("exact", 1.0)
        if exact < self.min_exact_accuracy:
            reasons.append(
                f"exact accuracy {exact:.2%} < {self.min_exact_accuracy:.0%}"
            )
        detail_map = {d["question"]: d for d in details}
        for required in self.required_cases:
            q = required["question"]
            d = detail_map.get(q)
            if d is None or not d["ok"]:
                reasons.append(f"required case failed: {q}")
        return len(reasons) == 0, reasons


def run_gate(candidate_path: str, device: str = "cuda", gate: Gate = None):
    """Load candidate and run the regression battery."""
    gate = gate or Gate()
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_path}")

    print(f"\n=== promotion gate for {candidate_path} ===")
    model = load_checkpoint(candidate_path, device=device)
    acc, per_kind, details = evaluate_agent(model, DEFAULT_CASES, device, toolbox=Toolbox())

    print(f"\noverall accuracy: {acc:.2%}")
    for k, v in per_kind.items():
        print(f"  {k}: {v:.2%}")

    ok, reasons = gate.passes(acc, per_kind, details)
    if ok:
        print("\nGATE PASSED")
    else:
        print("\nGATE FAILED")
        for r in reasons:
            print(f"  - {r}")
    return ok, acc, per_kind, details


def promote(
    candidate_path: str,
    best_path: str,
    device: str = "cuda",
    gate: Gate = None,
    dry_run: bool = False,
):
    """Run gate and, if it passes, copy candidate to best_path."""
    ok, acc, per_kind, details = run_gate(candidate_path, device=device, gate=gate)
    if not ok:
        print(f"\nrejecting {candidate_path}; {best_path} was not changed.")
        return False

    if dry_run:
        print(f"\nDRY RUN: would promote {candidate_path} -> {best_path}")
        return True

    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
    # Keep a backup of the previous best so promotion is reversible.
    if os.path.exists(best_path):
        backup = best_path.replace(".pt", "_prev.pt")
        shutil.copy2(best_path, backup)
        print(f"backed up previous best to {backup}")
    shutil.copy2(candidate_path, best_path)
    print(f"promoted {candidate_path} -> {best_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Promote an agent checkpoint only if it passes the regression gate."
    )
    parser.add_argument(
        "candidate",
        help="Path to the candidate checkpoint to evaluate",
    )
    parser.add_argument(
        "--best",
        default="checkpoints/agent_best.pt",
        help="Path to the current best checkpoint to overwrite if gate passes",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.75,
        help="Minimum overall accuracy required for promotion",
    )
    parser.add_argument(
        "--min-numeric",
        type=float,
        default=0.60,
        help="Minimum numeric-case accuracy required",
    )
    parser.add_argument(
        "--min-exact",
        type=float,
        default=0.80,
        help="Minimum exact-string accuracy required",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the gate but do not copy the checkpoint",
    )
    parser.add_argument(
        "--required",
        nargs="*",
        default=None,
        help="Extra questions that must all pass (format: question=expected)",
    )
    args = parser.parse_args()

    required = []
    if args.required:
        for item in args.required:
            if "=" in item:
                q, e = item.split("=", 1)
                required.append({"question": q, "expected": e, "kind": "exact"})
            else:
                required.append({"question": item, "kind": "exact"})

    gate = Gate(
        min_accuracy=args.min_accuracy,
        min_numeric_accuracy=args.min_numeric,
        min_exact_accuracy=args.min_exact,
        required_cases=required,
    )
    promoted = promote(
        args.candidate,
        args.best,
        device=args.device,
        gate=gate,
        dry_run=args.dry_run,
    )
    sys.exit(0 if promoted or args.dry_run else 1)


if __name__ == "__main__":
    main()
