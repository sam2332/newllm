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
        print("\nabsolute thresholds: PASSED")
    else:
        print("\nabsolute thresholds: FAILED")
        for r in reasons:
            print(f"  - {r}")
    return ok, acc, per_kind, details


def _compare_checkpoints(candidate: tuple, best: tuple) -> tuple:
    """Compare candidate and current best across five promotion checks.

    Returns (wins, checks) where checks is a list of human-readable
    comparison strings and wins is the number of checks the candidate won.
    """
    cand_acc, cand_per_kind, cand_details = candidate
    best_acc, best_per_kind, best_details = best

    cand_numeric = cand_per_kind.get("numeric", 0.0)
    best_numeric = best_per_kind.get("numeric", 0.0)
    cand_exact = cand_per_kind.get("exact", 0.0)
    best_exact = best_per_kind.get("exact", 0.0)

    # Required questions: all DEFAULT_CASES except date (same as PyTest core).
    required_qs = {d["question"] for d in DEFAULT_CASES if d.get("kind") != "date"}
    cand_required = sum(1 for d in cand_details if d["question"] in required_qs and d["ok"])
    best_required = sum(1 for d in best_details if d["question"] in required_qs and d["ok"])
    total_required = len(required_qs)
    cand_required_rate = cand_required / total_required
    best_required_rate = best_required / total_required

    # Composite robustness score as a 5th check.
    cand_robust = cand_acc + cand_numeric + cand_exact
    best_robust = best_acc + best_numeric + best_exact

    checks = [
        ("overall accuracy", cand_acc, best_acc),
        ("numeric accuracy", cand_numeric, best_numeric),
        ("exact accuracy", cand_exact, best_exact),
        ("required-question pass rate", cand_required_rate, best_required_rate),
        ("robustness score (overall+numeric+exact)", cand_robust, best_robust),
    ]

    wins = 0
    report = []
    for name, c_val, b_val in checks:
        won = c_val > b_val
        if won:
            wins += 1
        report.append(
            f"  {name}: candidate {c_val:.4f} vs best {b_val:.4f} -> "
            f"{'win' if won else 'loss/tie'}"
        )
    return wins, report


def promote(
    candidate_path: str,
    best_path: str,
    device: str = "cuda",
    gate: Gate = None,
    dry_run: bool = False,
):
    """Run gate and, if it passes, copy candidate to best_path.

    Promotion checklist:
    1. Candidate passes absolute thresholds.
    2. Candidate wins at least 3 out of 5 metric comparisons against the
       current best checkpoint (or there is no best yet, in which case it
       automatically clears this check).
    If both checks pass, the candidate becomes the new best and the old best
    is kept as a *_prev.pt backup.
    """
    ok, acc, per_kind, details = run_gate(candidate_path, device=device, gate=gate)
    if not ok:
        print(f"\nrejecting {candidate_path}; absolute thresholds not met.")
        return False

    candidate = (acc, per_kind, details)
    best_exists = os.path.exists(best_path)
    if best_exists:
        print(f"\n=== 5-check competition against current best {best_path} ===")
        best_model = load_checkpoint(best_path, device=device)
        best_result = evaluate_agent(best_model, DEFAULT_CASES, device, toolbox=Toolbox())
        print(f"current best overall accuracy: {best_result[0]:.2%}")
        for k, v in best_result[1].items():
            print(f"  {k}: {v:.2%}")

        wins, report = _compare_checkpoints(candidate, best_result)
        for line in report:
            print(line)
        print(f"\nwins: {wins}/5")

        if wins < 3:
            print(
                f"\nrejecting {candidate_path}; needs to win at least 3/5 checks "
                f"against current best, won {wins}/5."
            )
            return False
        print("\ncandidate wins the competition")

    if dry_run:
        print(f"\nDRY RUN: would promote {candidate_path} -> {best_path}")
        return True

    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
    # Keep a backup of the previous best so promotion is reversible.
    if best_exists:
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
