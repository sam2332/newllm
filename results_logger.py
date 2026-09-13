"""Persist experiment results for newllm.

Results live in results/ as JSON files with timestamp, config, metrics, sample generations.
"""

import json
import os
import time
from datetime import datetime, timezone


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LEADERBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LEADERBOARD.md")
os.makedirs(RESULTS_DIR, exist_ok=True)


def utc_now_str():
    return datetime.now(timezone.utc).isoformat()


def save_result(name: str, data: dict) -> str:
    """Write a JSON result file. Returns the path written."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    path = os.path.join(RESULTS_DIR, filename)

    payload = {
        "experiment": name,
        "timestamp": utc_now_str(),
        "data": data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_leaderboard()
    return path


def load_all_results(name_prefix: str = None):
    """Load all saved results, optionally filtered by experiment name prefix."""
    results = []
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith(".json"):
            continue
        if name_prefix and not filename.startswith(name_prefix):
            continue
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_filename"] = filename
        results.append(payload)
    return results


def _metric(result: dict) -> tuple[str, float, bool]:
    """Return (label, score, higher_is_better) for one experiment result."""
    data = result.get("data", {})
    if "accuracy" in data:
        return "accuracy", float(data["accuracy"]), True
    for key in ("best_val_loss", "best_loss", "final_loss"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return key, float(value), False
    return "no metric", 0.0, True


def _rank_results(results: list[dict]) -> dict[int, int]:
    """Rank results inside each experiment family by their primary metric."""
    grouped = {}
    for index, result in enumerate(results):
        grouped.setdefault(result.get("experiment", "unknown"), []).append((index, result))

    ranks = {}
    for entries in grouped.values():
        higher_is_better = _metric(entries[0][1])[2]
        ordered = sorted(
            entries,
            key=lambda entry: _metric(entry[1])[1],
            reverse=higher_is_better,
        )
        rank = 0
        previous_score = None
        for position, (index, result) in enumerate(ordered, start=1):
            score = _metric(result)[1]
            if previous_score is None or score != previous_score:
                rank = position
            ranks[index] = rank
            previous_score = score
    return ranks


def _format_config(data: dict) -> str:
    parts = []
    if "model_size" in data:
        parts.append(f"size={data['model_size']}")
    if "attention_type" in data:
        parts.append(f"attention={data['attention_type']}")
    if data.get("use_moe"):
        parts.append("MoE")
    if "num_experts" in data:
        parts.append(f"experts={data['num_experts']}")
    if "vocab_size" in data:
        parts.append(f"vocab={data['vocab_size']}")
    if "max_len" in data:
        parts.append(f"ctx={data['max_len']}")
    if "d_model" in data:
        parts.append(f"d={data['d_model']}")
    if "n_layers" in data:
        parts.append(f"layers={data['n_layers']}")
    elif "expert_layers" in data:
        parts.append(f"expert-layers={data['expert_layers']}")
    if "n_heads" in data:
        parts.append(f"heads={data['n_heads']}")
    if "d_ff" in data:
        parts.append(f"ff={data['d_ff']}")
    if "num_samples" in data:
        parts.append(f"samples={data['num_samples']}")
    if "iters" in data:
        parts.append(f"iters={data['iters']}")
    if data.get("curriculum"):
        parts.append("curriculum")
    if data.get("math_only"):
        parts.append("math-only")
    replay = data.get("math_replay_fraction", 0.0)
    if replay:
        parts.append(f"math-replay={replay:.0%}")
    if "learning_rate" in data:
        parts.append(f"lr={data['learning_rate']:.0e}")
    return ", ".join(parts) or "-"


def _format_parameters(data: dict) -> str:
    parameters = data.get("model_parameters")
    if not isinstance(parameters, int):
        return "-"
    return f"{parameters / 1_000_000:.2f}M"


def _format_losses(data: dict) -> str:
    fields = []
    for label, key in (("final", "final_loss"), ("best", "best_loss"), ("val", "best_val_loss")):
        value = data.get(key)
        if isinstance(value, (int, float)):
            fields.append(f"{label}={value:.4f}")
    return ", ".join(fields) or "-"


def _format_evaluation(data: dict) -> str:
    fields = []
    if isinstance(data.get("accuracy"), (int, float)):
        fields.append(f"overall={data['accuracy']:.1%}")
    per_kind = data.get("per_kind_accuracy", {})
    if isinstance(per_kind, dict):
        for kind in ("numeric", "exact", "date"):
            value = per_kind.get(kind)
            if isinstance(value, (int, float)):
                fields.append(f"{kind}={value:.1%}")
    return ", ".join(fields) or "-"


def _format_runtime(data: dict) -> str:
    seconds = data.get("elapsed_seconds")
    if not isinstance(seconds, (int, float)):
        return "-"
    return f"{seconds / 60:.1f}m"


def write_leaderboard() -> str:
    """Write a newest-first experiment leaderboard to ``LEADERBOARD.md``.

    Rankings are only comparable within the same experiment family. Agent
    runs rank by accuracy; other runs rank by their lowest available loss.
    """
    results = load_all_results()
    results.sort(key=lambda result: result.get("timestamp", ""), reverse=True)
    ranks = _rank_results(results)
    lines = [
        "# Experiment Leaderboard",
        "",
        "Rows are newest first. `Rank` is computed within each experiment family, "
        "not across different objectives. Agent runs rank by accuracy; other runs "
        "rank by lowest available validation, best, or final loss. A dash indicates "
        "metadata that older result files did not record.",
        "",
        "| Timestamp (UTC) | Experiment | Rank | Primary Metric | Parameters | Configuration | Losses | Evaluation | Runtime | Checkpoint | Result File |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for index, result in enumerate(results):
        metric_name, metric_value, _ = _metric(result)
        data = result.get("data", {})
        timestamp = result.get("timestamp", "unknown").replace("+00:00", "Z")
        filename = f"results/{result.get('_filename', 'unknown.json')}"
        lines.append(
            f"| {timestamp} | {result.get('experiment', 'unknown')} | "
            f"#{ranks[index]} | {metric_name}={metric_value:.4f} | "
            f"{_format_parameters(data)} | {_format_config(data)} | "
            f"{_format_losses(data)} | {_format_evaluation(data)} | "
            f"{_format_runtime(data)} | {data.get('checkpoint', '-')} | "
            f"[{result.get('_filename', 'unknown.json')}]({filename}) |"
        )
    if not results:
        lines.append("| - | - | - | No recorded results | - | - | - | - | - | - | - |")
    lines.append("")
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return LEADERBOARD_PATH
