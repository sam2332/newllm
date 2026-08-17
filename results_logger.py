"""Persist experiment results for newllm.

Results live in results/ as JSON files with timestamp, config, metrics, sample generations.
"""

import json
import os
import time
from datetime import datetime, timezone


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
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
            results.append(json.load(f))
    return results
