"""Smoke test for the new fake web_search tool and dataset traces."""

from agent.tools import Toolbox
from agent.agent_dataset import generate_agent_dataset, generate_simple_agent_dataset, _web_then_math
import random


def main():
    tb = Toolbox()
    print("=== web_search tool samples ===")
    for q in ["capital of france", "president of the united states", "number of planets", "speed of light"]:
        print(f"  {q} -> {tb.run_json({'tool': 'web_search', 'args': {'query': q}})}")

    print("\n=== simple dataset (4 samples) ===")
    for s in generate_simple_agent_dataset(num_samples=4, seed=1):
        print(s.replace("\x03", "<EOT>"))
        print("-" * 40)

    print("\n=== full dataset multi-step web sample ===")
    rng = random.Random(7)
    print(_web_then_math(rng, use_json_tools=True).replace("\x03", "<EOT>"))

    print("\n=== full dataset sample counts ===")
    full = generate_agent_dataset(num_samples=1000, seed=42)
    kinds = {"web_search": 0, "calc": 0, "search_memory": 0, "now": 0, "multi": 0}
    for s in full:
        if "web_search" in s:
            kinds["web_search"] += 1
        if '"tool":"calc"' in s:
            kinds["calc"] += 1
        if '"tool":"search_memory"' in s:
            kinds["search_memory"] += 1
        if '"tool":"now"' in s:
            kinds["now"] += 1
        if s.count("Action:") > 1:
            kinds["multi"] += 1
    for k, v in kinds.items():
        print(f"  {k}: {v} / {len(full)}")


if __name__ == "__main__":
    main()
