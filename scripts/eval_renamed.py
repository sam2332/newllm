"""Test whether the agent relies on tool SEMANTICS or memorized tool NAMES.

If the model has learned "what a calculator is", renaming calc -> compute_expr
should cost little. If it has memorized the literal string "calc", renaming is
catastrophic. The difference decides whether the model can ever serve a user
who brings their own tools.
"""

import argparse
import sys

sys.path.insert(0, "/home/lmeadows/llm")
import torch

from agent.tools import Toolbox
from agent.agent_loop import run_agent
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from scripts.eval_agent import CASES, load

# Same semantics, different surface names - what any real user would hand us.
RENAMES = {
    "calc": "compute_expression",
    "search_memory": "lookup_stored_value",
    "web_search": "query_knowledge_base",
    "now": "current_timestamp",
}


def renamed_toolbox(mapping):
    """A Toolbox whose tools carry different names but identical behaviour."""
    tb = Toolbox()
    tools = {}
    for old, spec in tb.tools.items():
        new = mapping.get(old, old)
        tools[new] = spec
    tb.tools = tools
    return tb


def score(model, toolbox, device="cuda", label=""):
    correct, called = 0, 0
    for case in CASES:
        r = run_agent(model, case["question"], toolbox, device=device,
                      greedy=True, tokenizer=TOK)
        got = (r["final_answer"] or "").strip()
        exp = case.get("expected")
        kind = case.get("kind", "exact")
        if r.get("steps"):
            called += 1
        if kind == "date":
            ok = bool(got) and any(c.isdigit() for c in got)
        elif kind == "numeric":
            try:
                ok = abs(float(got) - float(exp)) < 1e-3
            except (TypeError, ValueError):
                ok = got.lower() == str(exp).lower()
        else:
            ok = got.lower() == str(exp).lower()
        correct += ok
    n = len(CASES)
    print(f"  {label:24s} accuracy {correct}/{n} = {correct/n:5.1%}   "
          f"tool calls landed {called}/{n}")
    return correct / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    a = ap.parse_args()
    m = load(a.checkpoint)
    print(f"{a.checkpoint}  ({m.count_parameters():,} params)\n")
    base = score(m, Toolbox(), label="original names")
    ren = score(m, renamed_toolbox(RENAMES), label="renamed tools")
    print(f"\n  drop: {base:.1%} -> {ren:.1%}  "
          f"({(ren-base)*100:+.1f} points)")
    if ren < base * 0.5:
        print("  => the model is matching NAMES, not semantics.")
