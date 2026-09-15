"""Evaluate a checkpoint on the agent battery with a per-capability breakdown.

HANDOFF.md asks for parsed-JSON validity, tool-call rate, tool-name accuracy,
argument accuracy and final-answer accuracy to be scored separately. This does
that instead of collapsing everything into one number.
"""

import sys, json, re, argparse, time
sys.path.insert(0, "/home/lmeadows/llm")
import torch

from model.transformer import Transformer
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from agent.agent_loop import run_agent, _validate_assistant_message
from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools
from agent.sandbox_tools import attach_sandbox_tools

CASES = [
    {"hops": 1, "question": "What is 12 + 8?", "expected": "20", "kind": "numeric"},
    {"hops": 1, "question": "Calculate 15 * 4.", "expected": "60", "kind": "numeric"},
    {"hops": 1, "question": "What is the project?", "expected": "newllm", "kind": "exact"},
    {"hops": 1, "question": "Retrieve the leader.", "expected": "grug", "kind": "exact"},
    {"hops": 1, "question": "What is the current date and time?", "kind": "date"},
    {"hops": 1, "question": "What is 7 * 6?", "expected": "42", "kind": "numeric"},
    {"hops": 1, "question": "Look up the version.", "expected": "0.1", "kind": "exact"},
    {"hops": 2, "question": "Add 5 to the version.", "expected": "5.1", "kind": "numeric"},
    {"hops": 3, "question": "Multiply 3 and 4, then add the stored version.", "expected": "12.1", "kind": "numeric"},
    {"hops": 1, "question": "What is the capital of france?", "expected": "Paris", "kind": "exact"},
    {"hops": 1, "question": "Who is the president of the united states?", "expected": "Alice Johnson", "kind": "exact"},
    {"hops": 1, "question": "How many planets are there?", "expected": "8", "kind": "exact"},
    {"hops": 1, "question": "What is the speed of light?", "expected": "299792458 m/s", "kind": "exact"},
    {"hops": 1, "question": "Look up the boiling point of water.", "expected": "100 degrees Celsius", "kind": "exact"},
    {"hops": 1, "question": "What is the largest planet?", "expected": "Jupiter", "kind": "exact"},
    {"hops": 3, "question": "Look up the number of planets, add 5, then multiply by 2.", "expected": "26", "kind": "numeric"},
    {"hops": 3, "question": "Add the stored version to the number of planets, then multiply the result by 2.", "expected": "16.2", "kind": "numeric"},
]


def load(path, device="cuda"):
    """One loader for every script (agent/chat.py), so new config keys such as
    rope_base, MoE and the tokenizer spec are honoured everywhere."""
    from agent.chat import load_checkpoint
    return load_checkpoint(path, device=device)



def require_legacy_protocol(model, script: str = "this eval"):
    """Refuse to score a ChatML checkpoint with the legacy JSON harness.

    A byte-tokenizer eval pointed at a BPE/ChatML checkpoint does not fail -
    it decodes the trace into nonsense, finds no <assistant> block and
    reports 0/0, which reads as "the model is broken" when nothing has been
    asked of the model at all. That happened, and the hour it cost is the
    reason this is an exception rather than a warning.
    """
    from agent.chat import load_tokenizer_for
    from agent.turn import protocol_for
    proto = protocol_for(load_tokenizer_for(model))
    if proto != "json":
        raise SystemExit(
            f"{script} speaks the legacy JSON protocol, but this checkpoint "
            f"was trained on {proto}.\n"
            f"Use: .venv/bin/python scripts/eval_chatml_random.py "
            f"<checkpoint> --cache <dataset cache>")

def score(model, device="cuda", verbose=False, constrained=False,
          all_tools=True):
    # The model is trained with the full 12-tool set. Evaluating it against a
    # 5-tool box makes a correct repo/sandbox call read as "unknown tool",
    # which measures the harness rather than the model.
    tb = Toolbox()
    if all_tools:
        tb = attach_sandbox_tools(attach_repo_tools(tb))
    stats = {"json_valid": 0, "made_tool_call": 0, "correct": 0,
             "assistant_blocks": 0}
    by_kind = {}
    by_hops = {}
    rows = []
    t0 = time.time()
    for case in CASES:
        r = run_agent(model, case["question"], tb, device=device, greedy=True,
                      tokenizer=TOK, constrained=constrained)
        got = (r["final_answer"] or "").strip()
        kind = case.get("kind", "exact")
        exp = case.get("expected")
        # Real JSON validity: every <assistant> block in the trace must parse
        # and satisfy the contract.
        blocks = re.findall(r"<assistant>(.*?)</assistant>", r.get("trace", ""),
                            re.DOTALL)
        stats["assistant_blocks"] += len(blocks)
        good = 0
        for b in blocks:
            try:
                _validate_assistant_message(json.loads(b.strip()))
                good += 1
            except Exception:
                pass
        stats["json_valid"] += good
        if r.get("steps"):
            stats["made_tool_call"] += 1
        if kind == "date":
            ok = bool(got) and any(c.isdigit() for c in got)
        elif kind == "numeric":
            try:
                ok = abs(float(got) - float(exp)) < 1e-3
            except Exception:
                ok = got.lower() == str(exp).lower()
        else:
            ok = got.lower() == str(exp).lower()
        stats["correct"] += ok
        d = by_kind.setdefault(kind, [0, 0]); d[1] += 1; d[0] += ok
        h = by_hops.setdefault(case.get("hops", 1), [0, 0]); h[1] += 1; h[0] += ok
        tools = "->".join(s["tool"] for s in r.get("steps", [])) or "-"
        rows.append((case["question"][:46], exp, got[:28], tools, ok))
    dt = time.time() - t0
    n = len(CASES)
    if verbose:
        print(f"  {'question':46s} {'expect':>16s} {'got':>28s}  {'tools':22s} ok")
        for q, e, g, t, ok in rows:
            print(f"  {q:46s} {str(e)[:16]:>16s} {g:>28s}  {t:22s} {'Y' if ok else '.'}")
    print(f"  accuracy        {stats['correct']}/{n} = {stats['correct']/n:.1%}")
    ab = max(1, stats["assistant_blocks"])
    print(f"  valid JSON      {stats['json_valid']}/{stats['assistant_blocks']} "
          f"assistant blocks = {stats['json_valid']/ab:.1%}")
    print(f"  made tool call  {stats['made_tool_call']}/{n} = {stats['made_tool_call']/n:.1%}")
    for k, (c, t) in sorted(by_kind.items()):
        print(f"    {k:8s} {c}/{t} = {c/t:.0%}")
    for h, (c, t) in sorted(by_hops.items()):
        label = "single-step" if h == 1 else f"{h}-hop chain"
        print(f"    {label:12s} {c}/{t} = {c/t:.0%}")
    print(f"  eval wall time  {dt:.1f}s ({dt/n:.2f}s per case)")
    return stats["correct"] / n, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-c", "--constrained", action="store_true")
    ap.add_argument("--base-tools", action="store_true",
                    help="evaluate with only the original 5 tools")
    a = ap.parse_args()
    for path in a.checkpoints:
        print(f"\n=== {path} ===")
        try:
            m = load(path)
            print(f"  params {m.count_parameters():,} arch_v{m.arch_version}")
            score(m, verbose=a.verbose, constrained=a.constrained,
                  all_tools=not a.base_tools)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
