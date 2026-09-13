"""Evaluate the chat model on multi-turn conversation skills.

Single-turn accuracy says nothing about whether a model can hold a
conversation. These cases isolate the abilities the chat split exists for:

  coref       "multiply that by 3"      - resolve a pronoun to the last result
  ellipsis    "and Japan?"              - recover an omitted verb and object
  backref     "what was my first answer" - reach past the most recent turn
  switch      "different question ..."   - do NOT reuse the previous entity
  no_tool     answerable from context    - resist calling a tool at all

Each case scores only its FINAL turn; earlier turns exist to build context.
"""

import sys, argparse, time
sys.path.insert(0, "/home/lmeadows/llm")
import torch

from model.transformer import Transformer
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from agent.agent_loop import run_chat
from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools
from agent.sandbox_tools import attach_sandbox_tools

CASES = [
    {"skill": "coref", "turns": ["What is 12 + 8?", "Multiply that by 3."],
     "expected": "60"},
    {"skill": "coref", "turns": ["What is 40 + 2?", "Add 8 to that."],
     "expected": "50"},
    {"skill": "coref", "turns": ["Compute 9 * 9.", "Subtract 1 from that."],
     "expected": "80"},
    {"skill": "ellipsis",
     "turns": ["What is the capital of france?", "And japan?"],
     "expected": "Tokyo"},
    {"skill": "ellipsis",
     "turns": ["Look up the leader.", "And the project?"],
     "expected": "newllm"},
    {"skill": "backref",
     "turns": ["What is 30 + 5?", "Look up the leader.",
               "What was the result of my first question?"],
     "expected": "35"},
    {"skill": "backref",
     "turns": ["What is 7 * 7?", "What is the capital of germany?",
               "Remind me what the first answer was."],
     "expected": "49"},
    {"skill": "switch",
     "turns": ["Compute 6 * 7.", "Different question - what is the largest planet?"],
     "expected": "Jupiter"},
    {"skill": "switch",
     "turns": ["Look up the version.", "Different question - what is 10 + 10?"],
     "expected": "20"},
    {"skill": "no_tool",
     "turns": ["What is 100 + 1?", "What number did we get at the start?"],
     "expected": "101"},
]


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    n_layers = len(set(k.split(".")[1] for k in ck["model"]
                       if k.startswith("layers.")))
    model = Transformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=n_layers,
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], max_len=cfg["max_len"],
        dropout=0.0, use_rope=cfg.get("use_rope", True),
        tie_weights=cfg.get("tie_weights", False),
        arch_version=cfg.get("arch_version", 1),
        n_kv_heads=cfg.get("n_kv_heads"), qk_norm=cfg.get("qk_norm", True))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def score(model, device="cuda", verbose=False, constrained=False,
          all_tools=True):
    # The model is trained with the full 12-tool set. Evaluating it against a
    # 5-tool box makes a correct repo/sandbox call read as "unknown tool",
    # which measures the harness rather than the model.
    tb = Toolbox()
    if all_tools:
        tb = attach_sandbox_tools(attach_repo_tools(tb))
    by_skill, rows, correct = {}, [], 0
    t0 = time.time()
    for case in CASES:
        results = run_chat(model, case["turns"], tb, device=device,
                           greedy=True, tokenizer=TOK, constrained=constrained)
        got = (results[-1]["final_answer"] or "").strip()
        exp = case["expected"]
        try:
            ok = abs(float(got) - float(exp)) < 1e-3
        except ValueError:
            ok = got.lower() == exp.lower()
        correct += ok
        d = by_skill.setdefault(case["skill"], [0, 0]); d[1] += 1; d[0] += ok
        tools = "|".join("->".join(s["tool"] for s in r["steps"]) or "-"
                         for r in results)
        rows.append((" / ".join(case["turns"])[:58], exp, got[:20], tools, ok))
    n = len(CASES)
    if verbose:
        for q, e, g, t, ok in rows:
            print(f"  {q:58s} exp={e:>8s} got={g:>20s} [{t}] "
                  f"{'Y' if ok else '.'}")
    print(f"  multi-turn accuracy {correct}/{n} = {correct/n:.1%}")
    for k, (c, t) in sorted(by_skill.items()):
        print(f"    {k:9s} {c}/{t} = {c/t:.0%}")
    print(f"  wall time {time.time()-t0:.1f}s")
    return correct / n


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
            print(f"  params {m.count_parameters():,} arch_v{m.arch_version} "
                  f"max_len={m.max_len}")
            score(m, verbose=a.verbose, constrained=a.constrained,
                  all_tools=not a.base_tools)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
