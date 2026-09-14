"""Does the model obey a system prompt? Persona voice and output-format rules.

Runs the same client loop the server exposes (``agent/local_client.py``),
with a system message, and scores two things separately:

* **task** - the grounded answer is present (the tool loop still works);
* **compliance** - for a format rule, the rule's own deterministic check
  (``agent/format_rules.py``); for a persona, the reply is more than the bare
  answer and, with ``--judge``, the teacher says it is in character.

Reports n/N per rule and per persona, plus a multi-turn persona case where
the voice must hold across three questions in one conversation.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_system.py checkpoints_X/agent_best.pt
"""

import argparse
import json
import random
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chat import load_checkpoint, load_tokenizer_for
from agent.format_rules import RULES, verify
from agent.local_client import run_tool_loop
from agent.tools import Toolbox

QUESTIONS = [
    ("What is 12 + 8?", "20"),
    ("What is 7 * 6?", "42"),
    ("What is the capital of france?", "Paris"),
    ("Look up the version.", "0.1"),
    ("What is the largest planet?", "Jupiter"),
    ("Retrieve the leader.", "grug"),
]


def answer_present(content: str, expected: str, rule_id=None) -> bool:
    text = content or ""
    if rule_id == "json_only":
        try:
            text = str(json.loads(text).get("answer", ""))
        except (json.JSONDecodeError, AttributeError):
            return False
    return expected.lower() in text.lower()


def judge_in_character(persona, reply, model_name, endpoint):
    from scripts.gen_data_ollama import ollama_chat
    prompt = (f"Character: {persona['name']} - {persona.get('role', '')}. Voice: "
              f"{persona.get('voice', '')}.\nSystem prompt: {persona['system_prompt']}\n\n"
              f"Reply to judge:\n{reply}\n\nIs the reply written in this character's voice? "
              "Answer with exactly one word: yes or no.")
    out = ollama_chat(prompt, model_name, endpoint, temperature=0.0, num_predict=8)
    return out.strip().lower().startswith("yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--personas", default="data/personas.json")
    ap.add_argument("--n-personas", type=int, default=8)
    ap.add_argument("--n-questions", type=int, default=4)
    ap.add_argument("--judge", action="store_true", help="teacher judges persona voice")
    ap.add_argument("--judge-model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--judge-endpoint", default="http://localhost:11434")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint, device=args.device)
    tok = load_tokenizer_for(model)
    rng = random.Random(args.seed)
    questions = QUESTIONS[:args.n_questions]

    def ask(system, convo_questions):
        tb = Toolbox()
        messages = [{"role": "system", "content": system}] if system else []
        replies = []
        for q in convo_questions:
            messages.append({"role": "user", "content": q})
            out = run_tool_loop(model, tok, messages, tb, device=args.device)
            messages = out["messages"]
            replies.append(out["content"])
        return replies

    # ---- format rules ------------------------------------------------------
    print("format rules (task correct / rule obeyed):")
    fmt_task = fmt_ok = fmt_n = 0
    for rule in RULES:
        t_ok = r_ok = 0
        for q, expected in questions:
            reply = ask(rng.choice(rule["prompts"]), [q])[0]
            t = answer_present(reply, expected, rule["id"])
            r = verify(rule["id"], reply)
            t_ok += t; r_ok += r
            if args.verbose:
                print(f"    {rule['id']:16s} {q[:28]!r:30s} task={int(t)} rule={int(r)} {reply[:70]!r}")
        fmt_task += t_ok; fmt_ok += r_ok; fmt_n += len(questions)
        print(f"  {rule['id']:16s} task {t_ok}/{len(questions)}  rule {r_ok}/{len(questions)}")
    print(f"  {'TOTAL':16s} task {fmt_task}/{fmt_n}  rule {fmt_ok}/{fmt_n}")

    # ---- personas ----------------------------------------------------------
    try:
        personas = json.load(open(args.personas))
    except FileNotFoundError:
        personas = []
    if not personas:
        print("\nno persona pool at", args.personas)
        return
    personas = rng.sample(personas, min(args.n_personas, len(personas)))
    print(f"\npersonas (task correct / styled{' / judged in character' if args.judge else ''}):")
    p_task = p_styled = p_judge = p_n = 0
    for persona in personas:
        t_ok = s_ok = j_ok = 0
        for q, expected in questions:
            reply = ask(persona["system_prompt"], [q])[0]
            t = answer_present(reply, expected)
            styled = bool(reply.strip()) and reply.strip().lower() != expected.lower() \
                and len(reply.strip()) > len(expected) + 6
            j = judge_in_character(persona, reply, args.judge_model,
                                   args.judge_endpoint) if args.judge else False
            t_ok += t; s_ok += styled; j_ok += j
            if args.verbose:
                print(f"    {persona['name'][:18]:18s} {q[:24]!r:26s} task={int(t)} "
                      f"styled={int(styled)} {reply[:70]!r}")
        p_task += t_ok; p_styled += s_ok; p_judge += j_ok; p_n += len(questions)
        line = f"  {persona['name'][:24]:24s} task {t_ok}/{len(questions)}  styled {s_ok}/{len(questions)}"
        if args.judge:
            line += f"  judged {j_ok}/{len(questions)}"
        print(line)
    line = f"  {'TOTAL':24s} task {p_task}/{p_n}  styled {p_styled}/{p_n}"
    if args.judge:
        line += f"  judged {p_judge}/{p_n}"
    print(line)

    # ---- multi-turn persona consistency ------------------------------------
    print("\nmulti-turn persona (3 questions in one conversation; styled on every turn):")
    held = 0
    for persona in personas[:4]:
        replies = ask(persona["system_prompt"], [q for q, _ in questions[:3]])
        ok = all(len(r.strip()) > len(e) + 6 for r, (_, e) in zip(replies, questions[:3]))
        held += ok
        print(f"  {persona['name'][:24]:24s} {'held' if ok else 'dropped'}")
    print(f"  TOTAL held {held}/{min(4, len(personas))}")


if __name__ == "__main__":
    main()
