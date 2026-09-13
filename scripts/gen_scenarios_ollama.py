"""Generate complex multi-turn, multi-tool SCENARIO PLANS with a local teacher.

Why plans and not transcripts
-----------------------------
Measured throughput on this box: ~1,077 teacher calls/hour across both Ollama
instances. Driving every hop through the teacher costs one call per hop, so
four hours buys roughly 215 complete traces - useless as a training set. Asking
the teacher for a *plan* costs one call per scenario, and each plan instantiates
into many concrete traces with different facts and operands. The same four hours
buys ~215,000 traces.

It also preserves the invariant the rest of the pipeline depends on: the teacher
never authors a tool result. It supplies the thing templates genuinely cannot
invent - what a realistic 20-step task looks like, how a user redirects
mid-task, where a chain should recover from a failure - while every observation
still comes from the real Toolbox and is verified.

Step kinds are a closed vocabulary, so a plan is executable rather than prose.
Concrete facts, keys and operands are bound at instantiation time, not chosen by
the teacher, which is what keeps every emitted value grounded.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlrequest

sys.path.insert(0, "/home/lmeadows/llm")

ENDPOINTS = ["http://localhost:11434", "http://localhost:11435"]

# Closed vocabulary. A plan referencing anything else is rejected.
STEP_KINDS = {
    "lookup_fact":    "search the web for a factual value",
    "lookup_memory":  "read a stored internal value",
    "compute":        "arithmetic on the previous result",
    "combine":        "arithmetic combining the two previous results",
    "time":           "read the current timestamp",
    "run_code":       "execute a snippet in the sandbox",
    "inspect_code":   "read the agent's own source",
    "failing_call":   "a call that errors, which the next step must recover from",
}

PLAN_PROMPT = """You are designing training scenarios for a tool-using AI agent.

The agent has these step kinds, and ONLY these:
{kinds}

Design {n} DIFFERENT realistic multi-turn scenarios. Each scenario is a
conversation of 2 to 4 user turns. Each user turn is resolved by {lo} to {hi}
steps. Across the whole conversation, later turns must depend on earlier
results (pronouns like "that", "it", references to an earlier answer).

Rules:
- "user" is what a real person would type. Natural, varied, sometimes terse.
- Do NOT invent specific numbers, facts, or answers. Describe intent only.
- Use "failing_call" in roughly a third of scenarios, always followed by at
  least one more step that recovers.
- Vary domain: research, data wrangling, code inspection, calculation, mixed.
- Vary tone: terse, polite, frustrated, curious.

Return ONLY a JSON array of {n} objects shaped exactly like:
[
  {{"title": "short label",
    "turns": [
      {{"user": "what the person types",
       "steps": [{{"kind": "lookup_fact"}}, {{"kind": "compute"}}]}},
      {{"user": "a follow-up that refers back to the earlier result",
       "steps": [{{"kind": "compute"}}]}}
    ]}}
]"""


def ollama_chat(prompt, model, endpoint, temperature=1.0, timeout=420):
    body = json.dumps({
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": 4096},
    }).encode()
    req = urlrequest.Request(f"{endpoint}/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("message", {}).get("content", "")


def extract_json_array(text):
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("[")
    if start == -1:
        return []
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return []
    return []


def validate_plan(plan):
    """Return a normalized plan, or None if it is not executable.

    The teacher is untrusted: it may invent step kinds, omit fields, or write
    prose where structure is required. Anything that would not execute against
    the real Toolbox is dropped here rather than discovered mid-training.
    """
    if not isinstance(plan, dict):
        return None
    turns_in = plan.get("turns")
    if not isinstance(turns_in, list) or not 1 <= len(turns_in) <= 6:
        return None
    turns = []
    for turn in turns_in:
        if not isinstance(turn, dict):
            return None
        user = turn.get("user")
        steps_in = turn.get("steps")
        if not isinstance(user, str) or not 3 < len(user) < 400:
            return None
        if not isinstance(steps_in, list) or not 1 <= len(steps_in) <= 40:
            return None
        steps = []
        for step in steps_in:
            kind = step.get("kind") if isinstance(step, dict) else step
            if kind not in STEP_KINDS:
                return None
            steps.append({"kind": kind})
        # A failing call must be followed by a recovery step in the same turn.
        for i, step in enumerate(steps):
            if step["kind"] == "failing_call" and i == len(steps) - 1:
                steps.append({"kind": "compute"})
                break
        turns.append({"user": user.strip(), "steps": steps})
    title = plan.get("title")
    return {"title": title if isinstance(title, str) else "untitled",
            "turns": turns}


def job(index, model, endpoint, per_call, lo, hi):
    prompt = PLAN_PROMPT.format(n=per_call, kinds="\n".join(
        f"  {k}: {v}" for k, v in STEP_KINDS.items()), lo=lo, hi=hi)
    try:
        raw = ollama_chat(prompt, model, endpoint,
                          temperature=1.05 + (index % 3) * 0.05)
    except Exception as exc:                                  # noqa: BLE001
        return [], f"{type(exc).__name__}"
    plans = [p for p in (validate_plan(x) for x in extract_json_array(raw)) if p]
    return plans, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--out", default="data/scenarios.json")
    ap.add_argument("--hours", type=float, default=2.0,
                    help="wall-clock budget; stops cleanly when reached")
    ap.add_argument("--per-call", type=int, default=6,
                    help="scenarios requested per teacher call")
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # Resume: never throw away work already paid for in GPU time.
    plans = []
    if os.path.exists(args.out):
        try:
            plans = json.load(open(args.out))
            print(f"resuming from {len(plans)} existing scenarios")
        except (OSError, json.JSONDecodeError):
            plans = []

    deadline = time.time() + args.hours * 3600
    t0 = time.time()
    index, errors, rejected = 0, 0, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"budget {args.hours}h, {args.workers} workers across "
          f"{len(ENDPOINTS)} instances, {args.per_call} scenarios/call")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = set()
        while time.time() < deadline or pending:
            while len(pending) < args.workers and time.time() < deadline:
                pending.add(ex.submit(job, index, args.model,
                                      ENDPOINTS[index % len(ENDPOINTS)],
                                      args.per_call, args.min_steps,
                                      args.max_steps))
                index += 1
            done, pending = _wait_any(pending)
            for fut in done:
                got, err = fut.result()
                if err:
                    errors += 1
                    continue
                rejected += max(0, args.per_call - len(got))
                plans.extend(got)
            # Checkpoint after every batch; a power cut must not cost hours.
            tmp = args.out + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(plans, fh, indent=1)
            os.replace(tmp, args.out)
            elapsed = time.time() - t0
            remain = max(0, deadline - time.time())
            print(f"  [{elapsed/60:5.1f}m] {len(plans):6,d} scenarios "
                  f"({len(plans)/max(elapsed,1)*3600:,.0f}/h) "
                  f"errors={errors} rejected={rejected} "
                  f"remaining={remain/60:.0f}m", flush=True)

    steps = sum(len(t["steps"]) for p in plans for t in p["turns"])
    turns = sum(len(p["turns"]) for p in plans)
    print(f"\n{len(plans):,} scenarios, {turns:,} turns, {steps:,} steps")
    print(f"  mean {steps/max(1,len(plans)):.1f} steps/scenario, "
          f"{turns/max(1,len(plans)):.1f} turns/scenario")
    print(f"saved -> {args.out}")


def _wait_any(pending):
    from concurrent.futures import wait, FIRST_COMPLETED
    done, still = wait(pending, return_when=FIRST_COMPLETED, timeout=600)
    return done, set(still)


if __name__ == "__main__":
    main()
