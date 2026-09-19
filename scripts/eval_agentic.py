"""Does the model still behave like an agent at twenty steps?

Every existing harness measures depth against the *training distribution*.
``eval_chatml_random.py`` buckets held-out traces by tool-call count, but
those traces were composed by the same generators the model trained on, so a
deep bucket there is a deep bucket of familiar shapes.
``diag_deep_chains.py`` replays those same traces to classify failures, and
speaks the legacy protocol besides. Neither can answer "hand it a task it has
never seen that genuinely needs N calls - does it get there?"

This does. Tasks are generated with ground truth computed in Python, at a
depth the caller chooses, and scored on three separate things, because a
single percentage hides which one broke:

1. **solved** - did the final answer match the ground truth
2. **hops** - how many calls it used against the minimum the task needs, which
   is how early-stopping and looping show up
3. **where it first went wrong** - the failure taxonomy below, in the spirit
   of ``diag_deep_chains.py``, because "40% at depth 10" is not actionable and
   "it stops calling tools after hop 6" is

**The ground truth never comes from the model or a teacher.** Each task knows
its answer because Python computed it, and every observation comes from the
real ``Toolbox``/``VirtualWorkspace`` - the same invariant the dataset holds.

Three families, each scaling to any depth:

- ``chain``  - arithmetic where step k needs step k-1's result. The only one
  with a true sequential dependency: you cannot reorder or parallelise it, so
  it is the honest test of chaining. Operands are large enough that answering
  from mental arithmetic is implausible.
- ``ledger`` - write N values into a workspace, then aggregate them. Tests
  that state built up over many calls survives to the end.
- ``lookup`` - retrieve N keys from memory and combine them. Breadth rather
  than depth: N independent calls, then stop. Tests persistence and knowing
  when to quit.

``--rename`` draws opaque surface names from ``agent/tool_schema.py`` instead
of the canonical ones. Leave it off for the headline number; turn it on to
separate "can chain" from "can chain a tool it has not memorised", which is
the distinction ``scripts/eval_renamed.py`` exists to protect.

Usage:

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_agentic.py \\
        checkpoints_sft_chat/agent_best.pt --depths 2,5,10,15,20 -n 8
"""

import argparse
import json
import random
import re
import sys

sys.path.insert(0, "/home/lmeadows/llm")
import torch

from agent.chat import load_tokenizer_for
from agent.chatml import IM_END, render
from agent.tool_schema import NAME_POOLS
from agent.sandbox_tools import docker_available, run_python
from agent.tools import Toolbox
from agent.turn import generate_turn, sampler_from_options
from agent.virtual_workspace import VirtualWorkspace
from scripts.eval_agent import load

# Which canonical tools each family needs bound. Anything else is a
# distractor the model must decline to call.
FAMILY_TOOLS = {
    "chain": ["calc"],
    "ledger": ["write_file", "read_file", "list_files", "calc"],
    "lookup": ["search_memory", "calc"],
    "project": ["write_file", "read_file", "list_files"],
}


class Task:
    """One generated task and everything needed to score it."""

    def __init__(self, family, depth, prompt, answer, min_hops, memory=None,
                 wants=None):
        self.family = family
        self.depth = depth
        self.prompt = prompt
        self.answer = str(answer)
        self.min_hops = min_hops
        self.memory = memory or {}
        self.wants = wants or []      # filenames a project task must produce


# -- task generators ---------------------------------------------------------

def build_chain_prompt(start, words):
    body = ", then ".join(words)
    return (f"Start with {start}. Then {body}. "
            f"Use the calculator for every single step - do not do the "
            f"arithmetic yourself. Report only the final number.")


def gen_chain(depth, rng):
    value = start = rng.randint(23, 99)
    words = []
    for _ in range(depth):
        if abs(value) < 10_000 and rng.random() < 0.3:
            n = rng.randint(3, 7)
            value *= n
            words.append(f"multiply by {n}")
        elif rng.random() < 0.5:
            n = rng.randint(17, 499)
            value += n
            words.append(f"add {n}")
        else:
            n = rng.randint(17, 499)
            value -= n
            words.append(f"subtract {n}")
    return Task("chain", depth, build_chain_prompt(start, words), value, depth)


def gen_ledger(depth, rng):
    """Write ``depth`` readings to their own files, then total them.

    The minimum is ``depth`` writes plus one calc; a model that re-reads every
    file to be sure is not wrong, just slower, which is why hops are reported
    next to the answer rather than folded into it.
    """
    vals = [rng.randint(10, 999) for _ in range(depth)]
    lines = ", ".join(f"reading_{i+1}.txt = {v}" for i, v in enumerate(vals))
    prompt = (f"In the workspace, create one file per reading, each containing "
              f"just its number: {lines}. "
              f"When every file is written, report the sum of all the "
              f"readings. Report only the final number.")
    return Task("ledger", depth, prompt, sum(vals), depth + 1)


def gen_lookup(depth, rng):
    """Retrieve ``depth`` keys from memory, then sum them.

    The memory is generated per task, so the values cannot be answered from
    the weights - ``Toolbox.DEFAULT_MEMORY`` would be memorisable and is only
    seven keys deep anyway.
    """
    keys = [f"sensor_{rng.randint(100, 999)}_{i}" for i in range(depth)]
    memory = {k: str(rng.randint(10, 499)) for k in keys}
    total = sum(int(v) for v in memory.values())
    listed = ", ".join(keys)
    prompt = (f"Look up each of these keys in memory: {listed}. "
              f"Then report the sum of all their values. "
              f"Report only the final number.")
    return Task("lookup", depth, prompt, total, depth + 1, memory=memory)


# One-line operations, each of which becomes its own module. Kept trivial on
# purpose: the difficulty under test is "write several files that import each
# other and run", not "implement an algorithm".
OPS = [
    ("double",  "returns x multiplied by 2",  lambda x: x * 2),
    ("triple",  "returns x multiplied by 3",  lambda x: x * 3),
    ("add_ten", "returns x plus 10",          lambda x: x + 10),
    ("negate",  "returns x with its sign flipped", lambda x: -x),
    ("square",  "returns x multiplied by itself", lambda x: x * x),
    ("sub_five", "returns x minus 5",         lambda x: x - 5),
]


def gen_project(depth, rng):
    """Write ``depth`` modules plus a main.py that imports and chains them.

    This is the only family where the answer is established by *running* the
    result. The workspace snapshot is materialised inside the Docker sandbox
    and executed, so "did the imports resolve" is decided by CPython, not by
    the harness pattern-matching the source. Scoring is three progressive
    stages - files present, program runs, output correct - because a model
    that writes four good modules and a broken main.py has done something
    quite different from one that wrote nothing, and a single boolean hides
    that entirely.
    """
    chosen = rng.sample(OPS, min(depth, len(OPS)))
    while len(chosen) < depth:                      # depths beyond the pool
        chosen.append(rng.choice(OPS))
    start = rng.randint(3, 20)
    value = start
    mods, lines = [], []
    seen = {}
    for i, (fn, human, f) in enumerate(chosen):
        # A repeated op needs its own module name, or the second write
        # silently overwrites the first.
        seen[fn] = seen.get(fn, 0) + 1
        mod = f"op_{i+1}_{fn}" if seen[fn] > 1 else f"op_{fn}"
        mods.append((mod, fn, human))
        lines.append(f"- `{mod}.py` defining a function `{fn}(x)` that {human}")
        value = f(value)
    imports = ", then ".join(f"`{fn}` from `{mod}.py`" for mod, fn, _ in mods)
    prompt = (
        "Create a small Python project in the workspace. Write these files:\n"
        + "\n".join(lines) + "\n"
        f"- `main.py` which imports {imports}, starts with the value {start}, "
        f"applies each of those functions to it in that order, and prints "
        f"only the resulting number.\n"
        "Write every file with the workspace tools. When all the files exist, "
        "say that the project is complete."
    )
    wants = [f"{mod}.py" for mod, _, _ in mods] + ["main.py"]
    return Task("project", depth, prompt, value, depth + 1, wants=wants)


GENERATORS = {"chain": gen_chain, "ledger": gen_ledger, "lookup": gen_lookup,
              "project": gen_project}


def execute_project(ws, timeout=20):
    """Run the workspace's ``main.py`` inside the Docker sandbox.

    Nothing is mounted, so the files are shipped in as a JSON literal and
    written to the container's tmpfs before ``main.py`` is run as a
    subprocess - which is what makes ``from op_double import double`` resolve
    the way it would on a real disk. The sandbox keeps its usual guarantees
    (no network, read-only root, dropped capabilities): the code being run
    here was written by the model, so it does not run on the host.

    Returns ``(ran, stdout, stderr)``.
    """
    files = ws.snapshot()
    if not files:
        return False, "", "empty workspace"
    boot = (
        "import os, sys, json, subprocess\n"
        f"files = json.loads({json.dumps(json.dumps(files))})\n"
        "os.makedirs('/tmp/proj', exist_ok=True)\n"
        "os.chdir('/tmp/proj')\n"
        "for name, body in files.items():\n"
        "    d = os.path.dirname(name)\n"
        "    if d: os.makedirs(d, exist_ok=True)\n"
        "    open(name, 'w').write(body if isinstance(body, str) else str(body))\n"
        "if not os.path.exists('main.py'):\n"
        "    print('NORUN'); print('OUT'); print('ERR no main.py'); raise SystemExit\n"
        "try:\n"
        "    r = subprocess.run([sys.executable, 'main.py'], capture_output=True,"
        f"                      text=True, timeout={timeout})\n"
        "except Exception as exc:\n"
        "    print('NORUN'); print('OUT'); print('ERR', exc); raise SystemExit\n"
        "print('RC', r.returncode)\n"
        "print('OUT', r.stdout.strip()[:400])\n"
        "print('ERR', r.stderr.strip()[-400:])\n"
    )
    out = run_python(boot, timeout=timeout + 15)
    rc, stdout, stderr = None, "", ""
    for line in (out or "").splitlines():
        if line.startswith("RC "):
            rc = line[3:].strip()
        elif line.startswith("OUT"):
            stdout = line[4:].strip()
        elif line.startswith("ERR"):
            stderr = line[4:].strip()
    if rc is None:
        return False, stdout, stderr or (out or "")[:200]
    return rc == "0", stdout, stderr


# -- tool binding ------------------------------------------------------------

def bind(task, rng, rename=False):
    """Bind exactly the tools this family needs, plus two distractors.

    Returns ``(toolbox, schemas, workspace)``. With ``rename`` the surface
    name is drawn from the opaque end of the pool, so the description is the
    only signal - the condition ``eval_renamed.py`` treats a large gap on as a
    blocking defect.
    """
    tb = Toolbox(memory=dict(task.memory) if task.memory else None)
    ws = VirtualWorkspace()
    specs = ws.specs()
    available = dict(tb.tools)
    available.update(specs)

    wanted = list(FAMILY_TOOLS[task.family])
    # Two distractors the task does not need, so selection is required rather
    # than "call the only thing on offer".
    pool = [n for n in available if n not in wanted and n != "finish"]
    wanted += rng.sample(pool, min(2, len(pool)))

    bound, surface_of = {}, {}
    for canonical in wanted:
        spec = available.get(canonical)
        if not spec:
            continue
        surface = canonical
        if rename and canonical in NAME_POOLS:
            # Prefer the opaque tail of the pool; those are the names that
            # cannot be guessed from the capability.
            surface = rng.choice(NAME_POOLS[canonical][2:]
                                 or NAME_POOLS[canonical])
        bound[surface] = spec
        surface_of[canonical] = surface

    # ``finish`` is always bound. The model was trained to emit it, and an
    # unbound one would be scored as a hallucinated name - measuring the
    # harness rather than the model.
    if "finish" not in bound:
        bound["finish"] = Toolbox().tools["finish"]

    tb.tools = bound
    schemas = [{"type": "function",
                "function": {"name": name,
                             "description": spec["description"],
                             "parameters": spec["parameters"]}}
               for name, spec in bound.items()]
    return tb, schemas, ws


# -- running -----------------------------------------------------------------

class Run:
    """What happened on one task."""

    def __init__(self):
        self.calls = []          # (surface_name, args, observation)
        self.answer = None
        self.hops = 0
        self.unknown = []        # names the model invented
        self.errors = 0
        self.malformed = 0       # opened <tool_call> and never closed it


def run_task(model, tok, task, tb, schemas, *, device, max_steps, max_new,
             seed):
    allowed = [s["function"]["name"] for s in schemas]
    sampler = sampler_from_options({"temperature": 0})
    messages = [{"role": "user", "content": task.prompt}]
    run = Run()
    for hop in range(max_steps):
        text, _ = render(messages, schemas, add_generation_prompt=True)
        ids = tok.encode(text)
        res = generate_turn(model, ids, sampler=sampler, tokenizer=tok,
                            device=device, max_new=max_new, protocol="chatml",
                            allowed_names=allowed, seed=seed,
                            stop_texts=(IM_END,))
        if res.kind == "tool_call" and res.tool_call:
            name = res.tool_call.get("name")
            args = res.tool_call.get("arguments") or {}
            messages.append({"role": "assistant", "content": res.thought or "",
                             "tool_calls": [{"function": {"name": name,
                                                          "arguments": args}}]})
            spec = tb.tools.get(name)
            if not spec:
                run.unknown.append(name)
                obs = f"unknown tool: {name}"
            else:
                try:
                    obs = spec["execute"](args)
                except Exception as exc:                    # noqa: BLE001
                    obs = f"error: {exc}"
                    run.errors += 1
            run.calls.append((name, args, str(obs)))
            messages.append({"role": "tool", "content": str(obs)[:12288]})
            continue
        if res.kind == "malformed":
            # The model opened a <tool_call> and never closed it - on this
            # checkpoint, by looping on a repeated line until the token budget
            # runs out. Scoring that as "gave a final answer" credits it with a
            # decision it never made, so it gets its own class.
            run.malformed += 1
            run.answer = (res.response or "").strip()
            run.hops = hop
            return run
        run.answer = (res.response or "").strip()
        run.hops = hop
        return run
    run.hops = max_steps
    return run


NUM = re.compile(r"-?\d[\d,]*")


def scored(run, task):
    """Did the final answer contain the right number?

    Numeric containment rather than exact match: the model is prone to
    wrapping a correct figure in a sentence, and this eval is measuring
    whether it got there, not whether it obeyed "only the number" (that is
    reported separately as ``verbose`` noise).
    """
    if not run.answer:
        return False
    want = int(task.answer)
    for m in NUM.finditer(run.answer):
        try:
            if int(m.group(0).replace(",", "")) == want:
                return True
        except ValueError:
            continue
    return False


def score_project(run, task, ws):
    """Three progressive stages, reported separately.

    A model that writes every module but botches ``main.py`` has got much
    further than one that wrote nothing, and collapsing both to "failed"
    throws away the only signal that says which to fix.
    """
    snap = ws.snapshot() or {}
    have = set(snap)
    missing = [f for f in task.wants if f not in have]
    files_ok = not missing
    ran, stdout, stderr = (False, "", "no files")
    if "main.py" in have:
        ran, stdout, stderr = execute_project(ws)
    out_ok = False
    if ran:
        want = int(task.answer)
        for m in NUM.finditer(stdout):
            try:
                if int(m.group(0).replace(",", "")) == want:
                    out_ok = True
                    break
            except ValueError:
                continue
    return {"files_ok": files_ok, "missing": missing, "ran": ran,
            "out_ok": out_ok, "stdout": stdout, "stderr": stderr,
            "n_files": len(have)}


def classify_project(run, info):
    if info["out_ok"]:
        return "solved"
    if run.unknown:
        return "unknown_tool"
    if run.malformed:
        return "malformed_call"
    if info["n_files"] == 0:
        return "no_files"
    if info["missing"]:
        return "missing_files"
    if not info["ran"]:
        return "run_error"
    return "wrong_output"


def classify(run, task, ok):
    """Name the first thing that went wrong, most specific first."""
    if ok:
        return "solved"
    if run.unknown:
        return "unknown_tool"
    if run.malformed:
        return "malformed_call"
    if not run.calls:
        return "no_tool_use"
    if run.answer is None:
        return "budget"          # never stopped talking to tools
    if run.errors:
        return "tool_error"
    if len(run.calls) < task.min_hops:
        return "early_stop"
    return "wrong_value"         # used the tools, botched the chaining


ORDER = ["solved", "wrong_value", "wrong_output", "run_error",
         "missing_files", "no_files", "early_stop", "budget", "malformed_call",
         "unknown_tool", "no_tool_use", "tool_error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--depths", default="2,5,10,15,20",
                    help="comma-separated required-call counts to test")
    ap.add_argument("-n", type=int, default=8,
                    help="tasks per family per depth")
    ap.add_argument("--families", default="chain,ledger,lookup")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hop budget; default is 2x the deepest task + 8")
    ap.add_argument("--max-new", type=int, default=1536,
                    help="a write_file argument carries a whole "
                         "source file, so 512 truncates it")
    ap.add_argument("--rename", action="store_true",
                    help="opaque surface names, so only the description helps")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in families:
        if f not in GENERATORS:
            ap.error(f"unknown family {f!r}; pick from {list(GENERATORS)}")
    budget = args.max_steps or (2 * max(depths) + 8)

    model = load(args.checkpoint, device=args.device)
    tok = load_tokenizer_for(model)
    # A harness pointed at the wrong protocol does not fail, it reports 0/0
    # having asked the model nothing - which reads exactly like a broken model
    # and has cost a full afternoon before. Refuse instead.
    spec = getattr(model, "tokenizer_spec", None) or {}
    if spec.get("kind") != "bpe":
        sys.exit(f"{args.checkpoint} is not a ChatML/BPE checkpoint "
                 f"(tokenizer {spec.get('kind', 'byte')!r}). This eval speaks "
                 f"ChatML only; use scripts/eval_random.py for legacy ones.")

    print(f"{args.checkpoint}  depths={depths}  families={families}  "
          f"n={args.n}/cell  budget={budget} hops  rename={args.rename}")

    cells = {}
    reasons = {}
    # The project family's two intermediate stages, counted across all of it.
    stages = {"files": 0, "ran": 0, "total": 0}
    for depth in depths:
        for family in families:
            rng = random.Random(args.seed + depth * 1000 + hash(family) % 997)
            solved = hops_used = hops_min = 0
            if family == "project":
                stages["total"] += args.n
            for k in range(args.n):
                task = GENERATORS[family](depth, rng)
                tb, schemas, ws = bind(task, rng, rename=args.rename)
                run = run_task(model, tok, task, tb, schemas,
                               device=args.device, max_steps=budget,
                               max_new=args.max_new, seed=args.seed)
                if family == "project":
                    info = score_project(run, task, ws)
                    ok = info["out_ok"]
                    why = classify_project(run, info)
                    stages["files"] += info["files_ok"]
                    stages["ran"] += info["ran"]
                    detail = (f"files={info['n_files']}/{len(task.wants)} "
                              f"ran={info['ran']} out={info['stdout'][:24]!r}"
                              + (f" err={info['stderr'][:60]!r}"
                                 if info["stderr"] and not info["ran"] else ""))
                else:
                    ok = scored(run, task)
                    why = classify(run, task, ok)
                    detail = (f"want={task.answer} "
                              f"got={(run.answer or '')[:40]!r}")
                solved += ok
                hops_used += len(run.calls)
                hops_min += task.min_hops
                reasons[why] = reasons.get(why, 0) + 1
                if args.verbose:
                    print(f"  [{family} d{depth} #{k+1}] {why:13s} "
                          f"calls={len(run.calls):2d}/{task.min_hops:2d} "
                          + detail)
            cells[(depth, family)] = (solved, args.n, hops_used, hops_min)

    print("\nsolved, by required depth (and calls used / calls needed):")
    head = "  depth " + "".join(f"{f:>18s}" for f in families)
    print(head)
    for depth in depths:
        row = f"  {depth:5d} "
        for family in families:
            s, n, hu, hm = cells[(depth, family)]
            row += f"{s:>6d}/{n:<3d}{hu:>4d}/{hm:<4d}"
        print(row)

    total = sum(c[0] for c in cells.values())
    n_all = sum(c[1] for c in cells.values())
    print(f"\noverall {total}/{n_all} ({100.0*total/max(1,n_all):.0f}%)")
    if stages["total"]:
        t = stages["total"]
        print(f"project stages (of {t}): "
              f"all files written {stages['files']}/{t}, "
              f"runs clean {stages['ran']}/{t}, "
              f"correct output {sum(c[0] for (d, f), c in cells.items() if f == 'project')}/{t}")
    print("first failure mode:")
    for why in ORDER:
        if reasons.get(why):
            print(f"  {why:12s} {reasons[why]:3d}")

    if not args.no_notify:
        import os
        from agent.notify import notify
        # Last message of the process, so blocking=True: a daemon thread does
        # not survive interpreter shutdown.
        notify(f":bar_chart: **agentic** "
               f"`{os.path.basename(os.path.dirname(args.checkpoint))}` "
               f"{total}/{n_all} ({100.0*total/max(1,n_all):.0f}%) "
               f"depths {args.depths}" + (" renamed" if args.rename else ""),
               tag="eval", blocking=True)


if __name__ == "__main__":
    main()
