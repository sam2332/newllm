"""Generate a technical knowledge library with the Ollama teacher.

The corpus so far teaches the model *mechanics* - call a tool, hold a
persona, keep a 52-turn project straight - but almost nothing about the
subjects a coding assistant is actually asked about. A model that routes
tools perfectly and cannot say what a list comprehension does is not useful,
and an MoE has the capacity to hold this breadth if the data exists.

Two item kinds, both teacher-written, both cheap to recombine:

* **explain** - a question and a prose answer, no code file. Becomes a
  no-tool conversation (``agent/knowledge_traces.py``).
* **artifact** - a request, a complete source file, a one-line summary and
  follow-up questions answerable *from the file text*. Becomes a workspace
  arc: write the file, let it scroll out of context, read it back to answer.

Only the prose and the code come from the teacher. Every tool observation in
the composed trace comes from ``VirtualWorkspace``, so the "teacher never
fabricates a tool result" invariant holds here as everywhere else.

    .venv/bin/python scripts/gen_knowledge_ollama.py --items 4000 --out data/knowledge.json
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/lmeadows/llm")
from scripts.gen_data_ollama import ENDPOINTS, ollama_chat, save_atomic

# (domain, language, filename hint, topic list). The language decides the
# extension of the artifact file, which is what the model sees on disk.
DOMAINS = {
    "python": ("python", "py", [
        "list and dict comprehensions", "generators and yield", "decorators",
        "context managers and with", "dataclasses", "type hints and typing",
        "exceptions and custom exception classes", "iterators and __iter__",
        "argparse command line parsing", "pathlib file handling",
        "json serialization", "regular expressions with re",
        "collections: defaultdict, Counter, deque", "itertools recipes",
        "sorting with key functions", "f-strings and formatting",
        "virtual environments and pip", "modules, packages and __init__.py",
        "unittest and pytest basics", "mutable default argument pitfalls",
        "shallow vs deep copy", "async/await and asyncio",
        "threading vs multiprocessing and the GIL", "closures and scope",
        "properties and descriptors", "magic methods: __repr__, __eq__, __hash__",
        "slicing and negative indices", "enumerate and zip",
        "reading and writing CSV", "logging instead of print",
        "sets and set operations", "string methods and text cleaning",
        "functools: lru_cache, partial, wraps", "numpy array basics",
        "subprocess and running commands", "os.environ and configuration",
    ]),
    "bash": ("bash", "sh", [
        "pipes and redirection", "grep and regular expressions",
        "sed substitution", "awk field processing", "find with -exec",
        "for and while loops", "conditionals and test brackets",
        "variables, quoting and word splitting", "exit codes and set -e",
        "functions and arguments", "here documents", "xargs",
        "process substitution", "command substitution",
        "arrays in bash", "parameter expansion and defaults",
        "trap and cleanup", "background jobs and wait",
        "file permissions and chmod", "cron syntax",
        "tar and compression", "rsync", "ssh and scp",
        "environment variables and export", "PATH and which",
        "shebangs and script portability", "curl for HTTP requests",
        "jq for JSON on the command line", "sort, uniq and cut",
        "df, du and disk usage", "ps, top and killing processes",
        "symbolic and hard links",
    ]),
    "science": ("text", "md", [
        "Newton's laws of motion", "conservation of energy", "thermodynamics laws",
        "the periodic table and periodicity", "chemical bonding: ionic vs covalent",
        "acids, bases and pH", "the water cycle", "plate tectonics",
        "photosynthesis", "cellular respiration", "DNA replication",
        "protein synthesis: transcription and translation", "mitosis vs meiosis",
        "natural selection and evolution", "the scientific method",
        "electric circuits: Ohm's law", "magnetism and electromagnetic induction",
        "waves: frequency, wavelength, amplitude", "the electromagnetic spectrum",
        "optics: reflection and refraction", "gravity and orbits",
        "the life cycle of stars", "the Big Bang and cosmic expansion",
        "atomic structure and isotopes", "radioactivity and half-life",
        "special relativity basics", "quantum mechanics: wave-particle duality",
        "entropy and the arrow of time", "ecosystems and food webs",
        "the carbon cycle and climate change", "immune system basics",
        "the nervous system and neurons", "statistics: mean, median, variance",
        "probability basics", "significant figures and measurement error",
        "states of matter and phase changes", "solubility and solutions",
        "enzymes and catalysis", "genetics: dominant and recessive traits",
        "the human circulatory system",
    ]),
    "coding_principles": ("text", "md", [
        "DRY: don't repeat yourself", "KISS: keep it simple",
        "YAGNI: you aren't gonna need it", "single responsibility principle",
        "open/closed principle", "Liskov substitution principle",
        "interface segregation", "dependency inversion",
        "separation of concerns", "composition over inheritance",
        "pure functions and side effects", "immutability",
        "naming things well", "writing useful comments",
        "code review etiquette", "refactoring safely",
        "test-driven development", "unit vs integration vs end-to-end tests",
        "test doubles: mocks, stubs, fakes", "handling errors vs swallowing them",
        "fail fast and defensive programming", "logging levels and what to log",
        "premature optimization", "measuring before optimizing",
        "big-O complexity and when it matters", "caching and invalidation",
        "idempotency", "race conditions and locking",
        "version control: small commits and good messages",
        "branching strategies and pull requests", "semantic versioning",
        "continuous integration", "code coverage and its limits",
        "technical debt", "documentation that stays true",
        "designing APIs for callers", "backwards compatibility and deprecation",
        "security basics: input validation and least privilege",
        "secrets management", "the boy scout rule",
    ]),
    "algorithms": ("python", "py", [
        "binary search", "quicksort", "mergesort", "bubble sort and why not to use it",
        "hash tables", "linked lists", "stacks and queues", "binary search trees",
        "heaps and priority queues", "graphs: BFS", "graphs: DFS",
        "Dijkstra's shortest path", "topological sort", "dynamic programming: fibonacci",
        "dynamic programming: knapsack", "memoization", "two pointers technique",
        "sliding window technique", "recursion and base cases", "backtracking: n-queens",
        "string matching", "tries", "union-find", "sorting stability",
        "time vs space tradeoffs",
    ]),
    "debugging": ("text", "md", [
        "reading a stack trace", "bisecting to find a regression",
        "print debugging vs a debugger", "pdb basics",
        "reproducing a bug reliably", "rubber duck debugging",
        "off-by-one errors", "null and None errors",
        "encoding and unicode errors", "floating point surprises",
        "timezone and datetime bugs", "memory leaks",
        "deadlocks", "flaky tests", "heisenbugs and timing",
        "reading logs effectively", "using git blame", "minimal reproduction",
    ]),
}

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "followups": {"type": "array", "items": {
                "type": "object",
                "properties": {"question": {"type": "string"},
                               "answer": {"type": "string"}},
                "required": ["question", "answer"]}},
        },
        "required": ["question", "answer", "followups"],
    }}},
    "required": ["items"],
}

ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "request": {"type": "string"},
            "filename": {"type": "string"},
            "code": {"type": "string"},
            "summary": {"type": "string"},
            "followups": {"type": "array", "items": {
                "type": "object",
                "properties": {"question": {"type": "string"},
                               "answer": {"type": "string"}},
                "required": ["question", "answer"]}},
        },
        "required": ["request", "filename", "code", "summary", "followups"],
    }}},
    "required": ["items"],
}

EXPLAIN_PROMPT = """Write {n} distinct question-and-answer items about **{topic}** ({domain}).

Each item:
- "question": how a real user would ask, in their own words. Vary the phrasing and the level: beginner, intermediate, "why does X happen", "what's the difference between", "when should I".
- "answer": a correct, self-contained explanation of 3-8 sentences. Be concrete and use a short inline code example where it helps. Do not use markdown headings or bullet lists; plain paragraphs.
- "followups": 2-3 follow-up question/answer pairs that a user would naturally ask next, each answer 1-4 sentences.

Be accurate. If something is a common misconception, say so plainly. JSON only."""

ARTIFACT_PROMPT = """Write {n} distinct items in which a user asks for a **{lang}** file about **{topic}** and it is written for them.

Each item:
- "request": what the user asks for, phrased naturally, e.g. "Write me a script that ...". Vary difficulty.
- "filename": a short lowercase filename ending in .{ext}, no directories.
- "code": the complete file contents, {lo}-{hi} characters, correct and runnable, with a docstring or header comment and meaningful names. No markdown fences.
- "summary": one sentence saying what the file does.
- "followups": 3-5 question/answer pairs about THIS file specifically - what a named function does, why a choice was made, how to extend or run it. Each answer 1-4 sentences and answerable from the code itself.

JSON only, no markdown."""


def gen_explain(model, domain, topic, n, endpoint):
    text = ollama_chat(EXPLAIN_PROMPT.format(n=n, topic=topic, domain=domain),
                       model, endpoint, temperature=0.9, timeout=900,
                       num_predict=max(4096, 900 * n), fmt=SCHEMA)
    out = []
    for it in (json.loads(text).get("items") or []):
        if not isinstance(it, dict):
            continue
        q, a = (it.get("question") or "").strip(), (it.get("answer") or "").strip()
        if len(q) < 8 or len(a) < 60:
            continue
        fu = [f for f in (it.get("followups") or [])
              if isinstance(f, dict) and len((f.get("question") or "").strip()) > 6
              and len((f.get("answer") or "").strip()) > 15]
        out.append({"kind": "explain", "domain": domain, "topic": topic,
                    "question": q, "answer": a,
                    "followups": [{"question": f["question"].strip(),
                                   "answer": f["answer"].strip()} for f in fu]})
    return out


def gen_artifact(model, domain, lang, ext, topic, n, lo, hi, endpoint):
    text = ollama_chat(ARTIFACT_PROMPT.format(n=n, lang=lang, ext=ext, topic=topic,
                                              lo=lo, hi=hi),
                       model, endpoint, temperature=0.9, timeout=1200,
                       num_predict=max(6144, 1800 * n), fmt=ARTIFACT_SCHEMA)
    out = []
    for it in (json.loads(text).get("items") or []):
        if not isinstance(it, dict):
            continue
        code = (it.get("code") or "").strip()
        if code.startswith("```"):
            code = code.strip("`").strip()
        name = (it.get("filename") or "").strip().replace("/", "_").lstrip(".")
        if not name or "." not in name:
            continue
        if len(code) < 120 or len((it.get("request") or "").strip()) < 10:
            continue
        fu = [f for f in (it.get("followups") or [])
              if isinstance(f, dict) and len((f.get("question") or "").strip()) > 6
              and len((f.get("answer") or "").strip()) > 15]
        if not fu:
            continue
        out.append({"kind": "artifact", "domain": domain, "topic": topic,
                    "language": lang, "request": it["request"].strip(),
                    "filename": name, "code": code,
                    "summary": (it.get("summary") or "").strip(),
                    "followups": [{"question": f["question"].strip(),
                                   "answer": f["answer"].strip()} for f in fu]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--items", type=int, default=4000,
                    help="approximate number of items to generate")
    ap.add_argument("--per-request", type=int, default=4)
    ap.add_argument("--artifact-fraction", type=float, default=0.4)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domain", action="append", default=None,
                    help="restrict to these domains, repeatable")
    ap.add_argument("--out", default="data/knowledge.json")
    ap.add_argument("--endpoint", action="append", default=None,
                    help="explicit Ollama URL, repeatable. The container names "
                         "say nothing about which GPU serves a model.")
    ap.add_argument("--endpoints", type=int, default=len(ENDPOINTS))
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="job ordering; two instances with different seeds on "
                         "different endpoints keep both GPUs busy")
    args = ap.parse_args()

    endpoints = args.endpoint or ENDPOINTS[:max(1, args.endpoints)]
    partial = args.out.replace(".json", ".partial.json")
    rng = random.Random(args.seed)
    t0 = time.time()

    # Resume: a power cut costs at most --save-every requests.
    items = json.load(open(partial)) if os.path.exists(partial) else []
    if items:
        print(f"resuming {partial}: {len(items):,} items already written")

    names = args.domain or list(DOMAINS)
    jobs = []
    for dom in names:
        lang, ext, topics = DOMAINS[dom]
        for topic in topics:
            for kind in ("explain", "artifact"):
                if kind == "artifact" and lang == "text":
                    continue   # prose domains get no source file
                jobs.append((dom, lang, ext, topic, kind))
    # Weight the mix, then fill the item budget. There are only a few hundred
    # (domain, topic, kind) pairs, so a large budget cycles over them again -
    # the teacher samples at temperature 0.9, so a repeat is fresh questions
    # on the same topic, not the same items twice.
    rng.shuffle(jobs)
    want = max(1, (args.items - len(items)) // args.per_request)
    arts = [j for j in jobs if j[4] == "artifact"]
    exps = [j for j in jobs if j[4] == "explain"]
    n_art = int(want * args.artifact_fraction)
    picked = []
    for pool, n in ((arts, n_art), (exps, want - n_art)):
        if not pool:
            continue
        while len(picked) < n + (0 if pool is arts else n_art):
            rng.shuffle(pool)
            picked.extend(pool)
        picked = picked[:n + (0 if pool is arts else n_art)]
    jobs = picked[:want]
    rng.shuffle(jobs)
    print(f"{len(jobs)} requests x ~{args.per_request} items "
          f"({sum(1 for j in jobs if j[4] == "artifact")} artifact) across {len(names)} domains", flush=True)

    since, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i, (dom, lang, ext, topic, kind) in enumerate(jobs):
            ep = endpoints[i % len(endpoints)]
            if kind == "artifact":
                f = ex.submit(gen_artifact, args.model, dom, lang, ext, topic,
                              args.per_request, args.min_chars, args.max_chars, ep)
            else:
                f = ex.submit(gen_explain, args.model, dom, topic,
                              args.per_request, ep)
            futs[f] = (dom, topic, kind)
        for k, f in enumerate(as_completed(futs), 1):
            dom, topic, kind = futs[f]
            try:
                got = f.result()
            except Exception as exc:                        # noqa: BLE001
                failed += 1
                print(f"  {dom}/{kind} '{topic}' failed: {exc}"[:100], flush=True)
                continue
            items.extend(got)
            since += 1
            if since >= args.save_every:
                save_atomic(items, partial)
                since = 0
            if k % 20 == 0:
                print(f"  [{k}/{len(jobs)}] {len(items):,} items, {failed} failed, "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    save_atomic(items, partial)
    save_atomic(items, args.out)

    by_kind, by_dom = {}, {}
    for i in items:
        by_kind[i["kind"]] = by_kind.get(i["kind"], 0) + 1
        by_dom[i["domain"]] = by_dom.get(i["domain"], 0) + 1
    chars = sum(len(json.dumps(i)) for i in items)
    print(f"\n{len(items):,} items, {chars/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min "
          f"-> {args.out}")
    print("  by kind:  " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_kind.items())))
    print("  by domain: " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_dom.items())))


if __name__ == "__main__":
    main()
