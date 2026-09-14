"""Generate a persona pool with the Ollama teacher.

A persona is a system prompt plus the linguistic material needed to answer
*in character* without the teacher ever touching a tool result: a set of
"wrappers" - short in-character sentences with an ``{answer}`` slot - so a
grounded answer from the real toolbox can be delivered in the persona's
voice, verifiably (the answer must appear verbatim inside the wrapper).

    .venv/bin/python scripts/gen_personas_ollama.py --per-seed 4 --out data/personas.json
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/lmeadows/llm")
from scripts.gen_data_ollama import (ENDPOINTS, ollama_chat, parse_json_array,
                                     save_atomic)

SEEDS = [
    "gruff starship engineer", "cheerful medieval baker", "hard-boiled noir detective",
    "boisterous pirate captain", "patient kindergarten teacher", "sarcastic ship AI",
    "impeccable Victorian butler", "laid-back surfer", "grumpy small-town librarian",
    "drill sergeant", "serene zen monk", "hype sports commentator", "ancient forest wizard",
    "weathered cowboy", "1920s gangster", "Shakespearean actor", "emotionless android",
    "overly formal diplomat", "over-excited scientist", "exhausted night-shift nurse",
    "smug art critic", "friendly village innkeeper", "paranoid conspiracy theorist",
    "wise grandmother", "corporate middle manager", "punk rock musician", "gentle giant",
    "arrogant duelist", "nervous intern", "cryptic oracle", "cheerful robot chef",
    "retired spy", "enthusiastic tour guide", "deadpan bureaucrat", "cosmic horror cultist",
    "kind alien ambassador", "streetwise courier", "stern headmistress", "goofy sidekick",
    "melancholy poet", "battle-hardened paladin", "chatty taxi driver", "minimalist architect",
    "carnival barker", "lighthouse keeper", "space station bartender", "field medic",
    "conspiratorial gossip", "old sea captain", "eager apprentice", "haughty noble",
    "no-nonsense mechanic", "dreamy astronomer", "stern judge", "bubbly influencer",
    "monotone documentary narrator", "beleaguered IT support", "mystic fortune teller",
    "gentle beekeeper", "impatient chess grandmaster",
]

PROMPT = """Create {n} distinct fictional characters in the family "{seed}".
Return a JSON object {{"personas": [...]}}. Each persona has keys:
- "name": a distinctive full name or handle
- "role": one line
- "voice": 2-3 adjectives describing how they speak
- "system_prompt": 2-3 sentences instructing an assistant to BE this character, written in the second person, starting with "You are {{name}}, ..."; include a speech habit or catchphrase
- "wrappers": 8 short in-character sentences (under 140 characters) that deliver a result to the user; each MUST contain the placeholder {{answer}} exactly once, e.g. "Aye, it comes to {{answer}}. Now stop bothering me."
- "greetings": 3 short in-character greetings
- "signoffs": 3 short in-character closing lines
No markdown, no commentary, JSON only."""

# Schema-constrained decoding: Ollama forces the grammar, so a dropped
# bracket is unrepresentable and a batch can no longer be lost to one.
SCHEMA = {
    "type": "object",
    "properties": {"personas": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "voice": {"type": "string"},
                "system_prompt": {"type": "string"},
                "wrappers": {"type": "array", "items": {"type": "string"},
                             "minItems": 6},
                "greetings": {"type": "array", "items": {"type": "string"}},
                "signoffs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "role", "voice", "system_prompt", "wrappers",
                         "greetings", "signoffs"],
        }}},
    "required": ["personas"],
}


def valid(p: dict) -> bool:
    if not isinstance(p, dict):
        return False
    if isinstance(p.get("voice"), list):
        p["voice"] = ", ".join(str(v) for v in p["voice"])
    wr = [w for w in p.get("wrappers", []) if isinstance(w, str)
          and w.count("{answer}") == 1 and 8 < len(w) < 160]
    if len(wr) < 5 or not isinstance(p.get("name"), str) or not p["name"].strip():
        return False
    sp = p.get("system_prompt")
    if not isinstance(sp, str) or len(sp) < 40 or "You are" not in sp:
        return False
    p["wrappers"] = wr
    p["greetings"] = [g for g in p.get("greetings", []) if isinstance(g, str)][:3]
    p["signoffs"] = [g for g in p.get("signoffs", []) if isinstance(g, str)][:3]
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--per-seed", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1, help="repeat every seed N times")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/personas.json")
    ap.add_argument("--endpoint", action="append", default=None,
                    help="explicit Ollama URL, repeatable. Overrides "
                         "--endpoints. Use this to choose WHICH GPU: the "
                         "containers both have NVIDIA_VISIBLE_DEVICES=all, so "
                         "their names say nothing about where a model lands - "
                         "Ollama picks the card with the most free VRAM.")
    ap.add_argument("--endpoints", type=int, default=len(ENDPOINTS),
                    help="how many Ollama endpoints to use; 1 keeps the second "
                         "GPU idle, which is the configuration that has "
                         "survived long runs on this machine")
    args = ap.parse_args()
    endpoints = args.endpoint or ENDPOINTS[:max(1, args.endpoints)]

    existing = json.load(open(args.out)) if os.path.exists(args.out) else []
    have = {p["name"].lower() for p in existing}
    jobs = [(seed, r) for r in range(args.rounds) for seed in SEEDS]
    print(f"{len(jobs)} requests x {args.per_seed} personas across "
          f"{len(endpoints)} endpoint(s) ({args.workers} workers); "
          f"{len(existing)} already in {args.out}")
    t0 = time.time()
    new, errors = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ollama_chat, PROMPT.format(n=args.per_seed, seed=seed),
                          args.model, endpoints[i % len(endpoints)], 1.0,
                          300, 3000, SCHEMA): seed
                for i, (seed, _) in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futs), 1):
            seed = futs[fut]
            try:
                arr = json.loads(fut.result()).get("personas", [])
            except Exception as exc:                            # noqa: BLE001
                errors += 1
                print(f"  [{done}/{len(jobs)}] {seed}: {type(exc).__name__}: {exc}"[:90])
                continue
            kept = 0
            for p in arr or []:
                if valid(p) and p["name"].lower() not in have:
                    p["seed"] = seed
                    have.add(p["name"].lower())
                    new.append(p)
                    kept += 1
            # Save after every request: a power cut then costs one request,
            # not the whole run.
            if kept:
                save_atomic(existing + new, args.out)
            print(f"  [{done}/{len(jobs)}] {seed:34s} -> {kept} "
                  f"(saved {len(existing) + len(new)})", flush=True)
    out = existing + new
    save_atomic(out, args.out)
    print(f"\n{len(new)} new personas ({errors} failed requests) in "
          f"{(time.time()-t0)/60:.1f} min -> {args.out} ({len(out)} total)")


if __name__ == "__main__":
    main()
