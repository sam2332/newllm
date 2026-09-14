"""Generate a library of story outlines and chapters with the Ollama teacher.

Long-project training traces (``agent/project_traces.py``) are *composed*
from this library rather than generated one by one, which is what keeps the
teacher cost bounded: a few hundred stories with several chapters each can be
recombined into tens of thousands of 52-turn arcs.

Two passes. Outlines first (title, premise, characters, per-chapter title and
summary) so every chapter prompt carries the whole plan and chapters can be
written in parallel; then the chapters themselves.

    .venv/bin/python scripts/gen_chapters_ollama.py --stories 150 --out data/chapters.json
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

GENRES = ["space opera", "cozy mystery", "epic fantasy", "hard science fiction",
          "gothic horror", "heist thriller", "coming-of-age", "post-apocalyptic survival",
          "historical romance", "noir detective", "comic fantasy", "military science fiction",
          "fairy tale retelling", "cyberpunk", "western", "sea adventure",
          "workplace comedy", "supernatural mystery", "sports drama", "political intrigue"]

OUTLINE_PROMPT = """Invent {n} original {genre} stories. Return ONLY a JSON array.
Return {{"stories": [...]}}. Each story: {{"title": str, "premise": 2 sentences, "protagonist": {{"name": str, "trait": str}},
"characters": [3-5 objects with "name" and "role"], "setting": one line,
"chapters": [exactly {chapters} objects with "title" and "summary" (2 sentences, concrete events)]}}.
Vary tone, era and structure. JSON only, no markdown."""

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {"stories": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "premise": {"type": "string"},
                "protagonist": {"type": "object",
                                "properties": {"name": {"type": "string"},
                                               "trait": {"type": "string"}},
                                "required": ["name", "trait"]},
                "characters": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"},
                                   "role": {"type": "string"}},
                    "required": ["name", "role"]}},
                "setting": {"type": "string"},
                "chapters": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"},
                                   "summary": {"type": "string"}},
                    "required": ["title", "summary"]}},
            },
            "required": ["title", "premise", "protagonist", "characters",
                         "setting", "chapters"],
        }}},
    "required": ["stories"],
}

CHAPTER_PROMPT = """You are writing chapter {i} of {n}, titled "{ctitle}", of the {genre} story "{title}".
Premise: {premise}
Protagonist: {protagonist}. Characters: {characters}. Setting: {setting}.
Full outline: {outline}
This chapter must cover: {summary}
Write {lo}-{hi} characters of polished prose. Plain paragraphs only: no heading, no chapter number, no notes, no markdown. Keep names consistent with the outline."""


def gen_outlines(model, genre, n, chapters, endpoint):
    # Two stories per request: five overflowed the token budget and the
    # truncated JSON was thrown away.
    text = ollama_chat(OUTLINE_PROMPT.format(n=n, genre=genre, chapters=chapters),
                       model, endpoint, temperature=1.0, timeout=600,
                       num_predict=4096, fmt=OUTLINE_SCHEMA)
    out = []
    for s in (json.loads(text).get("stories") or []):
        if not isinstance(s, dict) or not isinstance(s.get("chapters"), list):
            continue
        chs = [c for c in s["chapters"] if isinstance(c, dict)
               and c.get("title") and c.get("summary")]
        if len(chs) < 3 or not s.get("title") or not s.get("premise"):
            continue
        s["chapters"] = chs[:chapters]
        s["genre"] = genre
        out.append(s)
    return out


def gen_chapter(model, story, i, lo, hi, endpoint):
    ch = story["chapters"][i]
    outline = "; ".join(f"{k+1}. {c['title']}: {c['summary']}"
                        for k, c in enumerate(story["chapters"]))
    prot = story.get("protagonist") or {}
    prompt = CHAPTER_PROMPT.format(
        i=i + 1, n=len(story["chapters"]), ctitle=ch["title"], genre=story["genre"],
        title=story["title"], premise=story["premise"],
        protagonist=f"{prot.get('name', '?')} ({prot.get('trait', '')})",
        characters=", ".join(f"{c.get('name')} ({c.get('role')})"
                             for c in story.get("characters", []) if isinstance(c, dict)),
        setting=story.get("setting", ""), outline=outline, summary=ch["summary"],
        lo=lo, hi=hi)
    text = ollama_chat(prompt, model, endpoint, temperature=0.9, timeout=600).strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--stories", type=int, default=150)
    ap.add_argument("--chapters", type=int, default=6)
    ap.add_argument("--min-chars", type=int, default=1200)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/chapters.json")
    ap.add_argument("--endpoint", action="append", default=None,
                    help="explicit Ollama URL, repeatable. Overrides "
                         "--endpoints. Use this to choose WHICH GPU: the "
                         "containers both have NVIDIA_VISIBLE_DEVICES=all, so "
                         "their names say nothing about where a model lands - "
                         "Ollama picks the card with the most free VRAM.")
    ap.add_argument("--endpoints", type=int, default=len(ENDPOINTS),
                    help="how many Ollama endpoints to use; 1 keeps the second "
                         "GPU idle")
    ap.add_argument("--save-every", type=int, default=10,
                    help="persist after this many chapters")
    args = ap.parse_args()
    endpoints = args.endpoint or ENDPOINTS[:max(1, args.endpoints)]
    partial_path = args.out.replace(".json", ".partial.json")

    rng = random.Random(0)
    existing = json.load(open(args.out)) if os.path.exists(args.out) else []
    t0 = time.time()

    # Resume: outlines and any chapters already written live in the partial
    # file, so a restart after a power cut re-does only unfinished chapters.
    resumed = json.load(open(partial_path)) if os.path.exists(partial_path) else []
    if resumed:
        have = sum(1 for s in resumed for c in s["chapters"] if c.get("text"))
        total = sum(len(s["chapters"]) for s in resumed)
        print(f"resuming {partial_path}: {len(resumed)} outlines, "
              f"{have}/{total} chapters already written")

    # Pass 1: outlines, ~5 stories per request, genres round-robin.
    per_req = 2
    stories = resumed
    if not stories:
        n_req = max(1, args.stories // per_req)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(gen_outlines, args.model, GENRES[i % len(GENRES)], per_req,
                              args.chapters, endpoints[i % len(endpoints)])
                    for i in range(n_req)]
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    got = f.result()
                except Exception as exc:                        # noqa: BLE001
                    print(f"  outline request failed: {exc}"[:90])
                    continue
                stories.extend(got)
                save_atomic(stories, partial_path)
                print(f"  outlines [{k}/{n_req}] +{len(got)} (total {len(stories)})", flush=True)
        print(f"{len(stories)} outlines in {(time.time()-t0)/60:.1f} min", flush=True)

    # Pass 2: every chapter of every story, fully parallel.
    jobs = [(si, ci) for si, s in enumerate(stories) for ci in range(len(s["chapters"]))
            if not s["chapters"][ci].get("text")]
    print(f"{len(jobs)} chapters to write", flush=True)
    done_n, failed, since_save = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_chapter, args.model, stories[si], ci, args.min_chars,
                          args.max_chars, endpoints[(si + ci) % len(endpoints)]): (si, ci)
                for si, ci in jobs}
        for f in as_completed(futs):
            si, ci = futs[f]
            done_n += 1
            try:
                text = f.result()
            except Exception as exc:                            # noqa: BLE001
                failed += 1
                print(f"  chapter {si}.{ci} failed: {exc}"[:90])
                continue
            if len(text) < args.min_chars // 2:
                failed += 1
                continue
            stories[si]["chapters"][ci]["text"] = text
            since_save += 1
            if since_save >= args.save_every:
                save_atomic(stories, partial_path)
                since_save = 0
            if done_n % 25 == 0:
                print(f"  chapters [{done_n}/{len(jobs)}] failed {failed} "
                      f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    save_atomic(stories, partial_path)

    complete = [s for s in stories if all(c.get("text") for c in s["chapters"])]
    out = existing + complete
    save_atomic(out, args.out)
    chars = sum(len(c["text"]) for s in complete for c in s["chapters"])
    print(f"\n{len(complete)} complete stories, {sum(len(s['chapters']) for s in complete)} "
          f"chapters, {chars/1e6:.1f} MB prose in {(time.time()-t0)/60:.1f} min "
          f"-> {args.out} ({len(out)} total)")


if __name__ == "__main__":
    main()
