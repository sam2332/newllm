"""Build a story library from Hugging Face instead of from the teacher.

`agent/project_traces.py` composes 52-turn arcs by writing chapters into files
and reading them back. Its library is 149 teacher-written stories / 894
chapters, and that is the ceiling on its diversity: 400 arcs write each
chapter verbatim ~4.5 times, which is why the slice measures 81% repeated
sentences against 4.7% for the tool traces. Composition cannot manufacture
prose nobody wrote, and the teacher is the expensive part.

Human-written stories are free and plentiful. This converts them into the
same shape `data/chapters.json` uses, by splitting each story into
chapter-sized segments on paragraph boundaries.

What it deliberately does NOT do is invent metadata. The teacher library
carries a protagonist, named characters and a setting, and `project_traces`
asks questions grounded in them; inventing those from an arbitrary story
would be the one thing this repo does not do - assert something the source
never said. Stories imported here have no `characters`, and the generator
skips the question kinds that need them.

    .venv/bin/python scripts/import_hf_stories.py --limit 20000 \\
        --out data/hf_stories.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/home/lmeadows/llm")
from agent.notify import Progress  # noqa: E402

WP_TAG = re.compile(r"^\s*\[\s*(WP|EU|CW|TT|IP|RF|OT)\s*\]\s*", re.I)


def clean_prompt(text: str) -> str:
    text = WP_TAG.sub("", text or "").strip()
    # This corpus is tokenized: " n't", " ,", `` '' for quotes.
    text = text.replace(" n't", "n't").replace(" ,", ",").replace(" .", ".")
    text = text.replace(" !", "!").replace(" ?", "?").replace(" 's", "'s")
    text = text.replace("``", '"').replace("''", '"').replace(" ;", ";")
    return re.sub(r"\s+", " ", text).strip()


def clean_story(text: str) -> str:
    text = (text or "").replace("<newline>", "\n")
    text = text.replace(" n't", "n't").replace(" ,", ",").replace(" .", ".")
    text = text.replace(" !", "!").replace(" ?", "?").replace(" 's", "'s")
    text = text.replace("``", '"').replace("''", '"').replace(" ;", ";")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def to_chapters(story: str, lo: int, hi: int) -> list:
    """Split on paragraph boundaries into chapter-sized pieces."""
    paras = [p.strip() for p in story.split("\n") if p.strip()]
    chapters, buf, size = [], [], 0
    for para in paras:
        buf.append(para)
        size += len(para)
        if size >= lo:
            chapters.append("\n\n".join(buf))
            buf, size = [], 0
    if buf and size >= lo // 2:
        chapters.append("\n\n".join(buf))
    elif buf and chapters:
        chapters[-1] += "\n\n" + "\n\n".join(buf)
    return [c for c in chapters if lo // 2 <= len(c) <= hi * 2]


def first_sentence(text: str, limit: int = 220) -> str:
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    s = m[0] if m else text
    return (s[:limit] + "...") if len(s) > limit else s


def title_from(prompt: str, limit: int = 70) -> str:
    t = clean_prompt(prompt)
    t = re.split(r"(?<=[.!?])\s+", t)[0]
    t = t.rstrip(".!?")
    return (t[:limit].rstrip() + "...") if len(t) > limit else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="euclaise/writingprompts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--min-chapters", type=int, default=3)
    ap.add_argument("--max-chapters", type=int, default=12)
    ap.add_argument("--chapter-lo", type=int, default=900)
    ap.add_argument("--chapter-hi", type=int, default=2600)
    ap.add_argument("--out", default="data/hf_stories.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    out, seen, skipped = [], set(), 0
    progress = Progress(f"import {args.dataset}", args.limit, tag="import")
    for row in ds:
        if len(out) >= args.limit:
            break
        story = clean_story(row.get("story") or row.get("text") or "")
        prompt = row.get("prompt") or ""
        if len(story) < args.chapter_lo * args.min_chapters:
            skipped += 1
            continue
        chapters = to_chapters(story, args.chapter_lo, args.chapter_hi)
        if len(chapters) < args.min_chapters:
            skipped += 1
            continue
        chapters = chapters[:args.max_chapters]
        title = title_from(prompt) or first_sentence(story, 60)
        if title.lower() in seen:
            skipped += 1
            continue
        seen.add(title.lower())
        out.append({
            "title": title,
            "genre": "story",
            "premise": first_sentence(clean_prompt(prompt) or story),
            "protagonist": {},          # not invented: the source never said
            "characters": [],
            "setting": "",
            "chapters": [{"title": f"Chapter {i+1}",
                          "summary": first_sentence(c),
                          "text": c} for i, c in enumerate(chapters)],
        })
        progress.update(len(out), f"{skipped:,} skipped")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    n_ch = sum(len(s["chapters"]) for s in out)
    chars = sum(len(c["text"]) for s in out for c in s["chapters"])
    print(f"{len(out):,} stories, {n_ch:,} chapters, {chars/1e6:.1f} MB "
          f"({skipped:,} skipped) -> {args.out}")
    progress.finish(f"{len(out):,} stories, {chars/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
