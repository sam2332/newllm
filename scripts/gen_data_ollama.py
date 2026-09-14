"""Generate diverse agent training data with a local Ollama teacher model.

Why this exists
---------------
The hand-written generator produces 34 distinct "thought" strings, 12 web facts
and 7 memory keys. A model trained on it memorizes those 34 sentences instead of
learning to use tools, which is why validation loss goes to ~0 while held-out
accuracy stalls.

What this does NOT do
---------------------
It never lets the teacher compute a tool result. Every observation in every
emitted trace is produced by the real ``Toolbox`` and validated. The teacher is
used only for *linguistic* diversity - paraphrases, reasoning sentences, and
factual key/value pairs - never as an oracle. This keeps the data correct by
construction while making it far more varied.

Both Ollama instances are driven concurrently (11434 + 11435).
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlrequest

sys.path.insert(0, "/home/lmeadows/llm")

ENDPOINTS = ["http://localhost:11434", "http://localhost:11435"]
_counter = threading.local()


def ollama_chat(prompt: str, model: str, endpoint: str,
                temperature: float = 1.0, timeout: int = 300,
                num_predict: int = 2048, fmt=None) -> str:
    """``fmt`` is Ollama's ``format``: "json" or a JSON schema.

    Worth using for anything that must parse. Unconstrained, the teacher
    silently drops a closing bracket often enough to lose whole batches -
    a run produced a persona array whose last field read
    ``"signoffs":[...,"Just don't."}]`` and was discarded entire.
    """
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if fmt is not None:
        payload["format"] = fmt
    body = json.dumps(payload).encode()
    req = urlrequest.Request(f"{endpoint}/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


def save_atomic(obj, path: str):
    """Write JSON via a temp file + rename.

    Generators that only saved at the end lost everything to a power cut.
    They now save after every completed unit, which makes a partial write the
    likely failure instead - rename is atomic, so the file on disk is always
    a complete earlier version.
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def parse_json_array(text: str):
    """Strict first: the outermost [...] as JSON. Compact, nested output from
    the teacher (lists inside objects) is valid JSON that the line-by-line
    fallback in extract_json_array cannot reassemble."""
    try:
        start, end = text.index("["), text.rindex("]") + 1
        arr = json.loads(text[start:end])
        if isinstance(arr, list):
            return arr
    except (ValueError, json.JSONDecodeError):
        pass
    return extract_json_array(text)


def extract_json_array(text: str):
    """Pull the first JSON array out of a model response.

    Teacher output at high temperature is not reliably clean: it may add a
    preamble, wrap the array in a code fence, or trail commentary. Falls back
    to scraping quoted strings / key-value objects line by line.
    """
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    parsed = _scan_array(text)
    if parsed:
        return parsed
    # Fallback 1: objects of the form {"key": ..., "value": ...}
    objs = re.findall(r'\{\s*"key"\s*:\s*"([^"]{2,60})"\s*,\s*"value"\s*:\s*"([^"]{1,60})"\s*\}', text)
    if objs:
        return [{"key": k, "value": v} for k, v in objs]
    # Fallback 2: one quoted string per line
    lines = re.findall(r'^\s*"([^"]{4,200})"\s*,?\s*$', text, flags=re.M)
    return lines


def _scan_array(text: str):
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


# ------------------------------------------------------------------ prompts

THOUGHT_PROMPT = """You are generating training data for a small tool-using AI agent.

Write {n} DIFFERENT one-sentence internal reasoning statements that an agent \
would think in this exact situation:

{situation}

Rules:
- Each must be a single natural sentence, 6 to 20 words.
- Vary the vocabulary, sentence structure, and level of detail heavily.
- Do not number them. Do not use quotes inside the sentences.
- Write them as the agent's own first-person reasoning.

Return ONLY a JSON array of {n} strings."""

PARAPHRASE_PROMPT = """You are generating training data for a tool-using AI agent.

Write {n} DIFFERENT natural ways a user might phrase this request:

{situation}

Use the placeholder {placeholder} exactly where the variable part belongs.
Vary politeness, length, directness, and phrasing (questions, commands, \
indirect requests). Include casual and formal styles.

Return ONLY a JSON array of {n} strings, each containing {placeholder}."""

FACTS_PROMPT = """Generate {n} well-known, factually TRUE reference facts for a \
lookup database, in the category: {category}.

Rules:
- Each key is a short lowercase noun phrase someone would search for.
- Each value is a SHORT factual answer (under 6 words).
- Only include facts that are stable and uncontroversial.
- No duplicates.

Return ONLY a JSON array of objects: [{{"key": "...", "value": "..."}}]"""

CATEGORIES = [
    "world capital cities", "chemical elements and their symbols",
    "planets and moons of the solar system", "units of measurement",
    "famous scientists and their main discovery",
    "programming languages and their creator",
    "countries and their currency", "human anatomy basics",
    "geography: longest rivers and highest mountains",
    "classic literature and their authors",
    "animals: speeds, lifespans and classifications",
    "physics and mathematics constants",
    "musical instruments and their family",
    "sports and the number of players per side",
    "food origins and main ingredients",
    "computer science: data structures and their complexity",
]

SITUATIONS = {
    "calc_call": "The user asked an arithmetic question, and the agent is about "
                 "to call a calculator tool to compute it.",
    "calc_done": "The calculator tool has returned a number, and the agent is "
                 "about to report that number as the final answer.",
    "memory_call": "The user asked for a stored internal value, and the agent "
                   "is about to search its memory store.",
    "memory_done": "The memory store returned the value, and the agent is about "
                   "to report it.",
    "web_call": "The user asked a factual question the agent does not know, so "
                "it is about to search the web.",
    "web_done": "The web search returned a fact, and the agent is about to "
                "report it.",
    "time_call": "The user asked what time or date it is, and the agent is "
                 "about to call a clock tool.",
    "time_done": "The clock tool returned a timestamp and the agent will "
                 "report it.",
    "chain_first": "This is a multi-step task. The agent just got the first "
                   "intermediate result and still needs another tool call.",
    "chain_last": "This is a multi-step task and the agent has just computed "
                  "the final value of the chain.",
    "coref": "The user said 'multiply that by 3', referring to the result of "
             "the previous turn. The agent is resolving what 'that' means.",
    "no_tool": "The user asked something the agent can answer from the "
               "conversation already, so no tool call is needed.",
    "repo_call": "The user asked about the agent's own source code, and the "
                 "agent is about to call a tool that reads its repository.",
    "repo_done": "A code-inspection tool returned source code, and the agent "
                 "is about to report what it found.",
    "sandbox_call": "The user asked something best solved by writing and "
                    "running code, so the agent is about to execute code in "
                    "an isolated sandbox.",
    "sandbox_done": "The sandbox finished running the code and returned "
                    "output, which the agent will now report.",
}

PARAPHRASE_JOBS = {
    "math": ("The user wants the agent to compute an arithmetic expression.",
             "{EXPR}"),
    "memory": ("The user wants to look up a stored internal value by name.",
               "{KEY}"),
    "web": ("The user wants a factual answer that requires a web search.",
            "{TOPIC}"),
    "time": ("The user wants the current date and time.", "{NOW}"),
}


def gen_task(kind, prompt, model, endpoint, temperature):
    try:
        raw = ollama_chat(prompt, model, endpoint, temperature=temperature)
        return kind, extract_json_array(raw)
    except Exception as exc:                      # noqa: BLE001
        return kind, {"error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b-q8_0")
    ap.add_argument("--out", default="data/ollama_pool.json")
    ap.add_argument("--thoughts-per-situation", type=int, default=60)
    ap.add_argument("--paraphrases", type=int, default=40)
    ap.add_argument("--facts-per-category", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for name, situation in SITUATIONS.items():
        jobs.append(("thought:" + name,
                     THOUGHT_PROMPT.format(n=args.thoughts_per_situation,
                                           situation=situation)))
    for name, (situation, placeholder) in PARAPHRASE_JOBS.items():
        jobs.append(("paraphrase:" + name,
                     PARAPHRASE_PROMPT.format(n=args.paraphrases,
                                              situation=situation,
                                              placeholder=placeholder)))
    for category in CATEGORIES:
        jobs.append(("facts:" + category,
                     FACTS_PROMPT.format(n=args.facts_per_category,
                                         category=category)))

    print(f"{len(jobs)} generation jobs across {len(ENDPOINTS)} Ollama instances "
          f"({args.workers} workers)")
    pool = {"thoughts": {}, "paraphrases": {}, "facts": {}}
    errors = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for i, (kind, prompt) in enumerate(jobs):
            endpoint = ENDPOINTS[i % len(ENDPOINTS)]
            temp = 1.1 if kind.startswith(("thought", "paraphrase")) else 0.4
            futures[ex.submit(gen_task, kind, prompt, args.model, endpoint,
                              temp)] = kind
        for fut in as_completed(futures):
            kind, result = fut.result()
            done += 1
            if isinstance(result, dict):
                errors.append((kind, result["error"]))
                print(f"  [{done}/{len(jobs)}] {kind}: {result['error'][:60]}")
                continue
            group, _, name = kind.partition(":")
            if group == "thought":
                vals = [s for s in result if isinstance(s, str) and 4 < len(s) < 200]
                pool["thoughts"][name] = sorted(set(vals))
            elif group == "paraphrase":
                vals = [s for s in result if isinstance(s, str) and len(s) < 200]
                pool["paraphrases"][name] = sorted(set(vals))
            else:
                for item in result:
                    if isinstance(item, dict) and item.get("key") and item.get("value"):
                        k = str(item["key"]).strip().lower()
                        v = str(item["value"]).strip()
                        if 2 < len(k) < 60 and 0 < len(v) < 60:
                            pool["facts"][k] = v
            n = (len(pool["thoughts"].get(name, [])) or
                 len(pool["paraphrases"].get(name, [])) or len(pool["facts"]))
            print(f"  [{done}/{len(jobs)}] {kind[:44]:44s} -> {n}")

    dt = time.time() - t0
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(pool, open(args.out, "w"), indent=1)
    n_th = sum(len(v) for v in pool["thoughts"].values())
    n_pp = sum(len(v) for v in pool["paraphrases"].values())
    print(f"\ngenerated in {dt/60:.1f} min")
    print(f"  thoughts    {n_th} across {len(pool['thoughts'])} situations")
    print(f"  paraphrases {n_pp} across {len(pool['paraphrases'])} intents")
    print(f"  facts       {len(pool['facts'])}")
    if errors:
        print(f"  errors      {len(errors)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
