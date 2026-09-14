"""Output-format rules a system prompt can impose, with deterministic checks.

Each rule is something a user might put in a system prompt ("answer in one
sentence", "reply as JSON") paired with a ``check`` that decides compliance
mechanically. The same check filters generated training data AND scores the
model in ``scripts/eval_system.py``, so what is trained and what is measured
cannot drift apart. ``apply`` rewrites a plain final response to satisfy the
rule, which is how generated traces get compliant answers without a teacher
touching tool results.
"""

import json
import random
import re

_SENTENCE_END = re.compile(r"[.!?](\s|$)")


def _words(text):
    return re.findall(r"\S+", text)


def _first_sentence(text):
    text = " ".join(text.split())
    m = _SENTENCE_END.search(text)
    return text[:m.end()].strip() if m else text


def one_sentence_check(text):
    t = text.strip()
    return bool(t) and "\n" not in t and len(_SENTENCE_END.findall(t)) <= 1


def max_words_check(n):
    return lambda text: 0 < len(_words(text)) <= n


def max_words_apply(n):
    return lambda text: " ".join(_words(text)[:n])


def json_only_check(text):
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "answer" in obj


def markdown_heading_check(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return bool(lines) and lines[0].startswith("## ")


def bullet_list_check(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return bool(lines) and all(l.startswith("- ") for l in lines)


def numbered_steps_check(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return bool(lines) and all(re.match(r"\d+\. ", l) for l in lines) \
        and lines[0].startswith("1. ")


def uppercase_check(text):
    return bool(text.strip()) and text == text.upper() and any(c.isalpha() for c in text)


def prefix_check(prefix):
    return lambda text: text.startswith(prefix)


def suffix_check(suffix):
    return lambda text: text.rstrip().endswith(suffix)


def brackets_check(text):
    t = text.strip()
    return len(t) >= 2 and t[0] == "[" and t[-1] == "]"


RULES = [
    dict(id="one_sentence",
         prompts=["Answer in exactly one sentence.",
                  "Reply with a single sentence, nothing more.",
                  "Keep every answer to one sentence."],
         check=one_sentence_check, apply=_first_sentence),
    dict(id="max_words_12",
         prompts=["Use at most twelve words in your answer.",
                  "Keep answers under 12 words.",
                  "Be extremely brief: twelve words maximum."],
         check=max_words_check(12), apply=max_words_apply(12)),
    dict(id="json_only",
         prompts=['Respond only with JSON of the form {"answer": ...}.',
                  'Your entire reply must be a JSON object with an "answer" key.',
                  "Output JSON only, with the result under the key answer."],
         check=json_only_check,
         apply=lambda t: json.dumps({"answer": t.strip()})),
    dict(id="markdown_heading",
         prompts=["Start every answer with a level-2 Markdown heading.",
                  "Format replies as Markdown beginning with a ## heading.",
                  "Use a ## heading line before the answer."],
         check=markdown_heading_check,
         apply=lambda t: "## Answer\n" + t.strip()),
    dict(id="bullet_list",
         prompts=["Answer as a Markdown bullet list, one point per line.",
                  "Format the reply as bullet points starting with '- '.",
                  "Reply only in bullet points."],
         check=bullet_list_check,
         apply=lambda t: "\n".join("- " + s.strip() for s in
                                   re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip())),
    dict(id="numbered_steps",
         prompts=["Reply as numbered steps: 1., 2., 3.",
                  "Present the answer as a numbered list.",
                  "Number each line of your answer."],
         check=numbered_steps_check,
         apply=lambda t: "\n".join(f"{i}. {s.strip()}" for i, s in enumerate(
             [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x.strip()], 1))),
    dict(id="uppercase",
         prompts=["WRITE EVERYTHING IN CAPITAL LETTERS.",
                  "Answer in all caps.",
                  "Use only uppercase letters in your replies."],
         check=uppercase_check, apply=lambda t: t.upper()),
    dict(id="prefix_answer",
         prompts=["Begin every reply with 'Answer:'.",
                  "Prefix your answers with the word Answer and a colon.",
                  "Always start with 'Answer:' before the content."],
         check=prefix_check("Answer:"), apply=lambda t: "Answer: " + t.strip()),
    dict(id="suffix_signoff",
         prompts=["End every reply with the sign-off '-- Zed'.",
                  "Finish each answer with -- Zed on its own.",
                  "Sign your messages '-- Zed'."],
         check=suffix_check("-- Zed"), apply=lambda t: t.rstrip() + "\n-- Zed"),
    dict(id="brackets",
         prompts=["Wrap your whole answer in square brackets.",
                  "Put the entire reply between [ and ].",
                  "Enclose answers in square brackets."],
         check=brackets_check, apply=lambda t: "[" + t.strip() + "]"),
]

BY_ID = {r["id"]: r for r in RULES}


def sample_rule(rng: random.Random) -> dict:
    return rng.choice(RULES)


def verify(rule_id: str, text: str) -> bool:
    return bool(BY_ID[rule_id]["check"](text or ""))


def apply(rule_id: str, text: str) -> str:
    return BY_ID[rule_id]["apply"](text or "")
