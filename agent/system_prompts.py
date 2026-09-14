"""Attach a system prompt to a trace and make the answers obey it.

Two families, both applied as a post-processing pass on the legacy tagged
trace AFTER ``randomize_trace`` has prepended the schema block, so the
free-text block lands second - the order the model is served
(``agent/ollama_context.py``, ``agent/chatml.py``).

* **Persona**: the persona's ``system_prompt`` goes in the block, and every
  final ``response`` is delivered through one of the persona's wrappers, e.g.
  ``"Aye, it comes to 20. Now stop bothering me."`` The grounded answer is
  kept verbatim inside the wrapper, so nothing the toolbox did not return is
  asserted.
* **Format rule**: a phrasing of the rule goes in the block, and every final
  ``response`` is rewritten with the rule's ``apply``; the rule's ``check``
  then verifies it, and a trace that fails is dropped rather than trained on.
"""

import json
import random
import re

from agent.format_rules import RULES, verify

_ASSISTANT_JSON = re.compile(r"(<assistant>)(\{.*?\})(</assistant>)", re.S)
_SCHEMA_BLOCK = re.compile(r'^<system>\{"tools":.*?</system>\n', re.S)


def _with_system_block(text: str, system_text: str) -> str:
    m = _SCHEMA_BLOCK.match(text)
    block = f"<system>{system_text}</system>\n"
    if m:
        return text[:m.end()] + block + text[m.end():]
    return block + text


def _rewrite_responses(text: str, fn) -> str:
    def fix(m):
        try:
            obj = json.loads(m.group(2))
        except json.JSONDecodeError:
            return m.group(0)
        if isinstance(obj.get("response"), str):
            obj["response"] = fn(obj["response"])
        return m.group(1) + json.dumps(obj, separators=(",", ":")) + m.group(3)
    return _ASSISTANT_JSON.sub(fix, text)


def apply_persona(text: str, persona: dict, rng: random.Random) -> str:
    """Persona system prompt + in-character delivery of every final answer."""
    wrappers = [w for w in persona.get("wrappers", []) if "{answer}" in w]
    greetings = persona.get("greetings") or []
    signoffs = persona.get("signoffs") or []
    if not wrappers:
        return text

    def style(answer: str) -> str:
        # Long answers (a recap, a chapter note) get a greeting/sign-off
        # instead of being jammed into a one-line wrapper.
        if len(answer) > 160 or "\n" in answer:
            head = rng.choice(greetings) + "\n\n" if greetings and rng.random() < 0.6 else ""
            tail = "\n\n" + rng.choice(signoffs) if signoffs and rng.random() < 0.6 else ""
            return head + answer + tail
        return rng.choice(wrappers).replace("{answer}", answer.rstrip("."))

    out = _rewrite_responses(text, style)
    return _with_system_block(out, persona["system_prompt"].strip())


def apply_format_rule(text: str, rule: dict, rng: random.Random):
    """Rule phrasing as the system prompt + compliant answers, or None if the
    rewritten answers do not pass the rule's own check."""
    out = _rewrite_responses(text, rule["apply"])
    for m in _ASSISTANT_JSON.finditer(out):
        try:
            obj = json.loads(m.group(2))
        except json.JSONDecodeError:
            return None
        if isinstance(obj.get("response"), str) and not verify(rule["id"], obj["response"]):
            return None
    return _with_system_block(out, rng.choice(rule["prompts"]))


def decorate_traces(traces: list, rng: random.Random, personas: list = None,
                    persona_fraction: float = 0.0, format_fraction: float = 0.0) -> list:
    """Apply persona / format-rule system prompts to random subsets.

    Always returns one trace per input. When a format rule cannot be applied
    so that its own check passes, the trace is emitted UNDECORATED rather than
    dropped: the data is still good, it just makes no claim about a rule. The
    invariant that matters is never training a system prompt the answers
    disobey - not discarding the trace.
    """
    out = []
    for t in traces:
        r = rng.random()
        if personas and r < persona_fraction:
            out.append(apply_persona(t, rng.choice(personas), rng))
        elif r < persona_fraction + format_fraction:
            done = apply_format_rule(t, rng.choice(RULES), rng)
            out.append(done if done is not None else t)
        else:
            out.append(t)
    return out
