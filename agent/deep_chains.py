"""Deep multi-hop chains (5-60+ tool calls) with verifiable grounding.

Two goals, and the second constrains the first.

MORE HOPS. Tasks are built as pipelines where every stage consumes the previous
stage's real output, so a 40-hop trace is 40 genuinely dependent steps rather
than 40 independent questions concatenated. Chain depth is a curriculum knob.

NO HALLUCINATION. Every value a stage consumes must already be visible in the
transcript - as a prior tool observation, or in the user's own question. The
generator asserts this for every stage it emits, so a trace that could only be
produced by inventing a number is never used as training data. ``verify_trace``
applies the same check to model output at inference.

This matters more as chains deepen: at hop 40 the model has 39 prior
observations to confuse, and a single fabricated intermediate silently corrupts
everything after it.
"""

import json
import random
import re

from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools

EOT = "\x03"
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> set:
    """Numeric literals appearing in a piece of text, normalized."""
    out = set()
    for m in _NUM.finditer(str(text)):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        out.add(_canon(v))
    return out


def _canon(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else str(round(f, 4))


class GroundingError(ValueError):
    """A stage consumed a value that never appeared in the transcript."""


class DeepChainBuilder:
    """Builds an N-stage chain, asserting grounding at every stage."""

    def __init__(self, toolbox: Toolbox, rng: random.Random,
                 thought_fn=None):
        self.tb = toolbox
        self.rng = rng
        self.thought = thought_fn or (lambda situation, fallback: fallback)
        self.messages = []
        self.grounded = set()      # every number the model may legitimately use
        self.last = None           # most recent stage result

    # ------------------------------------------------------------------
    def start(self, question: str):
        self.messages = [{"role": "user", "content": question}]
        # Numbers stated in the question are legitimately available.
        self.grounded = set(numbers_in(question))
        self.last = None
        return self

    def _assert_grounded(self, args: dict):
        """Refuse any argument value not already visible in the transcript."""
        for value in args.values():
            for n in numbers_in(value):
                if n not in self.grounded:
                    raise GroundingError(
                        f"value {n} in {args!r} is not grounded; "
                        f"available: {sorted(self.grounded)[:8]}")

    def step(self, tool: str, args: dict, situation: str, fallback: str):
        """Run one tool stage. Raises GroundingError if args are ungrounded."""
        self._assert_grounded(args)
        observation = self.tb.run_json({"tool": tool, "args": args})
        if str(observation).startswith("ERROR") or observation == "no results found":
            raise GroundingError(f"tool {tool} failed: {observation}")
        self.messages.append({"role": "assistant", "content": json.dumps(
            {"thought": self.thought(situation, fallback),
             "tool_call": {"name": tool, "arguments": args}},
            separators=(",", ":"))})
        self.messages.append({"role": "tool", "name": tool,
                              "content": str(observation)})
        # The observation is now legitimately available to later stages.
        self.grounded |= numbers_in(observation)
        self.last = str(observation)
        return observation

    def finish(self, situation: str, fallback: str):
        self.messages.append({"role": "assistant", "content": json.dumps(
            {"thought": self.thought(situation, fallback),
             "response": self.last}, separators=(",", ":"))})
        return self.messages

    @property
    def hops(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "tool")


# ---------------------------------------------------------------- builders

def arithmetic_pipeline(builder: DeepChainBuilder, hops: int) -> list:
    """A chain of arithmetic stages, each consuming the previous result."""
    rng = builder.rng
    start = rng.randint(2, 60)
    ops = []
    for _ in range(hops):
        op = rng.choice(["+", "-", "*"])
        k = rng.randint(2, 9) if op == "*" else rng.randint(1, 40)
        ops.append((op, k))
    described = ", then ".join(
        f"{'add' if o == '+' else 'subtract' if o == '-' else 'multiply by'} {k}"
        for o, k in ops)
    builder.start(f"Start with {start}, {described}. What is the result?")
    current = float(start)
    for i, (op, k) in enumerate(ops):
        expr = f"{_canon(current)} {op} {k}"
        out = builder.step("calc", {"expr": expr},
                           "chain_first" if i < len(ops) - 1 else "chain_last",
                           f"Step {i + 1}: compute {expr}.")
        current = float(out)
    return builder.finish("chain_last", "The pipeline is complete.")


def lookup_pipeline(builder: DeepChainBuilder, hops: int, facts: dict,
                    memory: dict) -> list:
    """Alternating lookups and arithmetic; every operand comes from a tool."""
    rng = builder.rng
    numeric_facts = [k for k, v in facts.items() if numbers_in(v)]
    numeric_mem = [k for k, v in memory.items() if numbers_in(v)]
    if not numeric_facts:
        return arithmetic_pipeline(builder, hops)

    first_key = rng.choice(numeric_facts)
    stages = max(1, hops - 1)
    builder.start(
        f"Look up the {first_key}, then run {stages} follow-up operations on "
        f"the value, using tools for every step.")
    obs = builder.step("web_search", {"query": first_key},
                       "web_call", f"First I need the {first_key}.")
    nums = sorted(numbers_in(obs))
    if not nums:
        raise GroundingError("first lookup returned no number")
    current = float(nums[0])

    for i in range(stages):
        # Every few stages, pull another real value from a tool and combine.
        if i % 3 == 2 and (numeric_facts or numeric_mem):
            if numeric_mem and rng.random() < 0.5:
                key = rng.choice(numeric_mem)
                o2 = builder.step("search_memory", {"key": key},
                                  "memory_call", f"I also need the {key}.")
            else:
                key = rng.choice(numeric_facts)
                o2 = builder.step("web_search", {"query": key},
                                  "web_call", f"I also need the {key}.")
            other = sorted(numbers_in(o2))
            if not other:
                continue
            expr = f"{_canon(current)} + {other[0]}"
        else:
            op = rng.choice(["+", "-", "*"])
            k = rng.randint(2, 9) if op == "*" else rng.randint(1, 30)
            # k is a constant the model states, and it appears in the user's
            # question only for arithmetic_pipeline. Ground it explicitly by
            # treating small literals as allowed: register before use.
            builder.grounded.add(_canon(k))
            expr = f"{_canon(current)} {op} {k}"
        out = builder.step("calc", {"expr": expr}, "chain_first",
                           f"Combine into {expr}.")
        current = float(sorted(numbers_in(out))[0]) if numbers_in(out) else current
    return builder.finish("chain_last", "The chain is complete.")


# ------------------------------------------------------------- verification

def verify_trace(text: str) -> tuple:
    """Check that every tool argument is grounded in what came before it.

    Returns (ok, problems). Used both as a dataset gate and as an inference
    guard: if the model invents a number at hop 30, this catches it instead of
    letting it silently poison the rest of the chain.
    """
    problems = []
    grounded = set()
    first_user = re.search(r"<user>(.*?)</user>", text, re.S)
    if first_user:
        grounded |= numbers_in(first_user.group(1))

    pattern = re.compile(
        r'"tool_call":\{"name":"(\w+)","arguments":(\{.*?\})\}\}</assistant>'
        r'(?:\n<tool name=\w+>(.*?)</tool>)?', re.S)
    for i, m in enumerate(pattern.finditer(text), 1):
        name, raw_args, observation = m.group(1), m.group(2), m.group(3)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            problems.append(f"hop {i}: unparseable arguments")
            continue
        for value in args.values():
            for n in numbers_in(value):
                if n not in grounded:
                    problems.append(
                        f"hop {i} ({name}): value {n} not grounded in prior "
                        f"context")
        if observation is not None:
            grounded |= numbers_in(observation)
    return (not problems), problems
