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
# Must handle scientific notation: generated physics facts look like
# "9.578833205e7 C/kg", and a regex without an exponent clause reads that as
# the two numbers 9.5788 and 7. The chain then computes correctly from
# 95788332.05 while the verifier, unable to see that value anywhere, reports a
# hallucination. A false alarm on correct behaviour is the failure mode that
# gets a safety check disabled.
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def numbers_in(text: str) -> set:
    """Numeric literals appearing in a piece of text, normalized."""
    return set(numbers_in_order(text))


def numbers_in_order(text: str) -> list:
    """Numeric literals in order of appearance.

    Order matters: a tool result's *first* number is its answer. Taking
    ``sorted(numbers_in(x))[0]`` sorts normalized STRINGS lexicographically,
    so "1000" sorts before "9", and the chain silently continues from the
    wrong operand.
    """
    out = []
    for m in _NUM.finditer(str(text)):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        out.append(_canon(v))
    return out


def floats_in_order(text: str) -> list:
    """Numeric literals as floats, in order of appearance."""
    out = []
    for m in _NUM.finditer(str(text)):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _is_grounded(value: float, pool, rel_tol: float = 1e-6) -> bool:
    """Numeric membership with tolerance.

    Comparing formatted strings is fragile in both directions: rounding to a
    fixed number of decimals collapses 1.6e-19 to 0.0, and full precision makes
    95788332.15 and 95788332.150001 look like different values. Compare as
    numbers instead.
    """
    for known in pool:
        if value == known:
            return True
        scale = max(abs(value), abs(known), 1e-12)
        if abs(value - known) / scale <= rel_tol:
            return True
    return False


def _canon(v) -> str:
    """Format a number for an expression without losing magnitude.

    round(x, 4) turns 1.6e-19 into 0.0, which both corrupts the arithmetic and
    makes the value unverifiable. Use significant digits instead.
    """
    f = float(v)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.10g}"


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
        self.grounded = list(floats_in_order(question))
        self.last = None
        return self

    def _assert_grounded(self, args: dict):
        """Refuse any argument value not already visible in the transcript."""
        for value in args.values():
            for n in floats_in_order(value):
                if not _is_grounded(n, self.grounded):
                    raise GroundingError(
                        f"value {_canon(n)} in {args!r} is not grounded")

    def step(self, tool: str, args: dict, situation: str, fallback: str):
        """Run one tool stage. Raises GroundingError if args are ungrounded."""
        if tool in DATAFLOW_TOOLS:
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
        self.grounded.extend(floats_in_order(observation))
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
    """Alternating lookups and arithmetic; every operand comes from a tool.

    The operations are stated in the question up front. An earlier version drew
    the constants randomly without mentioning them, which meant the only way to
    produce the trace was to invent numbers - exactly the behaviour this module
    exists to prevent. The verifier caught it on 600 of 3000 traces.
    """
    rng = builder.rng
    numeric_facts = [k for k, v in facts.items() if floats_in_order(v)]
    numeric_mem = [k for k, v in memory.items() if floats_in_order(v)]
    if not numeric_facts:
        return arithmetic_pipeline(builder, hops)

    first_key = rng.choice(numeric_facts)
    stages = max(1, hops - 1)

    # Plan every stage before writing the question, so the question can state
    # each constant and each additional lookup the chain will use.
    plan = []
    for i in range(stages):
        if i % 3 == 2 and (numeric_facts or numeric_mem):
            if numeric_mem and rng.random() < 0.5:
                plan.append(("mem", rng.choice(numeric_mem)))
            else:
                plan.append(("web", rng.choice(numeric_facts)))
        else:
            op = rng.choice(["+", "-", "*"])
            k = rng.randint(2, 9) if op == "*" else rng.randint(1, 30)
            plan.append(("op", (op, k)))

    described = []
    for kind, payload in plan:
        if kind == "op":
            op, k = payload
            word = {"+": "add", "-": "subtract", "*": "multiply by"}[op]
            described.append(f"{word} {k}")
        elif kind == "mem":
            described.append(f"add the stored {payload}")
        else:
            described.append(f"add the {payload}")
    question = (f"Look up the {first_key}, then {', then '.join(described)}. "
                f"Use a tool for every step.")

    builder.start(question)
    obs = builder.step("web_search", {"query": first_key},
                       "web_call", f"First I need the {first_key}.")
    nums = floats_in_order(obs)
    if not nums:
        raise GroundingError("first lookup returned no number")
    current = nums[0]

    for kind, payload in plan:
        if kind == "op":
            op, k = payload
            expr = f"{_canon(current)} {op} {k}"
        else:
            if kind == "mem":
                other_obs = builder.step("search_memory", {"key": payload},
                                         "memory_call",
                                         f"I also need the stored {payload}.")
            else:
                other_obs = builder.step("web_search", {"query": payload},
                                         "web_call",
                                         f"I also need the {payload}.")
            other = floats_in_order(other_obs)
            if not other:
                continue
            expr = f"{_canon(current)} + {_canon(other[0])}"
        out = builder.step("calc", {"expr": expr}, "chain_first",
                           f"Combine into {expr}.")
        got = floats_in_order(out)
        if got:
            current = got[0]
    return builder.finish("chain_last", "The chain is complete.")


# ------------------------------------------------------------- verification

# Grounding is only meaningful where a numeric argument is a DATA-FLOW claim:
# the model asserting "this is the value the previous step produced". In a code
# snippet or a line range, a number is part of an instruction, not a claim about
# prior output, and demanding it appear earlier flags correct behaviour. A
# verifier that cries wolf gets switched off, so it checks only what it can
# meaningfully check.
DATAFLOW_TOOLS = frozenset({"calc"})


def verify_trace(text: str, dataflow_tools=DATAFLOW_TOOLS) -> tuple:
    """Check that data-flow tool arguments are grounded in what came before.

    Returns (ok, problems). Used both as a dataset gate and as an inference
    guard: if the model invents a number at hop 30, this catches it instead of
    letting it silently poison the rest of the chain.
    """
    problems = []
    grounded = []

    # Walk the transcript in document order. An earlier version seeded
    # grounding from only the FIRST <user> block, so in a multi-turn
    # conversation a constant stated in turn 2 ("now multiply that by 7") was
    # invisible and every later computation was flagged. That rejected 97% of
    # legitimate multi-turn traces, and at inference it would have fired on
    # correct behaviour - the failure mode that gets a safety check switched off.
    event = re.compile(
        r"<user>(?P<user>.*?)</user>"
        r"|<tool name=\w+>(?P<obs>.*?)</tool>"
        r"|\"tool_call\":\{\"name\":\"(?P<tool>\w+)\","
        r"\"arguments\":(?P<args>\{.*?\})\}\}</assistant>",
        re.S)

    hop = 0
    for m in event.finditer(text):
        if m.group("user") is not None:
            grounded.extend(floats_in_order(m.group("user")))
        elif m.group("obs") is not None:
            grounded.extend(floats_in_order(m.group("obs")))
        else:
            hop += 1
            name = m.group("tool")
            if name not in dataflow_tools:
                continue
            try:
                args = json.loads(m.group("args"))
            except json.JSONDecodeError:
                problems.append(f"hop {hop}: unparseable arguments")
                continue
            for value in args.values():
                for n in floats_in_order(value):
                    if not _is_grounded(n, grounded):
                        problems.append(
                            f"hop {hop} ({name}): value {_canon(n)} not "
                            f"grounded in prior context")
    return (not problems), problems
