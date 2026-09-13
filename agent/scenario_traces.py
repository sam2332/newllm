"""Instantiate teacher-written scenario plans into verified training traces.

The teacher supplies structure only: how many turns, which step kinds, where a
follow-up refers back. This module binds concrete facts, memory keys and
operands, executes every step against the real Toolbox, and emits a multi-turn
trace. One plan instantiates into many traces with different bindings, which is
what makes a few thousand teacher calls worth hundreds of thousands of samples.

Two properties are enforced rather than hoped for:

  * every operand a step consumes is already visible in the transcript, so no
    trace can be reproduced only by inventing a number
  * every constant the agent uses is stated in the user's message, so the
    user's request fully determines the work

``failing_call`` is deliberately included. Every tool call in the earlier
datasets succeeded, so the model never learned that a tool can fail and the
chain must adapt - which is precisely what breaks 25-60 call tasks in practice.
"""

import json
import random

from agent.deep_chains import (_canon, floats_in_order, GroundingError,
                               verify_trace)
from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools

EOT = "\x03"

# Calls that fail for a realistic reason, paired with how the agent notices.
FAILURE_CASES = [
    ("search_memory", {"key": "budget"}, "not found"),
    ("search_memory", {"key": "deadline"}, "not found"),
    ("web_search", {"query": "the exact value I need"}, "no results found"),
    ("calc", {"expr": "10 / 0"}, "ERROR"),
]


class ScenarioInstantiator:
    """Binds a plan to concrete values and executes it for real."""

    def __init__(self, facts: dict, memory: dict, rng: random.Random,
                 thought_fn=None, sandbox_pool=None):
        self.facts = facts
        self.memory = memory
        self.rng = rng
        self.thought = thought_fn or (lambda situation, fallback: fallback)
        self.sandbox_pool = sandbox_pool or []
        self.toolbox = attach_repo_tools(
            Toolbox(memory=memory, web_kb=facts or None))
        self.numeric_facts = [k for k, v in facts.items() if floats_in_order(v)]
        self.numeric_mem = [k for k, v in memory.items() if floats_in_order(v)]

    # ------------------------------------------------------------------
    def _aj(self, thought, response=None, tool_call=None) -> str:
        data = {"thought": thought}
        if response is not None:
            data["response"] = response
        if tool_call is not None:
            data["tool_call"] = tool_call
        return json.dumps(data, separators=(",", ":"))

    def _plan_turn(self, steps, carried):
        """Choose concrete bindings for one turn before writing the user text.

        Planning before writing is what lets the user's message name every
        constant and every lookup the turn will use. Choosing them mid-execution
        produced traces whose only possible source was invention; the verifier
        rejected 600 of 3000 before this was fixed.
        """
        bound = []
        mentions = []
        have_value = carried is not None
        for step in steps:
            kind = step["kind"]
            if kind == "lookup_fact" and self.numeric_facts:
                key = self.rng.choice(self.numeric_facts)
                bound.append(("lookup_fact", key))
                mentions.append(f"look up the {key}")
                have_value = True
            elif kind == "lookup_memory" and self.numeric_mem:
                key = self.rng.choice(self.numeric_mem)
                bound.append(("lookup_memory", key))
                mentions.append(f"check the stored {key}")
                have_value = True
            elif kind == "time":
                bound.append(("time", None))
                mentions.append("note the current time")
            elif kind == "run_code" and self.sandbox_pool:
                item = self.rng.choice(self.sandbox_pool)
                bound.append(("run_code", item))
                mentions.append(item["question"].rstrip(".?").lower())
            elif kind == "inspect_code":
                sym = self.rng.choice(["RMSNorm", "Toolbox", "AgentTokenizer",
                                       "TransformerBlock", "Trainer"])
                bound.append(("inspect_code", sym))
                mentions.append(f"check what {sym} does")
            elif kind == "failing_call":
                case = self.rng.choice(FAILURE_CASES)
                bound.append(("failing_call", case))
            elif kind in ("compute", "combine") and have_value:
                op = self.rng.choice(["+", "-", "*"])
                k = self.rng.randint(2, 9) if op == "*" else self.rng.randint(1, 40)
                word = {"+": "add", "-": "subtract", "*": "multiply by"}[op]
                bound.append(("compute", (op, k)))
                mentions.append(f"{word} {k}")
            # steps that cannot bind are simply dropped
        return bound, mentions

    def _execute(self, tool, args, situation, fallback, messages,
                 allow_error=False):
        observation = self.toolbox.run_json({"tool": tool, "args": args})
        text = str(observation)
        failed = text.startswith("ERROR") or text in ("not found",
                                                      "no results found")
        if failed and not allow_error:
            raise GroundingError(f"{tool} failed: {text[:40]}")
        messages.append({"role": "assistant", "content": self._aj(
            self.thought(situation, fallback),
            tool_call={"name": tool, "arguments": args})})
        messages.append({"role": "tool", "name": tool, "content": text})
        return text

    def instantiate(self, plan: dict):
        """Return message list for a full multi-turn conversation, or None."""
        messages = []
        carried = None          # last numeric result, for cross-turn reference
        turns_done = 0
        for t_index, turn in enumerate(plan["turns"]):
            bound, mentions = self._plan_turn(turn["steps"], carried)
            if not bound:
                continue
            # Write the user message so it names every constant used below.
            if t_index == 0 or carried is None:
                user = f"{', then '.join(mentions).capitalize()}." if mentions \
                    else turn["user"]
            else:
                ref = self.rng.choice(["that", "that result", "it"])
                user = (f"Now take {ref} and "
                        f"{', then '.join(mentions)}." if mentions
                        else turn["user"])
            messages.append({"role": "user", "content": user})

            current = carried
            for kind, payload in bound:
                if kind == "lookup_fact":
                    out = self._execute("web_search", {"query": payload},
                                        "web_call",
                                        f"I need the {payload}.", messages)
                    nums = floats_in_order(out)
                    if nums:
                        current = nums[0]
                elif kind == "lookup_memory":
                    out = self._execute("search_memory", {"key": payload},
                                        "memory_call",
                                        f"I should check the stored {payload}.",
                                        messages)
                    nums = floats_in_order(out)
                    if nums:
                        current = nums[0]
                elif kind == "time":
                    self._execute("now", {}, "time_call",
                                  "I need the current time.", messages)
                elif kind == "run_code":
                    messages.append({"role": "assistant", "content": self._aj(
                        self.thought("sandbox_call",
                                     "I will run this in the sandbox."),
                        tool_call={"name": payload["tool"],
                                   "arguments": payload["args"]})})
                    messages.append({"role": "tool", "name": payload["tool"],
                                     "content": payload["output"]})
                    nums = floats_in_order(payload["output"])
                    if nums:
                        current = nums[0]
                elif kind == "inspect_code":
                    out = self._execute("describe_symbol", {"name": payload},
                                        "repo_call",
                                        f"I should look up {payload} in my "
                                        f"own source.", messages)
                    if len(out) > 300:
                        messages[-1]["content"] = out[:300] + "\n..."
                elif kind == "failing_call":
                    tool, args, _ = payload
                    self._execute(tool, args, "repo_call",
                                  "Let me try this.", messages,
                                  allow_error=True)
                    # Recovery: the agent must notice and say so, not pretend.
                    messages.append({"role": "assistant", "content": self._aj(
                        self.thought("no_tool",
                                     "That call did not return a usable "
                                     "value, so I will not use it."),
                        response="That lookup failed, so I could not use it.")})
                elif kind == "compute":
                    if current is None:
                        continue
                    op, k = payload
                    expr = f"{_canon(current)} {op} {k}"
                    out = self._execute("calc", {"expr": expr}, "chain_first",
                                        f"Compute {expr}.", messages)
                    nums = floats_in_order(out)
                    if nums:
                        current = nums[0]

            if messages and messages[-1]["role"] == "tool":
                messages.append({"role": "assistant", "content": self._aj(
                    self.thought("chain_last", "That completes this step."),
                    response=messages[-1]["content"])})
            carried = current
            turns_done += 1
        return messages if turns_done else None


def serialize(messages) -> str:
    """Tagged form, with EOT ending every assistant turn that answers."""
    parts = []
    for m in messages:
        if m["role"] == "tool":
            parts.append(f"<tool name={m['name']}>{m['content']}</tool>")
        else:
            block = f"<{m['role']}>{m['content']}</{m['role']}>"
            if m["role"] == "assistant" and '"response":' in m["content"]:
                block += EOT
            parts.append(block)
    return "\n".join(parts)


def generate_from_scenarios(plans, facts, memory, count, rng_seed=42,
                            max_len=16384, thought_fn=None, sandbox_pool=None,
                            verify=True):
    """Instantiate plans repeatedly into verified traces."""
    rng = random.Random(rng_seed)
    inst = ScenarioInstantiator(facts, memory, rng, thought_fn, sandbox_pool)
    traces, rejected, too_long, failed = [], 0, 0, 0
    guard = 0
    while len(traces) < count and guard < count * 20:
        guard += 1
        plan = rng.choice(plans)
        try:
            messages = inst.instantiate(plan)
        except GroundingError:
            failed += 1
            continue
        if not messages:
            failed += 1
            continue
        text = serialize(messages)
        if len(text) > max_len:
            too_long += 1
            continue
        if verify and not verify_trace(text)[0]:
            rejected += 1
            continue
        traces.append(text)
    return traces, {"ungrounded_rejected": rejected, "too_long": too_long,
                    "build_failed": failed}
