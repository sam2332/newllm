"""Multi-turn chat conversations for the chat/assistant model.

The instruct dataset (``agent/agent_dataset.py``) teaches one skill: turn a
single request into a tool plan and an answer. It cannot teach conversation,
because every trace is exactly one exchange.

This module generates multi-turn conversations whose later turns are only
answerable by reading earlier ones:

  * pronoun/coreference follow-ups   "multiply that by 3"
  * ellipsis                         "and Japan?"  (verb and object omitted)
  * back-reference to a prior answer "what was the first number I asked about?"
  * topic switches, so the model does not blindly reuse the last entity
  * clarification turns with no tool call at all

Every assistant turn ends with EOT, which is what teaches the model to stop and
yield to the user rather than inventing the next user message.
"""

import json
import random
import datetime

EOT = "\x03"

SYSTEM_PROMPTS = [
    "You are a helpful assistant with tools: calc, search_memory, web_search, now.",
    "You are a concise reasoning assistant. Use tools when you need facts or arithmetic.",
    "You answer questions using tools. Always explain your reasoning in 'thought'.",
]

MEMORY = {
    "project": "newllm", "device": "cuda", "leader": "grug",
    "tribe": "cavepeople", "language": "Python", "status": "active",
    "version": "0.1",
}

WEB_KB = {
    "capital of france": "Paris", "capital of japan": "Tokyo",
    "capital of germany": "Berlin",
    "president of the united states": "Alice Johnson",
    "prime minister of the uk": "Bob Williams",
    "largest planet": "Jupiter", "smallest planet": "Mercury",
    "number of planets": "8", "speed of light": "299792458 m/s",
    "boiling point of water": "100 degrees Celsius",
    "freezing point of water": "0 degrees Celsius",
}

CAPITALS = {k: v for k, v in WEB_KB.items() if k.startswith("capital of")}


def _aj(thought, response=None, tool_call=None) -> str:
    data = {"thought": thought}
    if response is not None:
        data["response"] = response
    if tool_call is not None:
        data["tool_call"] = tool_call
    return json.dumps(data, separators=(",", ":"))


def _num(value) -> str:
    f = float(value)
    return str(int(f)) if f.is_integer() else str(round(f, 4))


def _calc_turn(rng, question, expr, answer, thought_a=None, thought_b=None):
    """One user turn resolved by a single calc call."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _aj(
            thought_a or "I need the calculator for this.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": answer},
        {"role": "assistant", "content": _aj(
            thought_b or "The calculation is done.", response=answer)},
    ]


def _lookup_turn(rng, question, tool, args, result, thought_a, thought_b):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _aj(
            thought_a, tool_call={"name": tool, "arguments": args})},
        {"role": "tool", "name": tool, "content": result},
        {"role": "assistant", "content": _aj(thought_b, response=result)},
    ]


# ---------------------------------------------------------------- patterns

def _math_then_pronoun(rng):
    """'What is 12 + 8?' -> 20 -> 'Multiply that by 3.' -> 60"""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    first = a + b
    msgs = _calc_turn(rng, f"What is {a} + {b}?", f"{a} + {b}", _num(first))

    k = rng.randint(2, 12)
    op, word = rng.choice([("*", "Multiply that by"), ("+", "Add"),
                           ("-", "Subtract")])
    if word == "Add":
        follow, expr, ans = f"Add {k} to that.", f"{_num(first)} + {k}", first + k
    elif word == "Subtract":
        follow, expr, ans = f"Subtract {k} from that.", f"{_num(first)} - {k}", first - k
    else:
        follow, expr, ans = f"Multiply that by {k}.", f"{_num(first)} * {k}", first * k

    msgs += _calc_turn(
        rng, follow, expr, _num(ans),
        thought_a=f"'that' refers to the previous result {_num(first)}. "
                  f"I will compute {expr}.",
        thought_b="Done.")
    return msgs


def _capital_then_ellipsis(rng):
    """'Capital of France?' -> Paris -> 'And Japan?' -> Tokyo"""
    keys = list(CAPITALS.keys())
    rng.shuffle(keys)
    k1, k2 = keys[0], keys[1]
    c1, c2 = k1.split("capital of ")[1], k2.split("capital of ")[1]
    msgs = _lookup_turn(
        rng, f"What is the capital of {c1}?", "web_search", {"query": k1},
        CAPITALS[k1], "I should look this up.", "Found it.")
    follow = rng.choice([f"And {c2}?", f"What about {c2}?",
                         f"And what about {c2}?"])
    msgs += _lookup_turn(
        rng, follow, "web_search", {"query": k2}, CAPITALS[k2],
        f"The user is still asking about capitals, so '{c2}' means "
        f"the capital of {c2}.", "Found it.")
    return msgs


def _memory_then_followup(rng):
    """'Look up the leader.' -> grug -> 'And the project?' -> newllm"""
    keys = list(MEMORY.keys())
    rng.shuffle(keys)
    k1, k2 = keys[0], keys[1]
    msgs = _lookup_turn(
        rng, f"Look up the {k1}.", "search_memory", {"key": k1}, MEMORY[k1],
        f"I should search memory for '{k1}'.", "Found the stored value.")
    follow = rng.choice([f"And the {k2}?", f"What about the {k2}?",
                         f"Now the {k2}."])
    msgs += _lookup_turn(
        rng, follow, "search_memory", {"key": k2}, MEMORY[k2],
        f"The user is still asking about stored values, so this means "
        f"the {k2}.", "Found the stored value.")
    return msgs


def _web_then_compute_on_it(rng):
    """'Speed of light?' -> ... -> 'Divide it by 2.'"""
    numeric_keys = [k for k, v in WEB_KB.items()
                    if v.split()[0].replace(".", "").isdigit()]
    k = rng.choice(numeric_keys)
    value = WEB_KB[k]
    base = float(value.split()[0])
    msgs = _lookup_turn(
        rng, f"What is the {k}?", "web_search", {"query": k}, value,
        "I do not know this from memory; I should search the web.",
        "Found it online.")
    d = rng.choice([2, 3, 4, 5, 10])
    op, phrase = rng.choice([("*", f"Multiply it by {d}."),
                             ("+", f"Add {d} to it.")])
    expr = f"{_num(base)} {op} {d}"
    ans = base * d if op == "*" else base + d
    msgs += _calc_turn(
        rng, phrase, expr, _num(ans),
        thought_a=f"'it' refers to the {k}, which is {_num(base)}. "
                  f"I will compute {expr}.",
        thought_b="Done.")
    return msgs


def _back_reference(rng):
    """Third turn asks about the FIRST turn, not the most recent one."""
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    first = a + b
    msgs = _calc_turn(rng, f"What is {a} + {b}?", f"{a} + {b}", _num(first))
    key = rng.choice(list(MEMORY.keys()))
    msgs += _lookup_turn(
        rng, f"Look up the {key}.", "search_memory", {"key": key}, MEMORY[key],
        f"I should search memory for '{key}'.", "Found the stored value.")
    q = rng.choice(["What was the result of my first question?",
                    "Remind me what the first answer was.",
                    "What number did we get at the start?"])
    msgs += [
        {"role": "user", "content": q},
        # No tool call: the answer is already in the conversation.
        {"role": "assistant", "content": _aj(
            f"The first question was {a} + {b}, which gave {_num(first)}. "
            f"That is already in the conversation, so no tool is needed.",
            response=_num(first))},
    ]
    return msgs


def _topic_switch(rng):
    """Guards against blindly reusing the previous entity after a switch."""
    a, b = rng.randint(2, 50), rng.randint(2, 50)
    msgs = _calc_turn(rng, f"Compute {a} * {b}.", f"{a} * {b}", _num(a * b))
    key = rng.choice(list(WEB_KB.keys()))
    msgs += _lookup_turn(
        rng, f"Different question - what is the {key}?", "web_search",
        {"query": key}, WEB_KB[key],
        "This is a new topic, unrelated to the previous calculation.",
        "Found it.")
    return msgs


def _clarification(rng):
    """A turn the assistant answers with no tool call at all."""
    msgs = [
        {"role": "user", "content": rng.choice(
            ["What tools do you have?", "What can you do?",
             "Which tools can you use?"])},
        {"role": "assistant", "content": _aj(
            "This is a question about my own capabilities. No tool is needed.",
            response="I can use calc for arithmetic, search_memory for stored "
                     "values, web_search for facts, and now for the time.")},
    ]
    a, b = rng.randint(2, 60), rng.randint(2, 60)
    msgs += _calc_turn(rng, f"Great, what is {a} * {b}?", f"{a} * {b}",
                       _num(a * b))
    return msgs


def _time_then_followup(rng):
    now = datetime.datetime.now().isoformat()
    msgs = _lookup_turn(
        rng, "What time is it?", "now", {}, now,
        "I need the current time.", "Here is the timestamp.")
    key = rng.choice(list(MEMORY.keys()))
    msgs += _lookup_turn(
        rng, f"Thanks. Now look up the {key}.", "search_memory", {"key": key},
        MEMORY[key], f"Now the user wants the stored {key}.",
        "Found the stored value.")
    return msgs


PATTERNS = [
    (_math_then_pronoun, 20),
    (_capital_then_ellipsis, 14),
    (_memory_then_followup, 14),
    (_web_then_compute_on_it, 14),
    (_back_reference, 12),
    (_topic_switch, 10),
    (_clarification, 8),
    (_time_then_followup, 8),
]


def serialize_chat(messages: list, system: str = None) -> str:
    """Render a conversation, ending every assistant turn with EOT.

    EOT after each final assistant response is what teaches turn-taking: the
    model learns to stop and wait rather than hallucinating the user's reply.
    """
    parts = []
    if system:
        parts.append(f"<system>{system}</system>")
    for m in messages:
        if m["role"] == "tool":
            parts.append(f"<tool name={m['name']}>{m['content']}</tool>")
        else:
            block = f"<{m['role']}>{m['content']}</{m['role']}>"
            # An assistant turn that gives a final response ends the turn.
            if m["role"] == "assistant" and '"response":' in m["content"]:
                block += EOT
            parts.append(block)
    return "\n".join(parts)


def generate_chat_dataset(num_samples: int = 50000, max_len: int = 2048,
                          seed: int = 42, system_prob: float = 0.5) -> list:
    """Generate multi-turn conversations as tagged strings."""
    rng = random.Random(seed)
    fns = [f for f, _ in PATTERNS]
    weights = [w for _, w in PATTERNS]
    out = []
    for _ in range(num_samples):
        fn = rng.choices(fns, weights=weights, k=1)[0]
        messages = fn(rng)
        system = rng.choice(SYSTEM_PROMPTS) if rng.random() < system_prob else None
        text = serialize_chat(messages, system=system)
        if len(text) <= max_len:
            out.append(text)
    return out


if __name__ == "__main__":
    import collections
    data = generate_chat_dataset(num_samples=2000, seed=1)
    print(f"generated {len(data)} conversations")
    lens = [len(d) for d in data]
    print(f"length: mean {sum(lens)/len(lens):.0f} max {max(lens)}")
    turns = [d.count("<user>") for d in data]
    print(f"user turns per conversation: {collections.Counter(turns)}")
    print("\n--- sample ---")
    print(data[0].replace("\x03", "<EOT>"))
