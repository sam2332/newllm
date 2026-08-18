"""Generate synthetic agent training data as JSON message conversations.

Every assistant response is valid JSON containing:
    {"thought": "...", "tool_call": {"name": "...", "arguments": {...}}}
or
    {"thought": "...", "response": "..."}

Raw digits are used everywhere so the byte-level tokenizer can copy numbers
as-is.
"""

import json
import random
import datetime


EOT = "\x03"

SYSTEM_PROMPT = (
    "You are a helpful reasoning agent. Available tools: calc(expr), "
    "search_memory(key), web_search(query), now(). "
    "Every response must be valid JSON with a 'thought' string and exactly "
    "one of 'tool_call' or 'response'. "
    "Use tool_call to call a tool: {\"thought\": \"...\", \"tool_call\": "
    "{\"name\": \"calc\", \"arguments\": {\"expr\": \"12 + 8\"}}}. "
    "Use response to give the final answer: {\"thought\": \"...\", "
    "\"response\": \"20\"}."
)


def _assistant_json(thought: str, response: str = None, tool_call: dict = None) -> str:
    data = {"thought": thought}
    if response is not None:
        data["response"] = response
    if tool_call is not None:
        data["tool_call"] = tool_call
    return json.dumps(data, separators=(",", ":"))


def _math_question(rng: random.Random, single_digit: bool = False) -> list:
    if single_digit:
        a = rng.randint(0, 9)
        b = rng.randint(1, 9)
    else:
        a = rng.randint(1, 200)
        b = rng.randint(1, 200)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    elif op == "*":
        ans = a * b
    else:
        b = max(1, b)
        ans = round(a / b, 4)
    expr = f"{a} {op} {b}"
    templates = [
        f"What is {a} {op} {b}?",
        f"Calculate {a} {op} {b}.",
        f"Compute {a} {op} {b}.",
        f"Tell me the result of {a} {op} {b}.",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "I need to use the calculator.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": str(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The calculation is complete.",
            response=str(ans))},
    ]


def _memory_question(rng: random.Random, memory: dict) -> list:
    key = rng.choice(list(memory.keys()))
    templates = [
        f"What is the {key}?",
        f"Retrieve the {key}.",
        f"Look up the {key}.",
        f"Tell me the value of {key}.",
    ]
    question = rng.choice(templates)
    value = memory[key]
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            f"I should search memory for '{key}'.",
            tool_call={"name": "search_memory", "arguments": {"key": key}})},
        {"role": "tool", "name": "search_memory", "content": str(value)},
        {"role": "assistant", "content": _assistant_json(
            "I found the stored value.",
            response=str(value))},
    ]


def _date_question(rng: random.Random) -> list:
    now = datetime.datetime.now().isoformat()
    templates = [
        "What is the current date and time?",
        "What time is it now?",
        "Return the current timestamp.",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "I need the current time.",
            tool_call={"name": "now", "arguments": {}})},
        {"role": "tool", "name": "now", "content": now},
        {"role": "assistant", "content": _assistant_json(
            "Here is the current timestamp.",
            response=now)},
    ]


_WEB_KB = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of germany": "Berlin",
    "president of the united states": "Alice Johnson",
    "current president": "Alice Johnson",
    "prime minister of the uk": "Bob Williams",
    "largest planet": "Jupiter",
    "smallest planet": "Mercury",
    "number of planets": "8",
    "speed of light": "299792458 m/s",
    "boiling point of water": "100 degrees Celsius",
    "freezing point of water": "0 degrees Celsius",
}


def _web_lookup(query: str) -> str:
    q = query.strip().lower().rstrip("?")
    if q in _WEB_KB:
        return _WEB_KB[q]
    for key, value in _WEB_KB.items():
        if key in q or q in key:
            return value
    return "no results found"


def _web_search_question(rng: random.Random) -> list:
    key = rng.choice(list(_WEB_KB.keys()))
    templates = [
        f"What is the {key}?",
        f"Look up the {key}.",
        f"Search the web for the {key}.",
        f"Find the {key} online.",
    ]
    question = rng.choice(templates)
    answer = _WEB_KB[key]
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "I do not know this from memory, so I should search the web.",
            tool_call={"name": "web_search", "arguments": {"query": key}})},
        {"role": "tool", "name": "web_search", "content": answer},
        {"role": "assistant", "content": _assistant_json(
            "I found the answer online.",
            response=answer)},
    ]


def _web_then_math(rng: random.Random) -> list:
    key = rng.choice(list(_WEB_KB.keys()))
    value = _WEB_KB[key]
    try:
        numeric = float(value.split()[0])
    except Exception:
        numeric = float(len(value))
    delta = rng.randint(1, 100)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = numeric + delta
        expr = f"{numeric} + {delta}"
        op_word = "add"
    elif op == "-":
        ans = numeric - delta
        expr = f"{numeric} - {delta}"
        op_word = "subtract"
    else:
        ans = numeric * delta
        expr = f"{numeric} * {delta}"
        op_word = "multiply by"
    templates = [
        f"What is the {key} {op_word} {delta}?",
        f"Look up the {key} and {op_word} {delta}.",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "I need to retrieve the web fact first.",
            tool_call={"name": "web_search", "arguments": {"query": key}})},
        {"role": "tool", "name": "web_search", "content": value},
        {"role": "assistant", "content": _assistant_json(
            "Now I can combine it with arithmetic.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": str(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The final answer is computed.",
            response=str(ans))},
    ]


def _multi_hop_question(rng: random.Random, memory: dict) -> list:
    key = rng.choice(list(memory.keys()))
    a = rng.randint(2, 50)
    b = rng.randint(2, 50)
    raw = memory[key]
    length = len(raw) if isinstance(raw, str) else 0
    op = rng.choice(["+", "*"])
    if op == "+":
        ans = a * b + length
        expr = f"{a} * {b} + {length}"
    else:
        ans = a * b * length
        expr = f"{a} * {b} * {length}"
    templates = [
        f"Multiply {a} and {b}, then {op} the length of the {key}.",
        f"Compute {a} times {b} and {op} the number of characters in the {key}.",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "First I will multiply the two numbers.",
            tool_call={"name": "calc", "arguments": {"expr": f"{a} * {b}"}})},
        {"role": "tool", "name": "calc", "content": str(a * b)},
        {"role": "assistant", "content": _assistant_json(
            f"Then I need the length of the stored value '{key}'.",
            tool_call={"name": "search_memory", "arguments": {"key": key}})},
        {"role": "tool", "name": "search_memory", "content": str(raw)},
        {"role": "assistant", "content": _assistant_json(
            "Now I can combine the intermediate results.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": str(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The final answer is computed.",
            response=str(ans))},
    ]


def _memory_then_math(rng: random.Random, memory: dict) -> list:
    key = rng.choice(list(memory.keys()))
    raw = memory[key]
    try:
        base = float(raw)
    except Exception:
        base = len(raw)
    delta = rng.randint(1, 100)
    ans = base + delta
    expr = f"{base} + {delta}"
    templates = [
        f"Add {delta} to the {key}.",
        f"What is the {key} plus {delta}?",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "First retrieve the stored value.",
            tool_call={"name": "search_memory", "arguments": {"key": key}})},
        {"role": "tool", "name": "search_memory", "content": str(raw)},
        {"role": "assistant", "content": _assistant_json(
            "Then add the requested amount.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": str(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The final answer is computed.",
            response=str(ans))},
    ]


def _serialize(messages: list, include_system: bool = True) -> str:
    """Render a message list as a single string for the byte tokenizer.

    Trailing assistant messages are forced onto a single line to make
    structure easier to learn for the byte-level model.
    """
    parts = []
    for m in messages:
        if not include_system and m["role"] == "system":
            continue
        if m["role"] == "tool":
            parts.append(f"<tool name={m['name']}>{m['content']}</tool>")
        else:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
    return "\n".join(parts) + EOT


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    # Keep the final assistant answer block if possible.
    end_idx = text.rfind("<assistant>")
    if end_idx == -1:
        return text[:max_len]
    tail = text[end_idx:]
    head_budget = max_len - len(tail)
    if head_budget <= 0:
        return tail[-max_len:]
    return text[:head_budget] + tail


def generate_simple_agent_dataset(num_samples: int = 100000, max_len: int = 512,
                                  seed: int = 42, math_only: bool = False,
                                  single_digit_math: bool = False) -> list:
    """Single-step JSON message traces only (math, memory, date, web)."""
    rng = random.Random(seed)
    memory = {
        "project": "newllm",
        "device": "cuda",
        "leader": "grug",
        "tribe": "cavepeople",
        "language": "Python",
        "status": "active",
        "version": "0.1",
    }
    samples = []
    for _ in range(num_samples):
        if math_only:
            kind = "math"
        else:
            kind = rng.choices(
                ["math", "memory", "date", "web"],
                weights=[35, 30, 20, 15],
                k=1,
            )[0]
        if kind == "math":
            messages = _math_question(rng, single_digit=single_digit_math)
        elif kind == "memory":
            messages = _memory_question(rng, memory)
        elif kind == "date":
            messages = _date_question(rng)
        else:
            messages = _web_search_question(rng)
        samples.append(_truncate(_serialize(messages), max_len=max_len))
    return samples


def generate_agent_dataset(num_samples: int = 100000, max_len: int = 512, seed: int = 42) -> list:
    """Full agent dataset with single-step + multi-hop JSON message traces."""
    rng = random.Random(seed)
    memory = {
        "project": "newllm",
        "device": "cuda",
        "leader": "grug",
        "tribe": "cavepeople",
        "language": "Python",
        "status": "active",
        "version": "0.1",
    }
    samples = []
    for _ in range(num_samples):
        kind = rng.choices(
            ["math", "memory", "date", "web", "web_math", "memory_math", "multi_hop"],
            weights=[20, 15, 10, 15, 15, 15, 10],
            k=1,
        )[0]
        if kind == "math":
            messages = _math_question(rng)
        elif kind == "memory":
            messages = _memory_question(rng, memory)
        elif kind == "date":
            messages = _date_question(rng)
        elif kind == "web":
            messages = _web_search_question(rng)
        elif kind == "web_math":
            messages = _web_then_math(rng)
        elif kind == "memory_math":
            messages = _memory_then_math(rng, memory)
        else:
            messages = _multi_hop_question(rng, memory)
        samples.append(_truncate(_serialize(messages), max_len=max_len))
    return samples
