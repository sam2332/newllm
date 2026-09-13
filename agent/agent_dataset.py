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


def _number_text(value: float) -> str:
    """Format a number exactly as the calculator returns integer results."""
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


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
    numeric_text = _number_text(numeric)
    if op == "+":
        ans = numeric + delta
        expr = f"{numeric_text} + {delta}"
        op_word = "add"
    elif op == "-":
        ans = numeric - delta
        expr = f"{numeric_text} - {delta}"
        op_word = "subtract"
    else:
        ans = numeric * delta
        expr = f"{numeric_text} * {delta}"
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
        {"role": "tool", "name": "calc", "content": _number_text(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The final answer is computed.",
            response=_number_text(ans))},
    ]


def _multi_hop_question(rng: random.Random, memory: dict) -> list:
    key = "version"
    a = rng.randint(2, 50)
    b = rng.randint(2, 50)
    raw = memory[key]
    base = float(raw)
    first_result = a * b
    ans = first_result + base
    expr = f"{first_result} + {base}"
    templates = [
        f"Multiply {a} and {b}, then add the stored {key}.",
        f"Compute {a} times {b} and add the {key} value.",
    ]
    question = rng.choice(templates)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "First I will multiply the two numbers.",
            tool_call={"name": "calc", "arguments": {"expr": f"{a} * {b}"}})},
        {"role": "tool", "name": "calc", "content": str(first_result)},
        {"role": "assistant", "content": _assistant_json(
            f"Then I need the stored {key} value.",
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


def _two_stage_math_question(rng: random.Random) -> list:
    """Three calculator turns where each expression uses the prior result."""
    a = rng.randint(2, 30)
    b = rng.randint(2, 30)
    delta = rng.randint(1, 25)
    factor = rng.randint(2, 10)
    first = a * b
    second = first + delta
    answer = second * factor
    question = (
        f"Multiply {a} by {b}, add {delta} to that result, then multiply "
        f"the result by {factor}."
    )
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "First I will calculate the product.",
            tool_call={"name": "calc", "arguments": {"expr": f"{a} * {b}"}})},
        {"role": "tool", "name": "calc", "content": str(first)},
        {"role": "assistant", "content": _assistant_json(
            "Next I will add the requested amount to that result.",
            tool_call={"name": "calc", "arguments": {"expr": f"{first} + {delta}"}})},
        {"role": "tool", "name": "calc", "content": str(second)},
        {"role": "assistant", "content": _assistant_json(
            "Finally I will multiply the intermediate result.",
            tool_call={"name": "calc", "arguments": {"expr": f"{second} * {factor}"}})},
        {"role": "tool", "name": "calc", "content": str(answer)},
        {"role": "assistant", "content": _assistant_json(
            "The final calculation is complete.",
            response=str(answer))},
    ]


def _web_two_stage_math(rng: random.Random) -> list:
    """Retrieve a numeric web fact, then perform two dependent calculations."""
    key = rng.choice(["number of planets", "boiling point of water", "freezing point of water"])
    value = float(_WEB_KB[key].split()[0])
    delta = rng.randint(1, 20)
    factor = rng.randint(2, 8)
    value_text = _number_text(value)
    first = value + delta
    first_text = _number_text(first)
    answer = first * factor
    answer_text = _number_text(answer)
    question = f"Look up the {key}, add {delta}, then multiply by {factor}."
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "I need the numeric web fact first.",
            tool_call={"name": "web_search", "arguments": {"query": key}})},
        {"role": "tool", "name": "web_search", "content": _WEB_KB[key]},
        {"role": "assistant", "content": _assistant_json(
            "I will add the requested amount to the retrieved value.",
            tool_call={"name": "calc", "arguments": {"expr": f"{value_text} + {delta}"}})},
        {"role": "tool", "name": "calc", "content": first_text},
        {"role": "assistant", "content": _assistant_json(
            "Now I will multiply the intermediate result.",
            tool_call={"name": "calc", "arguments": {"expr": f"{first_text} * {factor}"}})},
        {"role": "tool", "name": "calc", "content": answer_text},
        {"role": "assistant", "content": _assistant_json(
            "The web-grounded calculation is complete.",
            response=answer_text)},
    ]


def _cross_source_math(rng: random.Random, memory: dict) -> list:
    """Combine a numeric memory value and numeric web fact in a final calc."""
    memory_key = "version"
    web_key = "number of planets"
    memory_value = float(memory[memory_key])
    web_value = float(_WEB_KB[web_key])
    memory_text = _number_text(memory_value)
    web_text = _number_text(web_value)
    factor = rng.randint(2, 10)
    answer = (memory_value + web_value) * factor
    answer_text = _number_text(answer)
    expr = f"({memory_text} + {web_text}) * {factor}"
    question = (
        f"Add the stored {memory_key} to the {web_key}, then multiply the "
        f"result by {factor}."
    )
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": _assistant_json(
            "First I need the stored numeric value.",
            tool_call={"name": "search_memory", "arguments": {"key": memory_key}})},
        {"role": "tool", "name": "search_memory", "content": memory[memory_key]},
        {"role": "assistant", "content": _assistant_json(
            "Next I need the numeric web fact.",
            tool_call={"name": "web_search", "arguments": {"query": web_key}})},
        {"role": "tool", "name": "web_search", "content": _WEB_KB[web_key]},
        {"role": "assistant", "content": _assistant_json(
            "I can now combine both retrieved values.",
            tool_call={"name": "calc", "arguments": {"expr": expr}})},
        {"role": "tool", "name": "calc", "content": answer_text},
        {"role": "assistant", "content": _assistant_json(
            "The cross-source calculation is complete.",
            response=answer_text)},
    ]


def _memory_then_math(rng: random.Random, memory: dict) -> list:
    key = "version"
    raw = memory[key]
    base = float(raw)
    delta = rng.randint(1, 100)
    ans = base + delta
    expr = f"{_number_text(base)} + {delta}"
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
        {"role": "tool", "name": "calc", "content": _number_text(ans)},
        {"role": "assistant", "content": _assistant_json(
            "The final answer is computed.",
            response=_number_text(ans))},
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
            ["math", "memory", "date", "web", "web_math", "memory_math", "multi_hop",
             "two_stage_math", "web_two_stage", "cross_source"],
            weights=[15, 10, 8, 10, 12, 12, 10, 10, 7, 6],
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
        elif kind == "multi_hop":
            messages = _multi_hop_question(rng, memory)
        elif kind == "two_stage_math":
            messages = _two_stage_math_question(rng)
        elif kind == "web_two_stage":
            messages = _web_two_stage_math(rng)
        else:
            messages = _cross_source_math(rng, memory)
        samples.append(_truncate(_serialize(messages), max_len=max_len))
    return samples
