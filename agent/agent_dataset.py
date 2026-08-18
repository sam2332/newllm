"""Generate synthetic agent training data (clean, non-caveman).

Each sample is a ReAct-style trace with explicit internal thinking:

Question: ... BEGIN_THINK Thought: ... Action: ... Observation: ... END_THINK Answer: ... <EOT>
"""

import random
import datetime


EOT = "\x03"
THINK_START = "BEGIN_THINK"
THINK_END = "END_THINK"


def _format_trace(question: str, steps: list, answer: str, use_json_tools: bool = True) -> str:
    """steps is list of (thought, tool_name, tool_arg, observation).

    If use_json_tools is True, actions are emitted as JSON function calls:
        Action: {"tool": "calc", "args": {"expr": "12 + 8"}}
    Otherwise the legacy bracket format is used:
        Action: calc[12 + 8]
    """
    lines = [f"Question: {question}", THINK_START]
    for thought, tool, arg, obs in steps:
        lines.append(f"Thought: {thought}")
        if use_json_tools:
            if tool == "calc":
                action = {"tool": "calc", "args": {"expr": arg}}
            elif tool == "search_memory":
                action = {"tool": "search_memory", "args": {"key": arg}}
            elif tool == "web_search":
                action = {"tool": "web_search", "args": {"query": arg}}
            elif tool == "now":
                action = {"tool": "now", "args": {}}
            elif tool == "finish":
                action = {"tool": "finish", "args": {}}
            else:
                action = {"tool": tool, "args": {"arg": arg}}
            import json
            lines.append(f"Action: {json.dumps(action, separators=(',', ':'))}")
        else:
            lines.append(f"Action: {tool}[{arg}]")
        lines.append(f"Observation: {obs}")
    lines.append(THINK_END)
    lines.append(f"Answer: {answer}")
    lines.append(EOT)
    return "\n".join(lines)


def _truncate_trace(text: str, max_len: int = 512) -> str:
    """Keep the question/header and the final answer if the trace is too long.

    Byte-level tokenization keeps one char per token, so max_len == max chars.
    For single-step traces this almost never fires; it is here as a safety net.
    """
    if len(text) <= max_len:
        return text
    # Find the answer block at the end.
    ans_idx = text.rfind("\nAnswer:")
    if ans_idx == -1:
        return text[:max_len]
    tail = text[ans_idx:]  # includes Answer: ... <EOT>
    head_budget = max_len - len(tail)
    if head_budget <= 0:
        return tail[-max_len:]
    # Keep head_budget chars from the start, then append the tail.
    return text[:head_budget] + tail


def _math_question(rng: random.Random, use_json_tools: bool = True):
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
    templates = [
        f"What is {a} {op} {b}?",
        f"Calculate {a} {op} {b}.",
        f"Compute {a} {op} {b}.",
        f"Tell me the result of {a} {op} {b}.",
    ]
    question = rng.choice(templates)
    steps = [("I need to use the calculator.", "calc", f"{a} {op} {b}", str(ans))]
    return _format_trace(question, steps, str(ans), use_json_tools=use_json_tools)


def _memory_question(rng: random.Random, memory: dict, use_json_tools: bool = True):
    key = rng.choice(list(memory.keys()))
    templates = [
        f"What is the {key}?",
        f"Retrieve the {key}.",
        f"Look up the {key}.",
        f"Tell me the value of {key}.",
    ]
    question = rng.choice(templates)
    steps = [("I should search memory for this.", "search_memory", key, memory[key])]
    return _format_trace(question, steps, memory[key], use_json_tools=use_json_tools)


def _date_question(rng: random.Random, use_json_tools: bool = True):
    now = datetime.datetime.now().isoformat()
    templates = [
        "What is the current date and time?",
        "What time is it now?",
        "Return the current timestamp.",
    ]
    question = rng.choice(templates)
    steps = [("I need the current time.", "now", "", now)]
    return _format_trace(question, steps, now, use_json_tools=use_json_tools)


# Fake web search knowledge base mirrors agent/tools.py.
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


def _web_search_question(rng: random.Random, use_json_tools: bool = True):
    key = rng.choice(list(_WEB_KB.keys()))
    templates = [
        f"What is the {key}?",
        f"Look up the {key}.",
        f"Search the web for the {key}.",
        f"Find the {key} online.",
    ]
    question = rng.choice(templates)
    answer = _WEB_KB[key]
    steps = [
        ("I do not know this from memory, so I should search the web.",
         "web_search", key, answer),
    ]
    return _format_trace(question, steps, answer, use_json_tools=use_json_tools)


def _web_then_math(rng: random.Random, use_json_tools: bool = True):
    key = rng.choice(list(_WEB_KB.keys()))
    value = _WEB_KB[key]
    # Use numeric part if possible, otherwise length of string answer.
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
    steps = [
        ("I need to retrieve the web fact first.", "web_search", key, value),
        ("Now I can combine it with arithmetic.", "calc", expr, str(ans)),
    ]
    return _format_trace(question, steps, str(ans), use_json_tools=use_json_tools)


def _multi_hop_question(rng: random.Random, memory: dict, use_json_tools: bool = True):
    key = rng.choice(list(memory.keys()))
    a = rng.randint(2, 50)
    b = rng.randint(2, 50)
    raw = memory[key]
    length = len(raw) if isinstance(raw, str) else 0
    op = rng.choice(["+", "*"])
    if op == "+":
        ans = a * b + length
        expr = f"{a} * {b} + {length}"
        op_word = "add"
    else:
        ans = a * b * length
        expr = f"{a} * {b} * {length}"
        op_word = "multiply by"
    templates = [
        f"Multiply {a} and {b}, then {op_word} the length of the {key}.",
        f"Compute {a} times {b} and {op_word} the number of characters in the {key}.",
    ]
    question = rng.choice(templates)
    steps = [
        ("First I will multiply.", "calc", f"{a} * {b}", str(a * b)),
        ("Then I need the length of the stored value.", "search_memory", key, raw),
        ("Now combine the results.", "calc", expr, str(ans)),
    ]
    return _format_trace(question, steps, str(ans), use_json_tools=use_json_tools)


def _memory_then_math(rng: random.Random, memory: dict, use_json_tools: bool = True):
    key = rng.choice(list(memory.keys()))
    raw = memory[key]
    try:
        base = float(raw)
    except Exception:
        base = len(raw)
    delta = rng.randint(1, 100)
    ans = base + delta
    templates = [
        f"Add {delta} to the {key}.",
        f"What is the {key} plus {delta}?",
    ]
    question = rng.choice(templates)
    steps = [
        ("First retrieve the stored value.", "search_memory", key, raw),
        ("Then add the requested amount.", "calc", f"{base} + {delta}", str(ans)),
    ]
    return _format_trace(question, steps, str(ans), use_json_tools=use_json_tools)


def generate_simple_agent_dataset(num_samples: int = 100000, max_len: int = 512, seed: int = 42, use_json_tools: bool = True):
    """Single-step ReAct traces only (math, memory, date, web).

    Each trace is guaranteed short enough that truncation is rarely needed.
    """
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
            ["math", "memory", "date", "web"],
            weights=[35, 30, 20, 15],
            k=1,
        )[0]
        if kind == "math":
            raw = _math_question(rng, use_json_tools=use_json_tools)
        elif kind == "memory":
            raw = _memory_question(rng, memory, use_json_tools=use_json_tools)
        elif kind == "date":
            raw = _date_question(rng, use_json_tools=use_json_tools)
        else:
            raw = _web_search_question(rng, use_json_tools=use_json_tools)
        samples.append(_truncate_trace(raw, max_len=max_len))
    return samples


def generate_agent_dataset(num_samples: int = 100000, max_len: int = 512, seed: int = 42, use_json_tools: bool = True):
    """Full agent dataset with single-step + multi-hop traces, including web search."""
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
            ["math", "memory", "date", "multi", "memory_math", "web", "web_math"],
            weights=[20, 15, 10, 15, 15, 15, 10],
            k=1,
        )[0]
        if kind == "math":
            raw = _math_question(rng, use_json_tools=use_json_tools)
        elif kind == "memory":
            raw = _memory_question(rng, memory, use_json_tools=use_json_tools)
        elif kind == "date":
            raw = _date_question(rng, use_json_tools=use_json_tools)
        elif kind == "multi":
            raw = _multi_hop_question(rng, memory, use_json_tools=use_json_tools)
        elif kind == "memory_math":
            raw = _memory_then_math(rng, memory, use_json_tools=use_json_tools)
        elif kind == "web":
            raw = _web_search_question(rng, use_json_tools=use_json_tools)
        else:
            raw = _web_then_math(rng, use_json_tools=use_json_tools)
        samples.append(_truncate_trace(raw, max_len=max_len))
    return samples
