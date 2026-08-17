"""Generate synthetic agent training data (clean, non-caveman).

Each sample is a ReAct-style trace with explicit internal thinking:

Question: ... BEGIN_THINK Thought: ... Action: ... Observation: ... END_THINK Answer: ... <EOT>
"""

import random
import datetime


EOT = "\x03"
THINK_START = "BEGIN_THINK"
THINK_END = "END_THINK"


def _format_trace(question: str, steps: list, answer: str) -> str:
    """steps is list of (thought, tool_name, tool_arg, observation)."""
    lines = [f"Question: {question}", THINK_START]
    for thought, tool, arg, obs in steps:
        lines.append(f"Thought: {thought}")
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


def _math_question(rng: random.Random):
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
    return _format_trace(question, steps, str(ans))


def _memory_question(rng: random.Random, memory: dict):
    key = rng.choice(list(memory.keys()))
    templates = [
        f"What is the {key}?",
        f"Retrieve the {key}.",
        f"Look up the {key}.",
        f"Tell me the value of {key}.",
    ]
    question = rng.choice(templates)
    steps = [("I should search memory for this.", "search_memory", key, memory[key])]
    return _format_trace(question, steps, memory[key])


def _date_question(rng: random.Random):
    now = datetime.datetime.now().isoformat()
    templates = [
        "What is the current date and time?",
        "What time is it now?",
        "Return the current timestamp.",
    ]
    question = rng.choice(templates)
    steps = [("I need the current time.", "now", "", now)]
    return _format_trace(question, steps, now)


def _multi_hop_question(rng: random.Random, memory: dict):
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
    steps = [
        ("First I will multiply.", "calc", f"{a} * {b}", str(a * b)),
        ("Then I need the length of the stored value.", "search_memory", key, raw),
        ("Now combine the results.", "calc", expr, str(ans)),
    ]
    return _format_trace(question, steps, str(ans))


def _memory_then_math(rng: random.Random, memory: dict):
    key = rng.choice(list(memory.keys()))
    raw = memory[key]
    try:
        base = int(raw)
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
    return _format_trace(question, steps, str(ans))


def generate_simple_agent_dataset(num_samples: int = 100000, max_len: int = 512, seed: int = 42):
    """Single-step ReAct traces only (math, memory, date).

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
        kind = rng.choices(["math", "memory", "date"], weights=[40, 35, 25], k=1)[0]
        if kind == "math":
            raw = _math_question(rng)
        elif kind == "memory":
            raw = _memory_question(rng, memory)
        else:
            raw = _date_question(rng)
        samples.append(_truncate_trace(raw, max_len=max_len))
    return samples


def generate_agent_dataset(num_samples: int = 100000, max_len: int = 512, seed: int = 42):
    """Full agent dataset with single-step + multi-hop traces."""
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
            ["math", "memory", "date", "multi", "memory_math"],
            weights=[30, 20, 10, 20, 20],
            k=1,
        )[0]
        if kind == "math":
            raw = _math_question(rng)
        elif kind == "memory":
            raw = _memory_question(rng, memory)
        elif kind == "date":
            raw = _date_question(rng)
        elif kind == "multi":
            raw = _multi_hop_question(rng, memory)
        else:
            raw = _memory_then_math(rng, memory)
        samples.append(_truncate_trace(raw, max_len=max_len))
    return samples
