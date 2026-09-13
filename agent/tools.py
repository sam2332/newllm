"""Simple tools for the tiny agent.

Tools are declared with a JSON schema so the model can emit standard
function-call style requests:

    {"tool": "calc", "args": {"expr": "12 + 8"}}
"""

import datetime
import difflib
import re
import json


def _safe_eval(expr: str) -> str:
    """Evaluate simple arithmetic with + - * / ** and parentheses."""
    expr = expr.replace("^", "**")
    if not re.fullmatch(r"[0-9+\-*/()\.\s\*]*", expr):
        return "ERROR: invalid expression"
    try:
        import ast
        node = ast.parse(expr, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return "ERROR: invalid expression"
        value = eval(compile(node, "<string>", "eval"),
                     {"__builtins__": {}}, {})
        return str(value)
    except Exception as e:
        return f"ERROR: {e}"


# Fake web knowledge base — the model must use web_search to retrieve these.
WEB_KB = {
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


def _web_search(query: str, kb: dict = None) -> str:
    """Simulate a web search by looking up a canned answer.

    The model must emit a web_search tool call to retrieve facts that are not
    in its parametric memory. ``kb`` allows a larger generated fact set to be
    injected for training without editing this module.
    """
    kb = WEB_KB if kb is None else kb
    query = query.strip().lower().rstrip("?")
    # Try exact match first.
    if query in kb:
        return kb[query]
    # Substring match.
    for key, value in kb.items():
        if key in query or query in key:
            return value
    match = difflib.get_close_matches(query, kb.keys(), n=1, cutoff=0.78)
    if match:
        return kb[match[0]]
    return "no results found"


class Toolbox:
    """A tiny set of tools the agent can call.

    Each tool has a name, description, and JSON Schema-style parameter list.
    The model emits JSON tool calls; `run_json` validates and executes them.
    """

    DEFAULT_MEMORY = {
        "project": "newllm",
        "device": "cuda",
        "leader": "grug",
        "tribe": "cavepeople",
        "language": "Python",
        "status": "active",
        "version": "0.1",
    }

    def __init__(self, memory: dict = None, web_kb: dict = None):
        self.memory = dict(memory) if memory else dict(self.DEFAULT_MEMORY)
        self.web_kb = dict(web_kb) if web_kb else dict(WEB_KB)
        self.tools = {
            "calc": {
                "description": "Evaluate a simple arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {
                            "type": "string",
                            "description": "Arithmetic expression with +, -, *, /, **, parentheses.",
                        }
                    },
                    "required": ["expr"],
                },
                "execute": lambda args: _safe_eval(args.get("expr", "")),
            },
            "now": {
                "description": "Return the current date and time as an ISO timestamp.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                "execute": lambda _: datetime.datetime.now().isoformat(),
            },
            "search_memory": {
                "description": "Look up a value in the agent's memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The memory key to retrieve.",
                        }
                    },
                    "required": ["key"],
                },
                "execute": lambda args: self.memory.get(
                    args.get("key", "").strip(), "not found"),
            },
            "web_search": {
                "description": "Search the web for a short factual answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A short search query.",
                        }
                    },
                    "required": ["query"],
                },
                "execute": lambda args: _web_search(args.get("query", ""),
                                                    self.web_kb),
            },
            "finish": {
                "description": "Signal that the task is complete.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                "execute": lambda _: "done",
            },
        }

    @property
    def schemas(self) -> list:
        """Return OpenAI-style function schemas for prompting."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for name, spec in self.tools.items()
        ]

    def run(self, name: str, arg: str) -> str:
        """Legacy string-argument entrypoint (kept for compatibility)."""
        name = name.strip()
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        return self.tools[name]["execute"](
            {"expr": arg} if name == "calc"
            else {"key": arg} if name == "search_memory"
            else {"query": arg} if name == "web_search"
            else {}
        )

    def _normalize_tool_name(self, name: str) -> str:
        name = name.strip()
        if name in self.tools:
            return name
        match = difflib.get_close_matches(name, self.tools.keys(), n=1, cutoff=0.82)
        return match[0] if match else name

    def _normalize_memory_key(self, key: str) -> str:
        key = key.strip()
        if key in self.memory:
            return key
        match = difflib.get_close_matches(key, self.memory.keys(), n=1, cutoff=0.78)
        return match[0] if match else key

    def run_json(self, call: dict) -> str:
        """Execute a JSON tool call of the form {'tool': str, 'args': dict}."""
        name = self._normalize_tool_name(call.get("tool", call.get("name", "")))
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        args = call.get("args", call.get("arguments", {}))
        if not isinstance(args, dict):
            return "ERROR: tool args must be a JSON object"
        if name == "search_memory" and isinstance(args.get("key"), str):
            args = dict(args)
            args["key"] = self._normalize_memory_key(args["key"])
        spec = self.tools[name]
        required = spec["parameters"].get("required", [])
        missing = [p for p in required if p not in args]
        if missing:
            return f"ERROR: missing required parameters {missing}"
        try:
            return spec["execute"](args)
        except Exception as e:
            return f"ERROR: {e}"
