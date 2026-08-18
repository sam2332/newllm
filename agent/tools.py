"""Simple tools for the tiny agent.

Tools are declared with a JSON schema so the model can emit standard
function-call style requests:

    Action: {"tool": "calc", "args": {"expr": "12 + 8"}}
"""

import datetime
import re
import json


def _safe_eval(expr: str) -> str:
    """Evaluate simple arithmetic with + - * / ** and parentheses."""
    expr = expr.replace("^", "**")
    # allow only digits, spaces, and operators
    if not re.fullmatch(r"[0-9+\-*/()\.\s\*]*", expr):
        return "ERROR: invalid expression"
    try:
        # use ast parsing for a bit more safety than raw eval
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
    except SyntaxError as e:
        # common failure mode: the model emits two numbers with no operator
        # (e.g. "10 17" from token-level decoding confusion). Fall back to
        # evaluating each numeric token and returning the first valid one.
        nums = re.findall(r"-?\d+(?:\.\d+)?", expr)
        if nums:
            return nums[0]
        return f"ERROR: {e}"
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


def _web_search(query: str) -> str:
    """Simulate a web search by looking up a canned answer.

    The model must emit a web_search tool call to retrieve facts that are not
    in its parametric memory.
    """
    query = query.strip().lower().rstrip("?")
    # Try exact match first.
    if query in WEB_KB:
        return WEB_KB[query]
    # Substring match.
    for key, value in WEB_KB.items():
        if key in query or query in key:
            return value
    return "no results found"


class Toolbox:
    """A tiny set of tools the agent can call.

    Each tool has a name, description, and JSON Schema-style parameter list.
    The model emits JSON tool calls; `run_json` validates and executes them.
    """

    def __init__(self):
        self.memory = {
            "project": "newllm",
            "device": "cuda",
            "leader": "grug",
            "tribe": "cavepeople",
            "language": "Python",
            "status": "active",
            "version": "0.1",
        }
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
                "execute": lambda args: _web_search(args.get("query", "")),
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

    def run_json(self, call: dict) -> str:
        """Execute a JSON tool call of the form {'tool': str, 'args': dict}."""
        name = call.get("tool", call.get("name", "")).strip()
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        args = call.get("args", call.get("arguments", {}))
        if not isinstance(args, dict):
            return "ERROR: tool args must be a JSON object"
        spec = self.tools[name]
        required = spec["parameters"].get("required", [])
        missing = [p for p in required if p not in args]
        if missing:
            return f"ERROR: missing required parameters {missing}"
        try:
            return spec["execute"](args)
        except Exception as e:
            return f"ERROR: {e}"
