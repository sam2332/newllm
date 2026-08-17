"""Simple tools for the tiny agent.

All tools take a single string argument and return a string result.
"""

import datetime
import re
import operator


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
    except Exception as e:
        return f"ERROR: {e}"


class Toolbox:
    """A tiny set of tools the agent can call."""

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
            "calc": lambda arg: _safe_eval(arg),
            "now": lambda _: datetime.datetime.now().isoformat(),
            "search_memory": lambda arg: self.memory.get(arg.strip(), "not found"),
            "finish": lambda _: "done",
        }

    def run(self, name: str, arg: str) -> str:
        name = name.strip()
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        return self.tools[name](arg)
