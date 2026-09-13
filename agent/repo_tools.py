"""Tools that let the agent inspect its own source code.

The model does not need to understand code in its weights - it needs to learn
*which tool to call and with what argument*, which is the same skill it already
has for calc/web_search. Actual code comprehension lives in the tool (and, for
hard questions, in a larger teacher model), not in the 25M-parameter router.

Every tool here is READ-ONLY and confined to the repository root. Nothing in
this module can modify, execute, or delete anything - self-inspection and
self-modification are deliberately separate concerns, and only the first is
safe to hand to a model that cannot yet be trusted to reason about code.
"""

import ast
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 4000
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "results",
             "data", "checkpoints"}


def _safe_path(relative: str) -> str:
    """Resolve a repo-relative path, refusing anything outside the repo."""
    candidate = os.path.realpath(os.path.join(REPO_ROOT, relative.strip()))
    root = os.path.realpath(REPO_ROOT)
    if not candidate.startswith(root + os.sep) and candidate != root:
        raise ValueError("path escapes the repository root")
    return candidate


def list_files(pattern: str = "") -> str:
    """List tracked source files, optionally filtered by substring."""
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not name.endswith((".py", ".md", ".txt", ".ps1")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), REPO_ROOT)
            if not pattern or pattern.lower() in rel.lower():
                out.append(rel)
    return "\n".join(sorted(out)[:120]) or "no matching files"


def read_file(path: str, start: int = 1, lines: int = 60) -> str:
    """Read a slice of a source file (1-indexed, line-limited)."""
    try:
        resolved = _safe_path(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not os.path.isfile(resolved):
        return f"ERROR: no such file '{path}'"
    with open(resolved, errors="replace") as fh:
        content = fh.readlines()
    start = max(1, int(start))
    chunk = content[start - 1:start - 1 + max(1, int(lines))]
    text = "".join(chunk)
    if len(text) > MAX_BYTES:
        text = text[:MAX_BYTES] + "\n... (truncated)"
    return text or "(empty range)"


def search_code(query: str, max_hits: int = 20) -> str:
    """Grep the repository for a literal string, returning path:line matches."""
    if not query.strip():
        return "ERROR: empty query"
    hits = []
    needle = query.lower()
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not name.endswith((".py", ".md", ".ps1")):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, REPO_ROOT)
            try:
                with open(full, errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if needle in line.lower():
                            hits.append(f"{rel}:{i}: {line.strip()[:110]}")
                            if len(hits) >= max_hits:
                                return "\n".join(hits)
            except OSError:
                continue
    return "\n".join(hits) or "no matches"


def describe_symbol(name: str) -> str:
    """Find a class or function by name and return its signature + docstring."""
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, REPO_ROOT)
            try:
                tree = ast.parse(open(full, errors="replace").read())
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)) and node.name == name:
                    doc = ast.get_docstring(node) or "(no docstring)"
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    if kind == "def":
                        argnames = [a.arg for a in node.args.args]
                        sig = f"{name}({', '.join(argnames)})"
                    else:
                        sig = name
                    return (f"{rel}:{node.lineno}\n{kind} {sig}\n\n"
                            f"{doc[:600]}")
    return f"no symbol named '{name}'"


def repo_stats(_: str = "") -> str:
    """Summarize the repository: file counts and total lines by directory."""
    counts = {}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(dirpath, REPO_ROOT)
            top = rel.split(os.sep)[0] if rel != "." else "(root)"
            try:
                n = sum(1 for _ in open(os.path.join(dirpath, name),
                                        errors="replace"))
            except OSError:
                n = 0
            entry = counts.setdefault(top, [0, 0])
            entry[0] += 1
            entry[1] += n
    return "\n".join(f"{k}: {v[0]} files, {v[1]} lines"
                     for k, v in sorted(counts.items()))


REPO_TOOL_SPECS = {
    "list_files": {
        "description": "List source files in the agent's own repository.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string",
                        "description": "Substring filter, e.g. 'attention'."}},
            "required": []},
        "execute": lambda args: list_files(args.get("pattern", "")),
    },
    "read_file": {
        "description": "Read lines from a file in the agent's own source code.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "start": {"type": "integer", "description": "First line (1-indexed)."},
            "lines": {"type": "integer", "description": "How many lines."}},
            "required": ["path"]},
        "execute": lambda args: read_file(args.get("path", ""),
                                          args.get("start", 1),
                                          args.get("lines", 60)),
    },
    "search_code": {
        "description": "Search the agent's own source code for a string.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Literal text to find."}},
            "required": ["query"]},
        "execute": lambda args: search_code(args.get("query", "")),
    },
    "describe_symbol": {
        "description": "Look up a class or function definition by name.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Class or function name."}},
            "required": ["name"]},
        "execute": lambda args: describe_symbol(args.get("name", "")),
    },
    "repo_stats": {
        "description": "Summarize the repository structure and size.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "execute": lambda args: repo_stats(),
    },
}


def attach_repo_tools(toolbox):
    """Add read-only self-inspection tools to an existing Toolbox."""
    toolbox.tools.update(REPO_TOOL_SPECS)
    return toolbox
