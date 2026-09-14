"""A third tool tier: files the model may WRITE, confined to a workspace.

``repo_tools`` reads the repository and runs nothing; ``sandbox_tools`` runs
code and cannot see the repository. Neither may write to disk the model can
later read, and that separation stays: this module writes only inside
``workspace/`` (or ``$NEWLLM_WORKSPACE``), never the repo, with the same
realpath containment ``repo_tools`` uses.

The workspace is what makes long projects possible. A 52-turn story does not
fit in any context window; chapters written to files scroll out of the
context and are read back when needed, so the filesystem is the model's
long-term memory. Observations are deliberately short ("wrote 2431 bytes to
ch03.md") so a write costs the context almost nothing.

``WorkspaceBase`` holds the tool specs and the exact observation strings;
``RealWorkspace`` (here) and ``VirtualWorkspace`` (``agent/virtual_workspace.py``)
implement the storage, so generated training traces and live serving produce
byte-identical observations from real code paths - the "teacher never
fabricates a tool result" invariant is kept because nothing is fabricated.
"""

import fnmatch
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.environ.get("NEWLLM_WORKSPACE", os.path.join(REPO_ROOT, "workspace"))

# ~3k tokens of a read reaches the context; longer files are paged with
# ``start``. Writes are capped so a looping model cannot fill the disk.
MAX_READ_CHARS = 12288
MAX_WRITE_CHARS = 200_000
MAX_FILES = 500


class WorkspaceError(Exception):
    pass


class WorkspaceBase:
    """Storage-agnostic tool behaviour. Subclasses implement _get/_put/_names."""

    # -- storage interface ---------------------------------------------------
    def _get(self, rel: str):            # -> str | None
        raise NotImplementedError

    def _put(self, rel: str, text: str):
        raise NotImplementedError

    def _names(self) -> list:            # -> sorted relative paths
        raise NotImplementedError

    # -- validation ----------------------------------------------------------
    @staticmethod
    def _rel(path) -> str:
        p = str(path or "").strip().replace("\\", "/")
        if not p:
            raise WorkspaceError("path is required")
        if p.startswith("/") or p.startswith("~") or ".." in p.split("/"):
            raise WorkspaceError("path must be relative to the workspace and may not contain ..")
        parts = [x for x in p.split("/") if x not in ("", ".")]
        if not parts:
            raise WorkspaceError("path is required")
        return "/".join(parts)

    # -- tools ---------------------------------------------------------------
    def write_file(self, path, content) -> str:
        rel = self._rel(path)
        text = str(content if content is not None else "")
        if len(text) > MAX_WRITE_CHARS:
            raise WorkspaceError(f"content exceeds {MAX_WRITE_CHARS} characters")
        if self._get(rel) is None and len(self._names()) >= MAX_FILES:
            raise WorkspaceError(f"workspace already holds {MAX_FILES} files")
        self._put(rel, text)
        return f"wrote {len(text.encode('utf-8'))} bytes to {rel}"

    def append_file(self, path, content) -> str:
        rel = self._rel(path)
        old = self._get(rel) or ""
        text = str(content if content is not None else "")
        if len(old) + len(text) > MAX_WRITE_CHARS:
            raise WorkspaceError(f"file would exceed {MAX_WRITE_CHARS} characters")
        self._put(rel, old + text)
        return (f"appended {len(text.encode('utf-8'))} bytes to {rel} "
                f"(now {len((old + text).encode('utf-8'))} bytes)")

    def read_file(self, path, start=0, chars=MAX_READ_CHARS) -> str:
        rel = self._rel(path)
        text = self._get(rel)
        if text is None:
            raise WorkspaceError(f"no such file: {rel}")
        start = max(0, int(start or 0))
        chars = max(1, min(int(chars or MAX_READ_CHARS), MAX_READ_CHARS))
        piece = text[start:start + chars]
        if start + chars < len(text):
            piece += (f"\n... (truncated: {len(text) - start - chars} more characters; "
                      f"read again with start={start + chars})")
        return piece

    def list_files(self, pattern="*") -> str:
        names = [n for n in self._names() if fnmatch.fnmatch(n, str(pattern or "*"))]
        if not names:
            return "(no files)"
        return "\n".join(f"{n}  {len((self._get(n) or '').encode('utf-8'))} bytes"
                         for n in names)

    # -- toolbox wiring ------------------------------------------------------
    def specs(self) -> dict:
        def run(fn):
            def execute(args):
                try:
                    return fn(**{k: v for k, v in (args or {}).items()})
                except WorkspaceError as exc:
                    return f"error: {exc}"
                except TypeError as exc:
                    return f"error: bad arguments ({exc})"
            return execute
        s = {"type": "string"}
        return {
            "write_file": {
                "description": "Create or overwrite a text file in the workspace.",
                "parameters": {"type": "object",
                               "properties": {"path": {**s, "description": "relative file path"},
                                              "content": {**s, "description": "full file contents"}},
                               "required": ["path", "content"]},
                "execute": run(self.write_file)},
            "append_file": {
                "description": "Append text to the end of a workspace file (creates it if missing).",
                "parameters": {"type": "object",
                               "properties": {"path": {**s, "description": "relative file path"},
                                              "content": {**s, "description": "text to append"}},
                               "required": ["path", "content"]},
                "execute": run(self.append_file)},
            "read_file": {
                "description": "Read a workspace file; long files are returned in pages.",
                "parameters": {"type": "object",
                               "properties": {"path": {**s, "description": "relative file path"},
                                              "start": {"type": "integer",
                                                        "description": "character offset to start from"}},
                               "required": ["path"]},
                "execute": run(self.read_file)},
            "list_files": {
                "description": "List workspace files with their sizes.",
                "parameters": {"type": "object",
                               "properties": {"pattern": {**s, "description": "glob such as *.md"}},
                               "required": []},
                "execute": run(self.list_files)},
        }


class RealWorkspace(WorkspaceBase):
    """Files under one directory; every path is realpath-checked against it."""

    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)

    def _abs(self, rel: str) -> str:
        candidate = os.path.realpath(os.path.join(self.root, rel))
        if not candidate.startswith(self.root + os.sep):
            raise WorkspaceError("path escapes the workspace")
        return candidate

    def _get(self, rel):
        p = self._abs(rel)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def _put(self, rel, text):
        p = self._abs(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)

    def _names(self):
        out = []
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                if f.endswith(".tmp"):
                    continue
                out.append(os.path.relpath(os.path.join(dirpath, f), self.root)
                           .replace(os.sep, "/"))
        return sorted(out)


def attach_workspace_tools(toolbox, workspace: WorkspaceBase = None):
    """Add the four workspace tools to a Toolbox (mutates and returns it)."""
    ws = workspace or RealWorkspace()
    toolbox.tools.update(ws.specs())
    toolbox.workspace = ws
    return toolbox
