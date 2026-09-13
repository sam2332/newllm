"""Sandboxed shell and Python execution for the agent, via Docker.

A model this small emits near-random bytes when it is unsure, so the isolation
here is load-bearing rather than decorative. Every execution runs in a fresh
throwaway container with:

    --network none          no network access at all
    --read-only             immutable root filesystem
    --tmpfs /tmp            small writable scratch area, noexec, capped
    --memory / --cpus       resource ceilings
    --pids-limit            no fork bombs
    --cap-drop ALL          no Linux capabilities
    --security-opt no-new-privileges
    --user 65534:65534      unprivileged (nobody)
    --rm                    container destroyed on exit
    timeout                 wall-clock kill

Nothing from the host filesystem is mounted. The agent cannot reach the repo,
the checkpoints, the Ollama instances, or the network from inside. Output is
truncated so a runaway loop cannot blow up the context window.

This is deliberately separate from ``repo_tools`` (read-only introspection):
reading the code and running code are different privileges and are kept apart.
"""

import shutil
import subprocess

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 20
MAX_OUTPUT = 2000

# Hard ceilings applied to every run regardless of caller.
_CONTAINER_ARGS = [
    "--rm",
    "--network", "none",
    "--read-only",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
    "--memory", "512m",
    "--memory-swap", "512m",
    "--cpus", "1.0",
    "--pids-limit", "128",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
    "--workdir", "/tmp",
]


class SandboxUnavailable(RuntimeError):
    pass


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=15,
                       check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            OSError):
        return False


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} bytes total)"


def _run(argv: list, timeout: int) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"

    out = _truncate(proc.stdout)
    err = _truncate(proc.stderr, 600)
    if proc.returncode != 0:
        parts = [f"exit code {proc.returncode}"]
        if out:
            parts.append(out)
        if err:
            parts.append(err)
        return "\n".join(parts)
    if out and err:
        return f"{out}\n[stderr]\n{err}"
    return out or err or "(no output)"


def run_bash(command: str, timeout: int = DEFAULT_TIMEOUT,
             image: str = DEFAULT_IMAGE) -> str:
    """Execute a shell command inside a locked-down throwaway container."""
    command = (command or "").strip()
    if not command:
        return "ERROR: empty command"
    if not docker_available():
        return "ERROR: docker is not available on this host"
    timeout = max(1, min(int(timeout), 60))
    argv = (["docker", "run"] + _CONTAINER_ARGS + [image, "timeout",
            str(timeout), "bash", "-lc", command])
    return _run(argv, timeout)


def run_python(code: str, timeout: int = DEFAULT_TIMEOUT,
               image: str = DEFAULT_IMAGE) -> str:
    """Execute a Python snippet inside the same locked-down container."""
    code = (code or "").strip()
    if not code:
        return "ERROR: empty code"
    if not docker_available():
        return "ERROR: docker is not available on this host"
    timeout = max(1, min(int(timeout), 60))
    argv = (["docker", "run", "-i"] + _CONTAINER_ARGS + [image, "timeout",
            str(timeout), "python", "-"])
    try:
        proc = subprocess.run(argv, input=code, capture_output=True, text=True,
                              timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    out, err = _truncate(proc.stdout), _truncate(proc.stderr, 600)
    if proc.returncode != 0:
        return "\n".join(p for p in
                         [f"exit code {proc.returncode}", out, err] if p)
    return out or err or "(no output)"


SANDBOX_TOOL_SPECS = {
    "run_bash": {
        "description": ("Run a shell command in an isolated sandbox with no "
                        "network and no access to the host filesystem."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "Shell command to execute."}},
            "required": ["command"]},
        "execute": lambda args: run_bash(args.get("command", "")),
    },
    "run_python": {
        "description": ("Run a Python snippet in an isolated sandbox with no "
                        "network and no access to the host filesystem."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string",
                     "description": "Python source to execute."}},
            "required": ["code"]},
        "execute": lambda args: run_python(args.get("code", "")),
    },
}


def attach_sandbox_tools(toolbox):
    """Add sandboxed execution tools to an existing Toolbox."""
    toolbox.tools.update(SANDBOX_TOOL_SPECS)
    return toolbox
