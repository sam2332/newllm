"""Serve the trained agent behind an Ollama-compatible /api/chat endpoint.

This is the thing that makes "will it work once implemented into a model"
answerable by test rather than by argument: any client that already speaks to
Ollama can point at this port and get the same message shapes back.

Implements:
    GET  /api/tags     - model listing
    POST /api/chat     - chat with optional tools, returns Ollama-shaped JSON
    POST /api/show     - capability advertisement

Run:
    .venv/bin/python scripts/serve_ollama_compat.py \
        --checkpoint checkpoints_v2_M/agent_best.pt --port 11500
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/home/lmeadows/llm")
import torch

from agent.chat import load_checkpoint, load_tokenizer_for
from agent.agent_loop import run_agent
from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools
from agent.ollama_format import (assistant_to_ollama, toolbox_to_ollama_tools,
                                 messages_from_ollama)

STATE = {"model": None, "toolbox": None, "tokenizer": None,
         "device": "cuda", "name": "newllm-agent"}
_LOCK = threading.Lock()


def _extract_last_user(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _prior_turns(messages: list) -> list:
    """Render completed earlier turns as tagged blocks for the chat context."""
    internal = messages_from_ollama(messages[:-1]) if len(messages) > 1 else []
    blocks = []
    for m in internal:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            blocks.append(f"<tool name={m['name']}>{m['content']}</tool>")
        else:
            blocks.append(f"<{m['role']}>{m['content']}</{m['role']}>")
    return blocks


def handle_chat(payload: dict) -> dict:
    messages = payload.get("messages", [])
    question = _extract_last_user(messages)
    system = next((m.get("content") for m in messages
                   if m.get("role") == "system"), None)
    conversation = _prior_turns(messages)

    with _LOCK:
        result = run_agent(
            STATE["model"], question, STATE["toolbox"],
            device=STATE["device"], greedy=True, tokenizer=STATE["tokenizer"],
            conversation=conversation or None, system=system,
            max_steps=5, max_new=300,
        )

    # The loop already executed the tools, so the reply is a finished answer.
    # Tool activity is surfaced so a client can see what ran.
    message = {
        "role": "assistant",
        "content": result.get("final_answer", ""),
        "thinking": result.get("thinking", ""),
    }
    return {
        "model": payload.get("model", STATE["name"]),
        "created_at": "1970-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": "stop",
        "tool_activity": [
            {"function": {"name": s["tool"], "arguments": json.loads(s["arg"])},
             "result": s["result"]}
            for s in result.get("steps", [])
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            n = STATE["model"].count_parameters()
            return self._send({"models": [{
                "name": STATE["name"], "model": STATE["name"],
                "size": n * 4,
                "details": {"family": "newllm", "parameter_size": f"{n/1e6:.1f}M",
                            "quantization_level": "F32"},
                "capabilities": ["completion", "tools"],
            }]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "invalid JSON"}, 400)
        if self.path.startswith("/api/chat"):
            try:
                return self._send(handle_chat(payload))
            except Exception as exc:                       # noqa: BLE001
                return self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if self.path.startswith("/api/show"):
            return self._send({"capabilities": ["completion", "tools"],
                               "details": {"family": "newllm"}})
        self._send({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_v2_M/agent_best.pt")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo-tools", action="store_true",
                    help="also expose read-only self-inspection tools")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    STATE["model"] = load_checkpoint(args.checkpoint, device=device)
    tb = Toolbox()
    if args.repo_tools:
        tb = attach_repo_tools(tb)
    STATE["toolbox"] = tb
    STATE["tokenizer"] = load_tokenizer_for(STATE["model"])
    print(f"tokenizer vocab: {STATE['tokenizer'].vocab_size}")

    print(f"tools advertised: {[t['function']['name'] for t in toolbox_to_ollama_tools(tb)]}")
    print(f"serving Ollama-compatible API on http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
