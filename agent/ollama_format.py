"""Bidirectional adapter between the internal agent protocol and Ollama's API.

Verified against a live Ollama instance (qwen3:30b-a3b-q8_0). Ollama speaks an
OpenAI-shaped chat API with two deviations that matter:

  * ``tool_calls`` is a LIST on the assistant message, each entry shaped
    ``{"id": ..., "function": {"index": N, "name": ..., "arguments": {...}}}``
  * ``function.arguments`` is a real JSON OBJECT, not a JSON-encoded string
    the way the OpenAI API delivers it.

  * tool results come back as ``{"role": "tool", "tool_name": ..., "content": ...}``

The internal protocol stays compact on purpose - a byte-level model pays for
every character, and ``{"thought":...,"tool_call":{...}}`` is far cheaper than
Ollama's envelope. This module converts between the two so the model can be
served behind an Ollama-compatible endpoint without changing how it is trained.

Mapping
-------
    internal                          ollama
    {"thought": T,                    {"role": "assistant",
     "tool_call": {"name": N,          "content": "",
                   "arguments": A}}    "tool_calls": [{"function":
                                          {"name": N, "arguments": A}}]}

    {"thought": T, "response": R}     {"role": "assistant", "content": R}

``thought`` has no Ollama equivalent. It is preserved through the optional
``thinking`` field, which Ollama uses for reasoning models, so no information
is lost in a round trip.
"""

import json

FINISH_TOOL = "finish"


# --------------------------------------------------------- internal -> ollama

def assistant_to_ollama(internal: dict, include_thinking: bool = True) -> dict:
    """Convert one internal assistant message to an Ollama chat message."""
    msg = {"role": "assistant", "content": ""}
    if include_thinking and isinstance(internal.get("thought"), str):
        msg["thinking"] = internal["thought"]

    if isinstance(internal.get("tool_call"), dict):
        call = internal["tool_call"]
        msg["tool_calls"] = [{
            "function": {
                "name": call.get("name", ""),
                # Ollama takes a real object here, not a JSON string.
                "arguments": call.get("arguments", {}),
            }
        }]
    else:
        msg["content"] = internal.get("response", "")
    return msg


def tool_result_to_ollama(name: str, content: str) -> dict:
    """Convert a tool observation to an Ollama tool message."""
    return {"role": "tool", "tool_name": name, "content": str(content)}


def toolbox_to_ollama_tools(toolbox) -> list:
    """Render a Toolbox as Ollama/OpenAI function schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in toolbox.tools.items()
    ]


# --------------------------------------------------------- ollama -> internal

def assistant_from_ollama(msg: dict) -> dict:
    """Convert an Ollama assistant message to the internal contract.

    Always returns a dict with a string ``thought`` and exactly one of
    ``tool_call`` or ``response``, which is what the internal validator
    requires.
    """
    thought = msg.get("thinking") or ""
    calls = msg.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        args = fn.get("arguments", {})
        # OpenAI-style servers may still hand back a JSON string here.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {
            "thought": thought or f"Calling {fn.get('name', '')}.",
            "tool_call": {"name": fn.get("name", ""), "arguments": args},
        }
    return {
        "thought": thought or "Answering directly.",
        "response": msg.get("content", "") or "",
    }


def messages_from_ollama(messages: list) -> list:
    """Convert an Ollama message list into internal-protocol message dicts."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content", "")})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "tool":
            out.append({"role": "tool",
                        "name": m.get("tool_name") or m.get("name", ""),
                        "content": m.get("content", "")})
        elif role == "assistant":
            internal = assistant_from_ollama(m)
            out.append({"role": "assistant",
                        "content": json.dumps(internal, separators=(",", ":"))})
    return out


def messages_to_ollama(messages: list) -> list:
    """Convert internal-protocol messages back to an Ollama message list."""
    out = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            out.append({"role": role, "content": m.get("content", "")})
        elif role == "tool":
            out.append(tool_result_to_ollama(m.get("name", ""),
                                             m.get("content", "")))
        elif role == "assistant":
            content = m.get("content", "")
            internal = json.loads(content) if isinstance(content, str) else content
            out.append(assistant_to_ollama(internal))
    return out


def run_result_to_ollama(result: dict) -> dict:
    """Render a ``run_agent`` result as a single Ollama /api/chat response."""
    return {
        "message": {
            "role": "assistant",
            "content": result.get("final_answer", ""),
            "thinking": result.get("thinking", ""),
        },
        "done": True,
        "done_reason": "stop",
    }
