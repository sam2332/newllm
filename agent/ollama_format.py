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

import hashlib
import json

FINISH_TOOL = "finish"


def canonical_args(arguments) -> str:
    """Stable key for a tool call's arguments (used for ids and caches)."""
    try:
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(arguments)


def tool_call_id(name: str, arguments) -> str:
    """Deterministic ``call_xxxxxxxx`` id, so tests can predict it."""
    digest = hashlib.sha1((name + canonical_args(arguments)).encode()).hexdigest()
    return "call_" + digest[:8]


# --------------------------------------------------------- internal -> ollama

def assistant_to_ollama(internal: dict, include_thinking: bool = True,
                        call_index: int = 0) -> dict:
    """Convert one internal assistant message to an Ollama chat message."""
    msg = {"role": "assistant", "content": ""}
    if include_thinking and isinstance(internal.get("thought"), str):
        msg["thinking"] = internal["thought"]

    if isinstance(internal.get("tool_call"), dict):
        call = internal["tool_call"]
        name = call.get("name", "")
        args = call.get("arguments", {})
        msg["tool_calls"] = [{
            "id": tool_call_id(name, args),
            "function": {
                "index": call_index,
                "name": name,
                # Ollama takes a real object here, not a JSON string.
                "arguments": args,
            }
        }]
    else:
        msg["content"] = internal.get("response", "")
    return msg


def tools_from_ollama(tools: list) -> list:
    """Client tools -> ``[{"name", "description", "parameters"}]`` entries."""
    out = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            out.append({"name": name,
                        "description": str(fn.get("description") or ""),
                        "parameters": fn.get("parameters")
                        if isinstance(fn.get("parameters"), dict)
                        else {"type": "object", "properties": {}}})
    return out


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

def _parse_args(args):
    # OpenAI-style servers may still hand back a JSON string here.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def assistants_from_ollama(msg: dict, thought_cache=None) -> list:
    """Convert an Ollama assistant message to one internal message per call.

    The model's protocol is one tool call per assistant turn, so a message
    carrying several ``tool_calls`` (from another model's history) becomes
    several internal turns. The ``thought`` is recovered in this order: the
    message's ``thinking``, then a cache of thoughts the server emitted for
    the same (name, arguments) - the official client echoes ``thinking``
    back, but many clients strip it - then a neutral default.
    """
    thought = msg.get("thinking") or ""
    calls = msg.get("tool_calls") or []
    if not calls:
        return [{"thought": thought or "Answering directly.",
                 "response": msg.get("content", "") or ""}]
    out = []
    for call in calls:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name", "") or ""
        args = _parse_args(fn.get("arguments", {}))
        cached = None
        if thought_cache is not None:
            cached = thought_cache.get((name, canonical_args(args)))
        out.append({"thought": thought or cached or f"Calling {name}.",
                    "tool_call": {"name": name, "arguments": args}})
    return out


def assistant_from_ollama(msg: dict) -> dict:
    """First internal message for an Ollama assistant message (see above)."""
    return assistants_from_ollama(msg)[0]


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
            for internal in assistants_from_ollama(m):
                out.append({"role": "assistant",
                            "content": json.dumps(internal,
                                                  separators=(",", ":"))})
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
