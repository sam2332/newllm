"""Convert a Hugging Face tool-calling / instruct dataset into our traces.

Every generator here writes the legacy tagged form (``<user>``,
``<assistant>{json}``, ``<tool name=X>``) and the dataset builder renders it
to ChatML, so an outside dataset only has to reach that form to get the rest
of the pipeline free: tool-name and parameter randomization, persona and
format-rule decoration, length bucketing, the lot.

The field names differ per dataset, so the mapping is configurable and the
common layouts are recognised automatically:

* ``messages`` / ``conversations`` - a list of turns, each with a role
  (``role``/``from``) and text (``content``/``value``), roles named
  ``user``/``human``, ``assistant``/``gpt``, ``tool``/``function``/
  ``observation``, ``system``;
* ``tools`` - a JSON string or list of function schemas;
* OpenAI-style ``tool_calls`` on an assistant turn, or a ``<tool_call>`` /
  ``{"name":..., "arguments":...}`` blob inside its content.

A thought is required by our assistant format and most datasets have none,
so one is synthesised from the action itself rather than invented: naming
the tool it is about to call is a description of the trace, not a claim
about the world.

    .venv/bin/python scripts/import_hf_dataset.py \\
        --dataset Salesforce/xlam-function-calling-60k --limit 20000 \\
        --out data/hf_traces.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chat_dataset import serialize_chat

ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
    "model": "assistant", "chatgpt": "assistant",
    "tool": "tool", "function": "tool", "observation": "tool",
    "function_response": "tool", "tool_response": "tool",
    "system": "system",
}
TURN_KEYS = ("messages", "conversations", "conversation", "chat", "dialog")
TEXT_KEYS = ("content", "value", "text")
ROLE_KEYS = ("role", "from", "speaker")

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _first(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_list(value):
    """Tools may arrive as a list, a JSON string, or a JSON-lines string."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            got = json.loads(text)
            return got if isinstance(got, list) else [got]
        except json.JSONDecodeError:
            out = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return out
    return [value]


def _calls_from(msg: dict, content: str):
    """OpenAI tool_calls, a <tool_call> tag, or a bare {"name","arguments"}."""
    calls = []
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function", c) if isinstance(c, dict) else {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if fn.get("name"):
            calls.append({"name": fn["name"], "arguments": args or {}})
    if calls:
        return calls, content
    for m in _TOOL_CALL_TAG.finditer(content or ""):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if obj.get("name"):
            calls.append({"name": obj["name"],
                          "arguments": obj.get("arguments") or obj.get("parameters") or {}})
    if calls:
        return calls, _TOOL_CALL_TAG.sub("", content or "").strip()
    stripped = (content or "").strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return calls, content
        for one in (obj if isinstance(obj, list) else [obj]):
            if isinstance(one, dict) and one.get("name") and (
                    "arguments" in one or "parameters" in one):
                calls.append({"name": one["name"],
                              "arguments": one.get("arguments")
                              or one.get("parameters") or {}})
        if calls:
            return calls, ""
    return calls, content


def _thought(call=None, answered=False):
    if call:
        return f"I'll call {call['name']} to get this."
    return "I have what I need to answer." if answered else "Answering directly."


def row_to_trace(row: dict, turn_key=None, tools_key="tools"):
    """One dataset row -> one legacy tagged trace, or None if unusable."""
    turns = _first(row, (turn_key,) if turn_key else TURN_KEYS)
    tools = _as_list(row.get(tools_key))
    messages = []

    if turns is None:
        # Flat single-turn rows: query/answer under any of several names.
        q = _first(row, ("query", "question", "instruction", "input", "prompt"))
        a = _first(row, ("answer", "output", "response", "completion"))
        if not q:
            return None, tools
        turns = [{"role": "user", "content": q}]
        if a is not None:
            turns.append({"role": "assistant", "content": a})

    for msg in turns:
        if not isinstance(msg, dict):
            continue
        role = ROLE_MAP.get(str(_first(msg, ROLE_KEYS) or "").lower())
        content = _first(msg, TEXT_KEYS)
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content)
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "tool":
            name = msg.get("name") or msg.get("tool_name") or "tool"
            messages.append({"role": "tool", "name": name, "content": content})
        elif role == "assistant":
            calls, text = _calls_from(msg, content)
            if calls:
                for call in calls:
                    messages.append({"role": "assistant", "content": json.dumps(
                        {"thought": _thought(call), "tool_call": call},
                        separators=(",", ":"))})
            if text.strip() or not calls:
                answered = any(m["role"] == "tool" for m in messages)
                messages.append({"role": "assistant", "content": json.dumps(
                    {"thought": _thought(answered=answered),
                     "response": text.strip()}, separators=(",", ":"))})
    if not messages or not any(m["role"] == "user" for m in messages):
        return None, tools
    if messages[-1]["role"] != "assistant":
        return None, tools
    system = None
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[1:]
    if not messages:
        return None, tools
    return serialize_chat(messages, system=system), tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id or a local path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--turn-key", default=None,
                    help="field holding the turns; auto-detected by default")
    ap.add_argument("--tools-key", default="tools")
    ap.add_argument("--out", default="data/hf_traces.json")
    ap.add_argument("--peek", action="store_true",
                    help="print the first row's fields and exit, so the keys "
                         "can be checked before importing anything")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets  (into .venv) to use this script")

    ds = load_dataset(args.dataset, args.config, split=args.split,
                      streaming=True)
    if args.peek:
        for row in ds:
            print(json.dumps({k: str(v)[:300] for k, v in row.items()}, indent=2))
            break
        return

    traces, tools_seen, skipped = [], 0, 0
    for i, row in enumerate(ds):
        if len(traces) >= args.limit:
            break
        try:
            trace, tools = row_to_trace(row, args.turn_key, args.tools_key)
        except Exception:                                   # noqa: BLE001
            skipped += 1
            continue
        if not trace:
            skipped += 1
            continue
        traces.append({"trace": trace, "tools": tools})
        tools_seen += bool(tools)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(traces, fh)
    chars = sum(len(t["trace"]) for t in traces)
    print(f"{len(traces):,} traces ({tools_seen:,} with tool schemas), "
          f"{skipped:,} skipped, {chars/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
