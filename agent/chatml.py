"""Render conversations in the Qwen3 chat template, with supervised spans.

The retrained model speaks the format Ollama and llama.cpp already parse:

    <|im_start|>system
    {system text}

    # Tools ... <tools> {json per line} </tools> ... <|im_end|>
    <|im_start|>user
    {content}<|im_end|>
    <|im_start|>assistant
    <think>
    {thought}
    </think>

    {content or <tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>}<|im_end|>
    <|im_start|>user
    <tool_response>
    {result}
    </tool_response><|im_end|>

``render(style="hf")`` reproduces ``data/templates/qwen3.jinja`` exactly
(``agent/test_chatml.py`` proves it against jinja2). The other styles vary
only the serialization of the tools JSON and the whitespace around the system
text, which is where Ollama's Go template and the HF Jinja legitimately
differ; training on a mix makes the model indifferent to the difference
instead of betting on one renderer's whitespace.

Two Qwen3 rules that are easy to miss and matter for training data:
  * Assistant turns *before* the last real user query are rendered without
    their ``<think>`` block - old reasoning is dropped from history.
  * Consecutive tool results are wrapped in ONE user turn.
"""

import json
import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
GEN_PROMPT = IM_START + "assistant\n"

TOOLS_HEADER = ("# Tools\n\nYou may call one or more functions to assist with the "
                "user query.\n\nYou are provided with function signatures within "
                "<tools></tools> XML tags:\n<tools>")
TOOLS_FOOTER = ("\n</tools>\n\nFor each function call, return a json object with "
                "function name and arguments within <tool_call></tool_call> XML "
                "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                "<args-json-object>}\n</tool_call>")

STYLES = ("hf", "compact", "ollama")


# ------------------------------------------------------------ serializers

def _hf_tojson(obj) -> str:
    """Jinja2's ``tojson``: sorted keys, default separators, HTML-safe."""
    s = json.dumps(obj, sort_keys=True)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e")
             .replace("&", "\\u0026").replace("'", "\\u0027"))


def _compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _go_json(obj) -> str:
    """Approximates Go's ``encoding/json``: compact, HTML-escaped, map keys
    sorted. Struct field order for tools is name, description, parameters."""
    if isinstance(obj, dict):
        s = "{" + ",".join(_go_json(k) + ":" + _go_json(v)
                           for k, v in obj.items()) + "}"
    elif isinstance(obj, list):
        s = "[" + ",".join(_go_json(v) for v in obj) + "]"
    else:
        s = json.dumps(obj)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _tool_line(tool: dict, style: str) -> str:
    if style == "hf":
        return _hf_tojson(tool)
    if style == "compact":
        return _compact(tool)
    fn = tool.get("function", tool)
    params = fn.get("parameters") or {}
    ordered_fn = {"name": fn.get("name", ""),
                  "description": fn.get("description", ""),
                  "parameters": {"type": params.get("type", "object"),
                                 "required": params.get("required", []),
                                 "properties": dict(sorted(
                                     (params.get("properties") or {}).items()))}}
    return '{"type": "function", "function": ' + _go_json(ordered_fn) + "}"


def _args_json(args, style: str) -> str:
    if isinstance(args, str):
        return args
    return _hf_tojson(args) if style == "hf" else _compact(args)


# ---------------------------------------------------------------- render

def _last_query_index(messages: list) -> int:
    last = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str) and not (
                c.startswith("<tool_response>") and c.endswith("</tool_response>")):
            return i
    return last


def _split_think(m: dict, content: str) -> tuple:
    reasoning = m.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = m.get("thinking") if isinstance(m.get("thinking"), str) else None
    if reasoning is None and "</think>" in content:
        reasoning = (content.split("</think>")[0].rstrip("\n")
                     .split("<think>")[-1].lstrip("\n"))
        content = content.split("</think>")[-1].lstrip("\n")
    return reasoning or "", content


def render(messages: list, tools: list = None, add_generation_prompt: bool = False,
           style: str = "hf", enable_thinking=None) -> tuple:
    """Return ``(text, spans)``; spans are ``(start, end)`` char ranges of the
    supervised assistant content, each ending just after ``<|im_end|>``."""
    assert style in STYLES, style
    msgs = list(messages)
    parts, spans = [], []
    pos = 0

    def emit(s: str):
        nonlocal pos
        parts.append(s)
        pos += len(s)

    system_text = (msgs[0].get("content") if msgs and msgs[0].get("role") == "system"
                   else None)
    if style == "ollama":
        # Ollama's Go template: a blank line after "system", and consecutive
        # tool results are NOT merged. Best-effort reading of the template;
        # Track C verifies it against the live server (OLLAMA_DEBUG).
        if tools:
            head = IM_START + "system\n"
            if system_text is not None:
                head += "\n" + system_text
            head += "\n\n" + TOOLS_HEADER
            for tool in tools:
                head += "\n" + _tool_line(tool, style)
            head += TOOLS_FOOTER + IM_END + "\n"
            emit(head)
        elif system_text is not None:
            emit(IM_START + "system\n\n" + system_text + IM_END + "\n")
    elif tools:
        head = IM_START + "system\n"
        if system_text is not None:
            head += system_text + "\n\n"
        head += TOOLS_HEADER
        for tool in tools:
            head += "\n" + _tool_line(tool, style)
        head += TOOLS_FOOTER + IM_END + "\n"
        emit(head)
    elif system_text is not None:
        emit(IM_START + "system\n" + system_text + IM_END + "\n")

    last_query = _last_query_index(msgs)
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        if role == "user" or (role == "system" and i > 0):
            emit(IM_START + role + "\n" + content + IM_END + "\n")
        elif role == "assistant":
            reasoning, content = _split_think(m, content)
            is_last = i == n - 1
            calls = m.get("tool_calls") or []
            if style == "ollama":
                body = ""
                if reasoning and (is_last or i > last_query):
                    body += "<think>" + reasoning + "</think>\n"
                if content:
                    body += content
                elif calls:
                    body += "<tool_call>\n" + "".join(
                        '{"name": "' + str(c.get("function", c).get("name", ""))
                        + '", "arguments": '
                        + _go_json(c.get("function", c).get("arguments", {})) + "}\n"
                        for c in calls) + "</tool_call>"
            else:
                if i > last_query and (is_last or reasoning):
                    body = ("<think>\n" + reasoning.strip("\n") + "\n</think>\n\n"
                            + content.lstrip("\n"))
                else:
                    body = content
                for j, call in enumerate(calls):
                    if (j == 0 and content) or j > 0:
                        body += "\n"
                    fn = call.get("function", call)
                    body += ('<tool_call>\n{"name": "' + str(fn.get("name", ""))
                             + '", "arguments": ' + _args_json(fn.get("arguments", {}), style)
                             + "}\n</tool_call>")
            body += IM_END
            emit(GEN_PROMPT)
            spans.append((pos, pos + len(body)))
            emit(body + "\n")
        elif role == "tool":
            merge = style != "ollama"
            prev_tool = merge and i > 0 and msgs[i - 1].get("role") == "tool"
            next_tool = merge and i + 1 < n and msgs[i + 1].get("role") == "tool"
            s = ("" if prev_tool else IM_START + "user")
            s += "\n<tool_response>\n" + content + "\n</tool_response>"
            if not next_tool:
                s += IM_END + "\n"
            emit(s)
    if add_generation_prompt:
        emit(GEN_PROMPT + ("<think>\n\n</think>\n\n" if enable_thinking is False else ""))
    return "".join(parts), spans


# ----------------------------------------------------------------- parse

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_assistant(text: str) -> dict:
    """Interpret one generated assistant turn.

    Returns ``{"thought", "content", "tool_calls": [{"name","arguments"}],
    "malformed": [raw...]}``. Truncated at the first end-of-turn marker.
    """
    for stop in (IM_END, IM_START, "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    thought = ""
    if "</think>" in text:
        thought = text.split("</think>")[0].split("<think>")[-1].strip("\n")
        text = text.split("</think>", 1)[1].lstrip("\n")
    elif text.startswith("<think>"):
        # Ran out of budget inside the thought.
        thought = text[len("<think>"):].strip("\n")
        text = ""
    calls, malformed = [], []
    for raw in _TOOL_CALL.findall(text):
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": str(obj.get("name", "")),
                          "arguments": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, AttributeError):
            malformed.append(raw)
    content = _TOOL_CALL.sub("", text).strip()
    if "<tool_call>" in content and not calls and not malformed:
        malformed.append(content)     # opened, never closed
        content = ""
    return {"thought": thought, "content": content, "tool_calls": calls,
            "malformed": malformed}


# ------------------------------------------------- internal trace -> chatml

def messages_from_internal(messages: list) -> list:
    """The dataset generators' message dicts -> Ollama-shaped messages.

    Internal assistant content is the compact JSON
    ``{"thought":..,"tool_call":{..}}`` / ``{"thought":..,"response":..}``.
    """
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            content = m.get("content", "")
            internal = json.loads(content) if isinstance(content, str) else content
            msg = {"role": "assistant", "content": "",
                   "reasoning_content": internal.get("thought", "")}
            if isinstance(internal.get("tool_call"), dict):
                msg["tool_calls"] = [{"function": internal["tool_call"]}]
            else:
                msg["content"] = internal.get("response", "")
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": str(m.get("content", "")),
                        "tool_name": m.get("name", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


_BLOCK = re.compile(r"<(system|user|assistant)>(.*?)</\1>(\x03?)|<tool name=([^>]+)>(.*?)</tool>",
                    re.S)


def trace_to_messages(text: str) -> tuple:
    """Parse a dataset trace in the legacy tagged form into (messages, tools).

    The generators (``agent/rich_dataset.py``) still produce the tagged
    serialization; this is how those traces are re-rendered as ChatML without
    touching the generators. The schema ``<system>`` block becomes the
    ``tools`` list; a free-text ``<system>`` block becomes a system message.
    """
    messages, tools = [], []
    for m in _BLOCK.finditer(text):
        tag, body, _eot, tool_name, tool_body = m.groups()
        if tool_name is not None:
            messages.append({"role": "tool", "content": tool_body,
                             "tool_name": tool_name})
            continue
        if tag == "system":
            if body.startswith('{"tools":'):
                try:
                    for entry in json.loads(body)["tools"]:
                        tools.append({"type": "function", "function": entry})
                    continue
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + body
            else:
                messages.insert(0, {"role": "system", "content": body})
        elif tag == "user":
            messages.append({"role": "user", "content": body})
        else:
            messages.extend(messages_from_internal(
                [{"role": "assistant", "content": body}]))
    return messages, tools
