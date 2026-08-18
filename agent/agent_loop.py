"""Inference loop for the JSON message agent.

The model always responds with valid JSON:
    {"thought": "...", "tool_call": {"name": "...", "arguments": {...}}}
    {"thought": "...", "response": "..."}

The loop starts from a system+user prompt, runs tool calls, appends real
observations, and continues until the model emits a response or a stop token.
"""

import json
import re
import torch

from agent.tools import Toolbox
from sampling import Sampler


EOT = "\x03"
SYSTEM_PROMPT = (
    "You are a helpful reasoning agent. Available tools: calc(expr), "
    "search_memory(key), web_search(query), now(). "
    "Every response must be valid JSON with a 'thought' string and exactly "
    "one of 'tool_call' or 'response'. "
    "Use tool_call to call a tool: {\"thought\": \"...\", \"tool_call\": "
    "{\"name\": \"calc\", \"arguments\": {\"expr\": \"12 + 8\"}}}. "
    "Use response to give the final answer: {\"thought\": \"...\", "
    "\"response\": \"20\"}."
)


_ROLE_TAGS = re.compile(r"<(system|user|assistant|tool)(?:\s+name=([^>]+))?>"
                        r"(.*?)</\1>", re.DOTALL)


def _tokenize(text: str, max_vocab: int = 256) -> list:
    return [min(ord(c), max_vocab - 1) for c in text]


def _build_context(question: str, history: list) -> str:
    """Build the text the model sees from the system prompt, question, and
    any prior assistant/tool turns."""
    parts = [f"<system>{SYSTEM_PROMPT}</system>", f"<user>{question}</user>"]
    parts.extend(history)
    return "\n".join(parts)


def _extract_assistant_json(text: str, after_pos: int = 0) -> tuple:
    """Find and parse the first <assistant> JSON block at or after after_pos."""
    tag_start = text.find("<assistant>", after_pos)
    if tag_start == -1:
        return None, None, None
    content_start = tag_start + len("<assistant>")
    tag_end = text.find("</assistant>", content_start)
    if tag_end == -1:
        return None, None, None
    raw = text[content_start:tag_end].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage a leading valid JSON object if extra tokens follow.
        for end in range(len(raw), 0, -1):
            try:
                parsed = json.loads(raw[:end])
                return tag_start, tag_end + len("</assistant>"), parsed
            except json.JSONDecodeError:
                continue
        return None, None, None
    return tag_start, tag_end + len("</assistant>"), parsed


def _find_unexecuted_action(text: str, executed_ends: set) -> tuple:
    """Return (end_pos, parsed_json, start_pos) for the first unexecuted assistant action."""
    pos = 0
    while True:
        start, end, parsed = _extract_assistant_json(text, after_pos=pos)
        if start is None:
            return None, None, None
        if end not in executed_ends:
            if isinstance(parsed.get("tool_call"), dict):
                return end, parsed, start
        pos = end


def _find_final_response(text: str) -> tuple:
    """Return (response_text, parsed_json) of the last assistant message."""
    last = None
    pos = 0
    while True:
        start, end, parsed = _extract_assistant_json(text, after_pos=pos)
        if start is None:
            break
        last = parsed
        pos = end
    if last is None:
        return "", None
    response = last.get("response")
    if isinstance(response, str):
        return response, last
    return "", last


def run_agent(model, question: str, toolbox: Toolbox,
              max_steps: int = 5, max_new: int = 120,
              sampler: Sampler = None, device: str = "cuda",
              greedy: bool = False) -> dict:
    """Run an interactive JSON tool-use loop with the model.

    Returns dict with final_answer, thinking, trace, steps, and success.
    """
    if greedy:
        sampler = Sampler(temperature=0.01, top_k=1, top_p=1.0,
                          repetition_penalty=1.0)
    elif sampler is None:
        sampler = Sampler(temperature=0.5, top_k=20, top_p=0.9,
                          repetition_penalty=1.0)

    max_len = getattr(model, "max_len", 512)
    history = []
    trace_parts = [f"<system>{SYSTEM_PROMPT}</system>", f"<user>{question}</user>"]
    steps = []
    executed_ends = set()

    for step in range(max_steps):
        context = _build_context(question, history)
        context_tokens = _tokenize(context)
        if len(context_tokens) > max_len:
            context_tokens = context_tokens[-max_len:]
        input_ids = torch.tensor([context_tokens], dtype=torch.long,
                                 device=device)
        generated = list(context_tokens)
        prefix_len = len(context_tokens)
        model.eval()
        generated_text = ""
        with torch.no_grad():
            for _ in range(max_new):
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
                if logits.dim() == 4:
                    logits = logits[:, :, 0, :]
                next_logits = logits[:, -1, :]
                gen_tensor = torch.tensor(generated, dtype=torch.long,
                                          device=device)
                next_token = sampler.sample(next_logits, gen_tensor)
                generated.append(next_token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)
                if input_ids.size(1) > max_len:
                    input_ids = input_ids[:, -max_len:]

                text = "".join(chr(min(t, 255)) for t in generated)
                generated_text = text

                if EOT in text:
                    break

                end_pos, parsed, start_pos = _find_unexecuted_action(text, executed_ends)
                if parsed is not None:
                    tool_call = parsed.get("tool_call", {})
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    result = toolbox.run_json({
                        "tool": tool_name,
                        "args": tool_args,
                    })
                    executed_ends.add(end_pos)
                    steps.append({
                        "tool": tool_name,
                        "arg": json.dumps(tool_args, separators=(",", ":")),
                        "result": result,
                        "thought": parsed.get("thought", ""),
                    })
                    assistant_block = text[start_pos:end_pos]
                    tool_block = f"<tool name={tool_name}>{result}</tool>"
                    history.append(assistant_block)
                    history.append(tool_block)
                    trace_parts.append(assistant_block)
                    trace_parts.append(tool_block)
                    break

            else:
                trace_parts.append(generated_text[prefix_len:])

        full_text = _build_context(question, history) + generated_text[prefix_len:]
        response, last = _find_final_response(full_text)
        if response:
            trace_parts.append(generated_text[prefix_len:])
            return {
                "thinking": last.get("thought", "") if last else "",
                "final_answer": response,
                "trace": "\n".join(trace_parts),
                "steps": steps,
                "success": True,
            }

        end_pos, parsed, _ = _find_unexecuted_action(full_text, executed_ends)
        if parsed is None:
            trace_parts.append(generated_text[prefix_len:])
            break

    response, last = _find_final_response(_build_context(question, history))
    return {
        "thinking": last.get("thought", "") if last else "",
        "final_answer": response,
        "trace": "\n".join(trace_parts),
        "steps": steps,
        "success": bool(response),
    }
