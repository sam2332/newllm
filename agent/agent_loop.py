"""Inference loop for the JSON message agent.

The model always responds with valid JSON:
    {"thought": "...", "tool_call": {"name": "...", "arguments": {...}}}
    {"thought": "...", "response": "..."}

The loop starts from a system+user prompt, runs tool calls, appends real
observations, and continues until the model emits a response or a stop token.
"""

import json
import torch

from agent.tools import Toolbox
from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.generate import generate
from agent.constrained import AssistantGrammar
from agent.tool_schema import schema_block_from_toolbox
from sampling import Sampler


EOT = "\x03"


def _build_context(question: str, history: list,
                   conversation: list = None, system: str = None) -> str:
    """Build the exact tagged message form used in the training dataset.

    ``conversation`` carries completed prior turns for multi-turn chat. Each
    entry is a raw tagged block (``<user>..</user>``, ``<assistant>..</assistant>``
    optionally followed by EOT, or ``<tool ..>..</tool>``) exactly as the
    training serializer would have written it.
    """
    parts = []
    if system:
        # Already-tagged blocks pass through; bare text gets wrapped.
        parts.append(system if system.startswith("<system>")
                     else f"<system>{system}</system>")
    if conversation:
        parts.extend(conversation)
    parts.append(f"<user>{question}</user>")
    parts.extend(history)
    # Training serializes every message boundary with a newline, including
    # the boundary before the next assistant turn.
    return "\n".join(parts) + "\n"


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
        parsed = _validate_assistant_message(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        # Try to salvage a leading valid JSON object if extra tokens follow.
        for end in range(len(raw), 0, -1):
            try:
                parsed = _validate_assistant_message(json.loads(raw[:end]))
                return tag_start, tag_end + len("</assistant>"), parsed
            except (json.JSONDecodeError, ValueError):
                continue
        return None, None, None
    return tag_start, tag_end + len("</assistant>"), parsed


def _validate_assistant_message(parsed: object) -> dict:
    """Validate the JSON contract before treating a model output as a turn."""
    if not isinstance(parsed, dict):
        raise ValueError("assistant message must be a JSON object")
    if not isinstance(parsed.get("thought"), str):
        raise ValueError("assistant message requires a string thought")
    has_tool_call = "tool_call" in parsed
    has_response = "response" in parsed
    if has_tool_call == has_response:
        raise ValueError("assistant message requires exactly one action field")
    if has_response:
        if not isinstance(parsed["response"], str):
            raise ValueError("assistant response must be a string")
    else:
        tool_call = parsed["tool_call"]
        if not isinstance(tool_call, dict):
            raise ValueError("tool_call must be an object")
        if not isinstance(tool_call.get("name"), str):
            raise ValueError("tool_call requires a string name")
        if not isinstance(tool_call.get("arguments"), dict):
            raise ValueError("tool_call requires object arguments")
    return parsed


def _find_unexecuted_action(text: str, after_pos: int = 0) -> tuple:
    """Return the first valid tool call generated at or after ``after_pos``."""
    pos = after_pos
    while True:
        start, end, parsed = _extract_assistant_json(text, after_pos=pos)
        if start is None:
            return None, None, None
        if isinstance(parsed.get("tool_call"), dict):
            return end, parsed, start
        pos = end


def _find_final_response(text: str, after_pos: int = 0) -> tuple:
    """Return the last final response generated at or after ``after_pos``."""
    last = None
    pos = after_pos
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


def _select_final_answer(response: str, steps: list) -> str:
    """Prefer real tool output over a model-generated copy of that output."""
    if steps:
        return str(steps[-1]["result"])
    return response


def run_agent(model, question: str, toolbox: Toolbox,
              max_steps: int = 5, max_new: int = 120,
              sampler: Sampler = None, device: str = "cuda",
              greedy: bool = False,
              tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
              constrained: bool = False,
              conversation: list = None,
              system: str = None) -> dict:
    """Run an interactive JSON tool-use loop with the model.

    Returns dict with final_answer, thinking, trace, steps, and success.
    """
    if greedy:
        sampler = Sampler(temperature=0.01, top_k=1, top_p=1.0,
                          repetition_penalty=1.0)
    elif sampler is None:
        sampler = Sampler(temperature=0.5, top_k=20, top_p=0.9,
                          repetition_penalty=1.0)

    grammar = (AssistantGrammar(list(toolbox.tools.keys()), tokenizer)
               if constrained else None)

    # The model is trained to read the tool schema from context, so it must be
    # given the schema of whatever toolbox it is actually serving. Without
    # this it can only guess names, which is what made renaming catastrophic.
    if system is None and getattr(toolbox, "tools", None):
        system = schema_block_from_toolbox(toolbox)

    max_len = getattr(model, "max_len", 512)
    history = []
    trace_parts = [f"<user>{question}</user>"]
    turn_blocks = []
    steps = []

    for step in range(max_steps):
        context = _build_context(question, history,
                                 conversation=conversation, system=system)
        context_tokens = tokenizer.encode(context)
        if len(context_tokens) > max_len:
            context_tokens = context_tokens[-max_len:]

        # One incremental pass with a KV cache, stopping at </assistant>.
        delta = generate(model, context_tokens, sampler, max_new=max_new,
                         device=device, tokenizer=tokenizer, grammar=grammar)

        full_text = context + delta
        context_char_len = len(context)

        end_pos, parsed, start_pos = _find_unexecuted_action(
            full_text, after_pos=context_char_len)
        if parsed is not None:
            tool_call = parsed.get("tool_call", {})
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            result = toolbox.run_json({"tool": tool_name, "args": tool_args})
            steps.append({
                "tool": tool_name,
                "arg": json.dumps(tool_args, separators=(",", ":")),
                "result": result,
                "thought": parsed.get("thought", ""),
            })
            assistant_block = full_text[start_pos:end_pos]
            tool_block = f"<tool name={tool_name}>{result}</tool>"
            history.append(assistant_block)
            history.append(tool_block)
            trace_parts.append(assistant_block)
            trace_parts.append(tool_block)
            turn_blocks.extend([assistant_block, tool_block])
            continue

        response, last = _find_final_response(full_text,
                                              after_pos=context_char_len)
        trace_parts.append(delta)
        if response:
            start, end, _ = _extract_assistant_json(full_text,
                                                    after_pos=context_char_len)
            if start is not None:
                turn_blocks.append(full_text[start:end] + EOT)
            return {
                "thinking": last.get("thought", "") if last else "",
                "final_answer": _select_final_answer(response, steps),
                "model_response": response,
                "trace": "\n".join(trace_parts),
                "steps": steps,
                "success": True,
                "turn_blocks": [f"<user>{question}</user>"] + turn_blocks,
            }
        break

    response, last = _find_final_response(
        _build_context(question, history, conversation=conversation,
                       system=system))
    return {
        "thinking": last.get("thought", "") if last else "",
        "final_answer": _select_final_answer(response, steps),
        "model_response": response,
        "trace": "\n".join(trace_parts),
        "steps": steps,
        "success": bool(response),
        "turn_blocks": [f"<user>{question}</user>"] + turn_blocks,
    }


def run_chat(model, turns, toolbox: Toolbox, system: str = None, **kwargs):
    """Run a multi-turn conversation, carrying context between turns.

    ``turns`` is a list of user messages. Each turn sees every prior user
    message, assistant answer and tool observation, which is what makes
    coreference ("multiply that by 3") resolvable.
    """
    conversation = []
    results = []
    for turn in turns:
        result = run_agent(model, turn, toolbox, conversation=list(conversation),
                           system=system, **kwargs)
        conversation.extend(result.get("turn_blocks", []))
        results.append(result)
    return results
