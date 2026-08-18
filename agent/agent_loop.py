"""Inference loop for the tiny ReAct agent.

Supports the standard JSON function-call format:

    Action: {"tool": "calc", "args": {"expr": "12 + 8"}}

Legacy bracket format is also parsed for backward compatibility:

    Action: calc[12 + 8]
"""

import re
import json
import torch
from agent.tools import Toolbox
from sampling import Sampler


# Legacy bracket-style action: Action: calc[12 + 8]
TOOL_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]")
THINK_START = "BEGIN_THINK"
THINK_END = "END_THINK"


def _extract_thinking_answer(text: str):
    """Pull out BEGIN_THINK block and Answer: string.

    The answer is only accepted if it appears after END_THINK.
    """
    think = ""
    answer = ""
    start = text.find(THINK_START)
    end = text.find(THINK_END)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(THINK_START):end].strip()
        # Answer must come after END_THINK
        ans_match = re.search(r"Answer:\s*([^\x03]+)", text[end:])
        if ans_match:
            answer = ans_match.group(1).split("\n")[0].strip()
    return think, answer


def _find_last_action(text: str, after_pos: int = 0):
    """Find the last Action:... line after after_pos."""
    last = None
    for m in TOOL_RE.finditer(text):
        if m.start() >= after_pos:
            last = m
    return last


def _find_unexecuted_action(text: str, executed_ends: set, after_pos: int = 0):
    """Find the first complete Action:... that has not been executed yet.

    Tries JSON function-call format first, then falls back to legacy bracket
    format.  Returns a dict with tool/args/position/end for JSON calls, or a
    regex match object for legacy calls.
    """
    # Look for JSON Action lines first.  We must parse balanced braces because
    # tool args contain nested JSON objects (e.g. {"args":{"expr":"1+1"}}).
    action_re = re.compile(r"Action:\s*")
    for m in action_re.finditer(text):
        if m.start() < after_pos:
            continue
        end = _find_json_end(text, m.end())
        if end is None:
            continue
        if end not in executed_ends:
            try:
                call = json.loads(text[m.end():end])
                return {"json": True, "call": call,
                        "start": m.start(), "end": end}
            except json.JSONDecodeError:
                continue
    # Fall back to legacy bracket format.
    for m in TOOL_RE.finditer(text):
        if m.end() not in executed_ends and m.start() >= after_pos:
            return {"json": False, "match": m,
                    "start": m.start(), "end": m.end()}
    return None


def _find_json_end(text: str, pos: int):
    """Return the index after a balanced JSON object starting at pos, or None."""
    i = pos
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    while i < n:
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def run_agent(model, question: str, toolbox: Toolbox,
              max_steps: int = 5, max_new: int = 120,
              sampler: Sampler = None, device: str = "cuda",
              greedy: bool = False) -> dict:
    """Run an interactive ReAct loop with the model.

    The model generates inside a BEGIN_THINK block. Whenever it emits an
    `Action: tool[arg]` line, we pause generation, run the tool, append the
    real observation, and let the model continue.  This supports both single-step
    and multi-step traces.

    Returns dict with final_answer, thinking, trace, and whether it succeeded.
    """
    if greedy:
        sampler = Sampler(temperature=0.01, top_k=1, top_p=1.0,
                          repetition_penalty=1.0)
    elif sampler is None:
        sampler = Sampler(temperature=0.5, top_k=20, top_p=0.9,
                          repetition_penalty=1.0)

    max_len = getattr(model, "max_len", 512)
    # seed with format prefix so model knows it must open a think block
    context = f"Question: {question}\n{THINK_START}\n"
    trace = [context]
    steps = []
    executed_action_ends = set()

    for step in range(max_steps):
        context_tokens = [min(ord(c), 255) for c in context]
        if len(context_tokens) > max_len:
            context_tokens = context_tokens[-max_len:]
        input_ids = torch.tensor([context_tokens], dtype=torch.long,
                                 device=device)
        generated = context_tokens[:]
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

                # If the model closed the think block, stop generating this turn.
                if THINK_END in text or "\x03" in text:
                    break

                # If it emitted a new Action line, execute it and continue.
                think_start = text.find(THINK_START)
                match = _find_unexecuted_action(text, executed_action_ends,
                                                after_pos=think_start)
                if match:
                    if match.get("json"):
                        call = match["call"]
                        result = toolbox.run_json(call)
                        tool_name = call.get("tool", call.get("name", "unknown"))
                        tool_arg = json.dumps(call.get("args", call.get("arguments", {})),
                                              separators=(",", ":"))
                    else:
                        m = match["match"]
                        tool_name, tool_arg = m.group(1), m.group(2)
                        result = toolbox.run(tool_name, tool_arg)
                    executed_action_ends.add(match["end"])
                    steps.append({"tool": tool_name, "arg": tool_arg,
                                  "result": result})
                    # Append the observation and continue generation in the
                    # same think block so multi-hop traces stay intact.  We
                    # update context but keep generating until END_THINK or
                    # another action appears.
                    context = text + f"\nObservation: {result}\n"
                    trace.append(generated_text[prefix_len:])
                    # Re-tokenise the updated context so the next token comes
                    # from the real observation, not stale generated noise.
                    context_tokens = [min(ord(c), 255) for c in context]
                    if len(context_tokens) > max_len:
                        context_tokens = context_tokens[-max_len:]
                    input_ids = torch.tensor([context_tokens], dtype=torch.long,
                                             device=device)
                    generated = context_tokens[:]
                    prefix_len = len(context_tokens)
                    continue
            else:
                # max_new reached without action or END_THINK
                trace.append(generated_text[prefix_len:])

        # If we exited because of END_THINK, extract final answer.
        if THINK_END in generated_text:
            trace.append(generated_text[prefix_len:])
            thinking, answer = _extract_thinking_answer(generated_text)
            # If there was an action inside, prefer the last tool result as answer.
            if steps:
                return {
                    "thinking": thinking,
                    "final_answer": str(steps[-1]["result"]),
                    "trace": "".join(trace),
                    "steps": steps,
                    "success": True,
                }
            return {
                "thinking": thinking,
                "final_answer": answer,
                "trace": "".join(trace),
                "steps": steps,
                "success": bool(answer),
            }

        # If no new action and no END_THINK, try to extract any answer.
        pending = _find_unexecuted_action(generated_text, executed_action_ends,
                                          after_pos=generated_text.find(THINK_START))
        if not pending:
            thinking, answer = _extract_thinking_answer(generated_text)
            if answer:
                trace.append(generated_text[prefix_len:])
                return {
                    "thinking": thinking,
                    "final_answer": answer,
                    "trace": "".join(trace),
                    "steps": steps,
                    "success": True,
                }
            # nothing useful happened; stop
            trace.append(generated_text[prefix_len:])
            break

    # Final fallback: use the last tool result if we executed any tools.
    if steps:
        return {
            "thinking": "",
            "final_answer": str(steps[-1]["result"]),
            "trace": "".join(trace),
            "steps": steps,
            "success": True,
        }

    thinking, answer = _extract_thinking_answer(context)
    return {
        "thinking": thinking,
        "final_answer": answer if answer else context,
        "trace": "".join(trace),
        "steps": steps,
        "success": bool(answer),
    }
