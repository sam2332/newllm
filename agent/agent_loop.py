"""Inference loop for the tiny ReAct agent.
"""

import re
import torch
from agent.tools import Toolbox
from sampling import Sampler


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


def run_agent(model, question: str, toolbox: Toolbox,
              max_steps: int = 5, max_new: int = 120,
              sampler: Sampler = None, device: str = "cuda",
              greedy: bool = False) -> dict:
    """Run a ReAct loop with the model.

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

    for step in range(max_steps):
        context_tokens = [min(ord(c), 255) for c in context]
        # keep only last max_len tokens so RoPE buffers fit
        if len(context_tokens) > max_len:
            context_tokens = context_tokens[-max_len:]
        input_ids = torch.tensor([context_tokens], dtype=torch.long,
                                 device=device)
        generated = context_tokens[:]
        prefix_len = len(context_tokens)
        model.eval()
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
                # stop if we closed thinking or emitted an answer
                if THINK_END in text or "Answer:" in text or "" in text:
                    break

        text = "".join(chr(min(t, 255)) for t in generated)
        trace.append(text[prefix_len:])

        # If the model closed thinking, parse answer directly.
        if THINK_END in text:
            thinking, answer = _extract_thinking_answer(text)
            return {
                "thinking": thinking,
                "final_answer": answer,
                "trace": "".join(trace),
                "steps": steps,
                "success": bool(answer),
            }

        # Otherwise look for the *last* action generated in this turn.
        match = _find_last_action(text, after_pos=text.find(THINK_START))
        if match:
            tool_name, tool_arg = match.group(1), match.group(2)
            result = toolbox.run(tool_name, tool_arg)
            steps.append({"tool": tool_name, "arg": tool_arg,
                          "result": result})
            # append observation on its own line, keep think block open
            context = text + f"\nObservation: {result}\n"
            if tool_name == "finish":
                # finish supplies the answer; synthesize a closed trace
                thinking, _ = _extract_thinking_answer(text)
                return {
                    "thinking": thinking,
                    "final_answer": str(result),
                    "trace": "".join(trace) + f"\nObservation: {result}",
                    "steps": steps,
                    "success": bool(result),
                }
            continue

        # no tool call and no closed thinking; try answer
        thinking, answer = _extract_thinking_answer(text)
        if answer:
            return {
                "thinking": thinking,
                "final_answer": answer,
                "trace": "".join(trace),
                "steps": steps,
                "success": True,
            }
        break

    # try to extract any answer from final context
    thinking, answer = _extract_thinking_answer(context)
    return {
        "thinking": thinking,
        "final_answer": answer if answer else context,
        "trace": "".join(trace),
        "steps": steps,
        "success": bool(answer),
    }
