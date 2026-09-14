"""An in-process Ollama-style client: the server's turn logic without HTTP.

Evaluations that want "what the client would see" use this rather than
``run_agent``, so they go through the same context assembly, generation and
parsing as ``scripts/serve_ollama.py``. Tools are executed here, on the
client side, exactly as an ``ollama`` client would execute them.
"""

from agent.ollama_context import assemble_chatml_context, assemble_context
from agent.ollama_format import assistant_to_ollama, toolbox_to_ollama_tools
from agent.turn import (generate_turn, make_grammar, protocol_for,
                        sampler_from_options)


def client_tools(toolbox) -> list:
    return [t for t in toolbox_to_ollama_tools(toolbox)
            if t["function"]["name"] != "finish"]


def chat_turn(model, tokenizer, messages, tools, *, device="cuda", options=None,
              constrained=True, num_ctx=16384, max_new=1024, seed=0) -> dict:
    """One assistant turn for an Ollama-shaped message list."""
    protocol = protocol_for(tokenizer)
    options = options or {"temperature": 0}
    reserve = min(max_new, max(256, num_ctx // 4))
    if protocol == "chatml":
        _, tokens, _ = assemble_chatml_context(messages, tools, num_ctx=num_ctx,
                                              reserve=reserve, tokenizer=tokenizer)
    else:
        _, tokens, _ = assemble_context(messages, tools, num_ctx=num_ctx,
                                        reserve=reserve, tokenizer=tokenizer)
    allowed = [t["function"]["name"] for t in tools] if tools else None
    result = generate_turn(model, tokens, sampler=sampler_from_options(options),
                           tokenizer=tokenizer, device=device, max_new=max_new,
                           grammar=make_grammar(tokenizer, allowed, constrained, protocol),
                           seed=seed, allowed_names=allowed, protocol=protocol)
    if result.kind == "tool_call":
        msg = assistant_to_ollama({"thought": result.thought, "tool_call": result.tool_call})
    else:
        msg = {"role": "assistant", "content": result.response, "thinking": result.thought}
    return {"message": msg, "done_reason": result.done_reason, "result": result}


def run_tool_loop(model, tokenizer, messages, toolbox, *, max_steps=8, **kw) -> dict:
    """The canonical client loop: call, execute, append, repeat.

    Returns ``{"content", "messages", "steps", "done_reason"}`` where ``steps``
    records every executed call as ``(name, arguments, observation)``.
    """
    tools = client_tools(toolbox)
    messages = list(messages)
    steps = []
    done_reason = "stop"
    for _ in range(max_steps):
        turn = chat_turn(model, tokenizer, messages, tools, **kw)
        msg = turn["message"]
        done_reason = turn["done_reason"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"content": msg.get("content", ""), "messages": messages,
                    "steps": steps, "done_reason": done_reason}
        fn = calls[0]["function"]
        obs = toolbox.run_json({"tool": fn["name"], "args": fn["arguments"]})
        steps.append((fn["name"], fn["arguments"], obs))
        messages.append({"role": "tool", "tool_name": fn["name"], "content": str(obs)})
    return {"content": "", "messages": messages, "steps": steps,
            "done_reason": "length"}
