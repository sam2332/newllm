"""The renderer must match Qwen3's own Jinja template character for character.

If these drift, the model is trained on one format and served another, and
Ollama's tool-call parser will not see what it expects.
"""

import json

import jinja2

from agent.chatml import (messages_from_internal, parse_assistant, render,
                          IM_END)

TEMPLATE = open("/home/lmeadows/llm/data/templates/qwen3.jinja").read()

TOOLS = [
    {"type": "function", "function": {
        "name": "evaluate_sum", "description": "Evaluate <expr> & return it.",
        "parameters": {"type": "object",
                       "properties": {"expr": {"type": "string",
                                               "description": "e.g. 12 + 8"}},
                       "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]

CONVERSATIONS = [
    # tools + system + a full loop with two tool results in a row
    dict(messages=[
        {"role": "system", "content": "You are Zed, a gruff engineer."},
        {"role": "user", "content": "What is 12 + 8, then save it?"},
        {"role": "assistant", "content": "", "reasoning_content": "Add first.",
         "tool_calls": [{"function": {"name": "evaluate_sum",
                                      "arguments": {"expr": "12 + 8"}}}]},
        {"role": "tool", "content": "20"},
        {"role": "assistant", "content": "Now save it.",
         "reasoning_content": "",
         "tool_calls": [{"function": {"name": "write_file",
                                      "arguments": {"path": "a.txt", "content": "20"}}}]},
        {"role": "tool", "content": "wrote 2 bytes to a.txt"},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Done: 20, saved to a.txt.",
         "reasoning_content": "All finished."},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "Any time.", "reasoning_content": "Polite."},
    ], tools=TOOLS),
    # no tools, system only
    dict(messages=[{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "Hello.", "reasoning_content": "greet"}],
         tools=None),
    # no system, no tools, thought embedded in content the HF way
    dict(messages=[{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nHello."}],
         tools=None),
]


def _jinja(messages, tools, add_generation_prompt, enable_thinking=None):
    env = jinja2.Environment()
    kwargs = dict(messages=messages, tools=tools,
                  add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return env.from_string(TEMPLATE).render(**kwargs)


def test_hf_style_matches_the_jinja_template():
    for conv in CONVERSATIONS:
        for gen in (False, True):
            ours, _ = render(conv["messages"], conv["tools"], add_generation_prompt=gen)
            theirs = _jinja(conv["messages"], conv["tools"], gen)
            assert ours == theirs, f"\n--- ours\n{ours}\n--- jinja\n{theirs}"
    ours, _ = render(CONVERSATIONS[1]["messages"], None, True, enable_thinking=False)
    assert ours == _jinja(CONVERSATIONS[1]["messages"], None, True, False)


def test_spans_cover_exactly_the_assistant_content():
    text, spans = render(CONVERSATIONS[0]["messages"], TOOLS)
    assert len(spans) == 4
    for start, end in spans:
        assert text[end - len(IM_END):end] == IM_END
        assert text[start - len("<|im_start|>assistant\n"):start] == "<|im_start|>assistant\n"
    # earlier assistant turns lose their think block; the last one keeps it
    first = text[spans[0][0]:spans[0][1]]
    last = text[spans[-1][0]:spans[-1][1]]
    assert "<think>" not in first and first.startswith("<tool_call>")
    assert last.startswith("<think>\nPolite.\n</think>\n\nAny time.")


def test_parse_assistant_round_trips_a_tool_call_and_a_response():
    turn = ('<think>\nAdd first.\n</think>\n\n<tool_call>\n{"name": "evaluate_sum", '
            '"arguments": {"expr": "12 + 8"}}\n</tool_call><|im_end|>')
    p = parse_assistant(turn)
    assert p["thought"] == "Add first." and p["content"] == ""
    assert p["tool_calls"] == [{"name": "evaluate_sum", "arguments": {"expr": "12 + 8"}}]
    p = parse_assistant("<think>\nok\n</think>\n\nThe answer is 20.<|im_end|>")
    assert p["content"] == "The answer is 20." and not p["tool_calls"]
    p = parse_assistant("<think>\nunfinished thought")
    assert p["thought"] == "unfinished thought" and p["content"] == ""
    p = parse_assistant("<tool_call>\n{not json}\n</tool_call>")
    assert p["malformed"] and not p["tool_calls"]


def test_internal_traces_convert_and_render():
    internal = [
        {"role": "user", "content": "What is 12 + 8?"},
        {"role": "assistant", "content": json.dumps(
            {"thought": "use calc", "tool_call": {"name": "op_42",
                                                  "arguments": {"expr": "12 + 8"}}},
            separators=(",", ":"))},
        {"role": "tool", "name": "op_42", "content": "20"},
        {"role": "assistant", "content": json.dumps(
            {"thought": "done", "response": "20"}, separators=(",", ":"))},
    ]
    msgs = messages_from_internal(internal)
    text, spans = render(msgs, TOOLS[:1])
    assert text == _jinja(msgs, TOOLS[:1], False)
    assert len(spans) == 2 and text[spans[1][0]:spans[1][1]].endswith("20" + IM_END)


def test_styles_differ_only_in_the_system_block():
    msgs = CONVERSATIONS[0]["messages"]
    hf, s1 = render(msgs, TOOLS, style="hf")
    compact, s2 = render(msgs, TOOLS, style="compact")
    ollama, s3 = render(msgs, TOOLS, style="ollama")
    user = "<|im_start|>user\nWhat is 12 + 8, then save it?<|im_end|>\n"
    assert user in hf and user in compact and user in ollama
    assert "\\u003c" in hf and "\\u003c" in ollama and '"expr":' in compact
    assert '{"type": "function", "function": {"name":"evaluate_sum"' in ollama
    # ollama style: no blank-line think block, tool results not merged
    assert "<think>Polite.</think>\nAny time." in ollama
    assert ollama.count("<|im_start|>user\n<tool_response>") == 3
    assert s1 and s2 and s3
