"""Context assembly from Ollama messages must reproduce the training form."""

import json

from agent.ollama_context import (assemble_context, from_model_text,
                                  system_blocks, to_model_text,
                                  turn_blocks_from_messages, EOT)
from agent.ollama_format import (assistant_to_ollama, toolbox_to_ollama_tools,
                                 tool_call_id)
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from agent.tool_schema import schema_block_from_toolbox, schema_block_from_tools
from agent.tools import Toolbox

ADD = [{"type": "function", "function": {
    "name": "add_numbers", "description": "Add two integers.",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "integer"},
                                  "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}]


def test_client_tools_render_like_the_toolbox():
    tb = Toolbox()
    tools = [t for t in toolbox_to_ollama_tools(tb)
             if t["function"]["name"] != "finish"]
    assert schema_block_from_tools(tools) == schema_block_from_toolbox(tb)


def test_empty_tools_means_no_schema_block():
    assert schema_block_from_tools([]) == ""
    assert system_blocks([], ["You are Zed."]) == ["<system>You are Zed.</system>"]


def test_schema_first_then_free_text():
    blocks = system_blocks(ADD, ["You are Zed.", "Be brief."])
    assert len(blocks) == 2
    assert blocks[0].startswith('<system>{"tools":[{"name":"add_numbers"')
    assert blocks[1] == "<system>You are Zed.\nBe brief.</system>"
    assert system_blocks(ADD, ["You are Zed."], system_mode="drop") == [blocks[0]]


def test_turn_blocks_round_trip_a_tool_loop():
    call = {"thought": "Use the adder.",
            "tool_call": {"name": "add_numbers", "arguments": {"a": 12, "b": 8}}}
    final = {"thought": "Done.", "response": "20"}
    messages = [
        {"role": "system", "content": "You are Zed."},
        {"role": "user", "content": "What is 12 + 8?"},
        assistant_to_ollama(call),
        {"role": "tool", "tool_name": "add_numbers", "content": "20"},
        assistant_to_ollama(final),
        {"role": "user", "content": "Thanks."},
        assistant_to_ollama(call),
        {"role": "tool", "tool_name": "add_numbers", "content": "20"},   # trailing
    ]
    turns = turn_blocks_from_messages(messages)
    assert len(turns) == 2
    assert turns[0] == [
        "<user>What is 12 + 8?</user>",
        "<assistant>" + json.dumps(call, separators=(",", ":")) + "</assistant>",
        "<tool name=add_numbers>20</tool>",
        "<assistant>" + json.dumps(final, separators=(",", ":")) + "</assistant>" + EOT,
    ]
    # The trailing observation is what the model must answer from; keep it.
    assert turns[1][-1] == "<tool name=add_numbers>20</tool>"


def test_multiple_tool_calls_become_interleaved_pairs():
    msg = {"role": "assistant", "thinking": "two calls",
           "tool_calls": [
               {"function": {"name": "add_numbers", "arguments": {"a": 1, "b": 2}}},
               {"function": {"name": "mul", "arguments": '{"a": 3}'}}]}
    messages = [{"role": "user", "content": "go"}, msg,
                {"role": "tool", "tool_name": "mul", "content": "9"},
                {"role": "tool", "tool_name": "add_numbers", "content": "3"}]
    blocks = turn_blocks_from_messages(messages)[0]
    assert blocks[1].startswith('<assistant>{"thought":"two calls","tool_call":{"name":"mul"')
    assert blocks[2] == "<tool name=mul>9</tool>"
    assert '"name":"add_numbers"' in blocks[3]
    assert blocks[4] == "<tool name=add_numbers>3</tool>"


def test_thought_recovered_from_cache_when_client_strips_thinking():
    cache = {("add_numbers", '{"a":1,"b":2}'): "cached thought"}
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"function": {"name": "add_numbers",
                                        "arguments": {"b": 2, "a": 1}}}]}
    turns = turn_blocks_from_messages([{"role": "user", "content": "go"}, msg],
                                      thought_cache=cache)
    assert '"thought":"cached thought"' in turns[0][1]


def test_tool_call_ids_are_deterministic_and_carry_index():
    a = assistant_to_ollama({"thought": "t", "tool_call": {"name": "x", "arguments": {"k": 1}}})
    b = assistant_to_ollama({"thought": "t", "tool_call": {"name": "x", "arguments": {"k": 1}}})
    assert a["tool_calls"][0]["id"] == b["tool_calls"][0]["id"] == tool_call_id("x", {"k": 1})
    assert a["tool_calls"][0]["function"]["index"] == 0


def test_transport_round_trips_non_ascii():
    text = "héllo € 😀"
    assert from_model_text(to_model_text(text)) == text
    assert all(ord(c) < 256 for c in to_model_text(text))
    assert all(t < 256 for t in TOK.encode(to_model_text(text)))


def test_truncation_keeps_head_and_last_turn():
    messages = [{"role": "system", "content": "You are Zed."}]
    for i in range(40):
        messages.append({"role": "user", "content": f"question number {i} " * 5})
        messages.append({"role": "assistant", "content": f"answer {i}"})
    messages.append({"role": "user", "content": "final question"})
    context, tokens, info = assemble_context(messages, ADD, num_ctx=600,
                                             reserve=100, tokenizer=TOK)
    assert context.startswith('<system>{"tools":')
    assert "<system>You are Zed.</system>" in context
    assert context.rstrip("\n").endswith("<user>final question</user>")
    assert info["dropped_turns"] > 0
    assert info["hard_cut_tokens"] == 0
    assert len(tokens) <= 500
    assert context.endswith("\n")


def test_last_turn_overflow_sheds_oldest_pairs_not_the_user_block():
    messages = [{"role": "user", "content": "q"}]
    for i in range(30):
        messages.append(assistant_to_ollama(
            {"thought": "t" * 20, "tool_call": {"name": "add_numbers",
                                                "arguments": {"a": i, "b": i}}}))
        messages.append({"role": "tool", "tool_name": "add_numbers", "content": "x" * 40})
    context, _, info = assemble_context(messages, ADD, num_ctx=700, reserve=50,
                                        tokenizer=TOK)
    assert "<user>q</user>" in context
    assert info["trimmed_pairs"] > 0
    assert info["dropped_turns"] == 0
