"""The HF importer must survive the shapes these datasets actually ship."""

import json
import re

from scripts.import_hf_dataset import row_to_trace


def _assistants(trace):
    return [json.loads(x) for x in re.findall(r"<assistant>(.*?)</assistant>",
                                              trace, re.S)]


def test_openai_style_tool_calls():
    row = {"messages": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather",
                          "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "name": "get_weather", "content": "18C, clear"},
        {"role": "assistant", "content": "It is 18C and clear in Paris."}],
        "tools": '[{"type": "function", "function": {"name": "get_weather"}}]'}
    trace, tools = row_to_trace(row)
    assert "<tool name=get_weather>18C, clear</tool>" in trace
    objs = _assistants(trace)
    assert objs[0]["tool_call"] == {"name": "get_weather",
                                    "arguments": {"city": "Paris"}}
    assert objs[-1]["response"] == "It is 18C and clear in Paris."
    assert tools and tools[0]["function"]["name"] == "get_weather"


def test_sharegpt_from_value_shape():
    row = {"conversations": [
        {"from": "human", "value": "Add 2 and 2."},
        {"from": "gpt", "value": "4"}]}
    trace, _ = row_to_trace(row)
    assert "<user>Add 2 and 2.</user>" in trace
    assert _assistants(trace)[-1]["response"] == "4"


def test_tool_call_tag_inside_content():
    row = {"messages": [
        {"role": "user", "content": "Search for cats."},
        {"role": "assistant",
         "content": '<tool_call>\n{"name": "search", "arguments": {"q": "cats"}}\n</tool_call>'},
        {"role": "tool", "content": "3 results"},
        {"role": "assistant", "content": "I found 3 results."}]}
    trace, _ = row_to_trace(row)
    objs = _assistants(trace)
    assert objs[0]["tool_call"]["name"] == "search"
    assert "thought" in objs[0]


def test_flat_query_answer_row():
    row = {"query": "Capital of France?", "answer": "Paris",
           "tools": [{"name": "noop"}]}
    trace, tools = row_to_trace(row)
    assert "<user>Capital of France?</user>" in trace
    assert _assistants(trace)[-1]["response"] == "Paris"
    assert tools == [{"name": "noop"}]


def test_bare_json_call_list_as_content():
    row = {"query": "Weather in Rome and Milan?",
           "answer": '[{"name": "wx", "arguments": {"city": "Rome"}}, '
                     '{"name": "wx", "arguments": {"city": "Milan"}}]'}
    trace, _ = row_to_trace(row)
    objs = _assistants(trace)
    assert [o["tool_call"]["arguments"]["city"] for o in objs if "tool_call" in o] \
        == ["Rome", "Milan"]


def test_unusable_rows_are_skipped_not_crashed():
    assert row_to_trace({})[0] is None
    assert row_to_trace({"messages": []})[0] is None
    # A trailing user turn has no answer to learn from.
    assert row_to_trace({"messages": [{"role": "user", "content": "hi"}]})[0] is None


def test_system_prompt_is_preserved():
    row = {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."}]}
    trace, _ = row_to_trace(row)
    assert trace.startswith("<system>You are terse.</system>")


def test_every_assistant_turn_is_valid_json_with_a_thought():
    row = {"messages": [
        {"role": "user", "content": "Do a thing"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "t", "arguments": "not json"}}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Done."}]}
    trace, _ = row_to_trace(row)
    for obj in _assistants(trace):
        assert obj["thought"]
        assert ("response" in obj) != ("tool_call" in obj)
