"""Unit tests for the JSON message agent loop.

These tests do not need a trained model. They verify that the parser/executor
wires tools correctly so that future trained models are evaluated on a working
loop.
"""

import pytest
from agent.agent_dataset import generate_agent_dataset
from agent.agent_loop import (
    _build_context,
    _extract_assistant_json,
    _find_unexecuted_action,
    _find_final_response,
    _select_final_answer,
)
from agent.tools import Toolbox


def test_extract_assistant_json_tool_call():
    text = '<assistant>{"thought":"compute","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>'
    start, end, parsed = _extract_assistant_json(text)
    assert parsed["thought"] == "compute"
    assert parsed["tool_call"]["name"] == "calc"


def test_extract_assistant_json_response():
    text = '<assistant>{"thought":"done","response":"20"}</assistant>'
    start, end, parsed = _extract_assistant_json(text)
    assert parsed["response"] == "20"


@pytest.mark.parametrize("content", [
    '{"response":"20"}',
    '{"thought":"done","response":20}',
    '{"thought":"done","tool_call":{"name":"calc","arguments":{}},"response":"20"}',
    '{"thought":"call","tool_call":{"name":"calc","arguments":"12 + 8"}}',
])
def test_extract_assistant_json_rejects_invalid_contract(content):
    text = f"<assistant>{content}</assistant>"
    assert _extract_assistant_json(text) == (None, None, None)


def test_inference_context_matches_training_message_format():
    assert _build_context("What is 12 + 8?", []) == "<user>What is 12 + 8?</user>\n"
    assert _build_context("What is 12 + 8?", [
        '<assistant>{"thought":"calculate","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>',
        "<tool name=calc>20</tool>",
    ]) == (
        "<user>What is 12 + 8?</user>\n"
        '<assistant>{"thought":"calculate","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>\n'
        "<tool name=calc>20</tool>\n"
    )


def test_full_dataset_traces_fit_json_context_budget():
    traces = generate_agent_dataset(num_samples=1000, max_len=768)
    assert max(map(len, traces)) <= 768


def test_find_unexecuted_action_skips_executed():
    text = (
        '<assistant>{"thought":"first","tool_call":{"name":"calc","arguments":{"expr":"1+1"}}}</assistant>'
        '<tool name=calc>2</tool>'
        '<assistant>{"thought":"second","tool_call":{"name":"calc","arguments":{"expr":"2+2"}}}</assistant>'
    )
    after_first = len(text.split("</assistant>")[0]) + len("</assistant>")
    end_pos, parsed, start_pos = _find_unexecuted_action(text, after_pos=after_first)
    assert parsed is not None
    assert parsed["tool_call"]["arguments"]["expr"] == "2+2"


def test_find_final_response():
    text = (
        '<assistant>{"thought":"calc","tool_call":{"name":"calc","arguments":{"expr":"1+1"}}}</assistant>'
        '<tool name=calc>2</tool>'
        '<assistant>{"thought":"done","response":"2"}</assistant>'
    )
    response, parsed = _find_final_response(text)
    assert response == "2"


def test_real_tool_result_is_authoritative_final_answer():
    steps = [{"tool": "calc", "result": "12.0"}]
    assert _select_final_answer("1.20", steps) == "12.0"
    assert _select_final_answer("plain answer", []) == "plain answer"


def test_toolbox_json_calc():
    tb = Toolbox()
    assert tb.run_json({"tool": "calc", "args": {"expr": "12 + 8"}}) == "20"


def test_toolbox_json_memory():
    tb = Toolbox()
    assert tb.run_json({"tool": "search_memory", "args": {"key": "leader"}}) == "grug"


def test_toolbox_json_web_search():
    tb = Toolbox()
    assert tb.run_json({"tool": "web_search", "args": {"query": "capital of france"}}) == "Paris"


def test_toolbox_recovers_conservative_near_miss_intent():
    tb = Toolbox()
    assert tb.run_json({"tool": "search_mememory", "args": {"key": "lersion"}}) == "0.1"
    assert tb.run_json({"tool": "web_search", "args": {"query": "capital of frence"}}) == "Paris"


def test_toolbox_legacy_web_search():
    tb = Toolbox()
    assert tb.run("web_search", "largest planet") == "Jupiter"


def test_toolbox_unknown_tool():
    tb = Toolbox()
    assert tb.run_json({"tool": "nope", "args": {}}).startswith("ERROR")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
