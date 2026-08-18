"""Unit tests for the JSON message agent loop.

These tests do not need a trained model. They verify that the parser/executor
wires tools correctly so that future trained models are evaluated on a working
loop.
"""

import pytest
from agent.agent_loop import _extract_assistant_json, _find_unexecuted_action, _find_final_response
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


def test_find_unexecuted_action_skips_executed():
    text = (
        '<assistant>{"thought":"first","tool_call":{"name":"calc","arguments":{"expr":"1+1"}}}</assistant>'
        '<tool name=calc>2</tool>'
        '<assistant>{"thought":"second","tool_call":{"name":"calc","arguments":{"expr":"2+2"}}}</assistant>'
    )
    executed = {len(text.split("</assistant>")[0]) + len("</assistant>")}
    end_pos, parsed, start_pos = _find_unexecuted_action(text, executed)
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


def test_toolbox_json_calc():
    tb = Toolbox()
    assert tb.run_json({"tool": "calc", "args": {"expr": "12 + 8"}}) == "20"


def test_toolbox_json_memory():
    tb = Toolbox()
    assert tb.run_json({"tool": "search_memory", "args": {"key": "leader"}}) == "grug"


def test_toolbox_json_web_search():
    tb = Toolbox()
    assert tb.run_json({"tool": "web_search", "args": {"query": "capital of france"}}) == "Paris"


def test_toolbox_legacy_web_search():
    tb = Toolbox()
    assert tb.run("web_search", "largest planet") == "Jupiter"


def test_toolbox_unknown_tool():
    tb = Toolbox()
    assert tb.run_json({"tool": "nope", "args": {}}).startswith("ERROR")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
