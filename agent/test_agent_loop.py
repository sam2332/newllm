"""Unit tests for the agent ReAct loop.

These tests do not need a trained model. They verify that the parser/executor
wires tools correctly so that future trained models are evaluated on a working
loop.
"""

import pytest
from agent.agent_loop import _find_unexecuted_action, _find_json_end, _extract_thinking_answer
from agent.tools import Toolbox


def test_find_json_end_nested():
    text = 'Action: {"tool":"calc","args":{"expr":"12 + 8"}}'
    end = _find_json_end(text, text.find("Action:") + len("Action:"))
    assert end == len(text)


def test_find_unexecuted_action_json():
    text = """BEGIN_THINK
Thought: compute.
Action: {"tool":"calc","args":{"expr":"12 + 8"}}
Observation:"""
    match = _find_unexecuted_action(text, set(), after_pos=text.find("BEGIN_THINK"))
    assert match is not None
    assert match["json"] is True
    assert match["call"] == {"tool": "calc", "args": {"expr": "12 + 8"}}


def test_find_unexecuted_action_legacy():
    text = """BEGIN_THINK
Thought: compute.
Action: calc[12 + 8]
Observation:"""
    match = _find_unexecuted_action(text, set(), after_pos=text.find("BEGIN_THINK"))
    assert match is not None
    assert match["json"] is False


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


def test_extract_thinking_answer():
    text = """BEGIN_THINK
Thought: compute.
Action: calc[12 + 8]
Observation: 20
END_THINK
Answer: 20"""
    think, ans = _extract_thinking_answer(text)
    assert "Thought:" in think
    assert "Observation: 20" in think
    assert ans == "20"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
