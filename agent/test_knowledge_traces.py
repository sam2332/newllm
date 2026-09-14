"""The knowledge slice must stay grounded and must not always call a tool."""

import json
import random
import re

from agent.knowledge_traces import KnowledgeComposer, generate_knowledge_traces

ITEMS = [
    {"kind": "explain", "domain": "python", "topic": "generators",
     "question": "What is a generator?",
     "answer": "A generator is a function that yields values lazily instead of "
               "building a whole list in memory.",
     "followups": [{"question": "When should I use one?",
                    "answer": "When the sequence is large or unbounded."}]},
    {"kind": "explain", "domain": "bash", "topic": "set -e",
     "question": "Why use set -e?",
     "answer": "It makes the script exit on the first failing command.",
     "followups": []},
    {"kind": "artifact", "domain": "python", "topic": "argparse",
     "language": "python", "request": "Write me a CLI that greets someone.",
     "filename": "greet.py", "summary": "Greets a name given on the command line.",
     "code": "import argparse\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n"
             "    ap.add_argument('name')\n    print('hello', ap.parse_args().name)\n",
     "followups": [{"question": "What does main() do?",
                    "answer": "It parses the name argument and prints a greeting."},
                   {"question": "How do I run it?",
                    "answer": "python greet.py Ada"}]},
]


def test_explain_traces_call_no_tool():
    comp = KnowledgeComposer(ITEMS, random.Random(0))
    for _ in range(20):
        msgs = comp.compose_explain()
        assert msgs, "composer produced nothing"
        assert all(m["role"] != "tool" for m in msgs)
        assert all("tool_call" not in m["content"] for m in msgs
                   if m["role"] == "assistant")


def test_artifact_observations_come_from_the_workspace():
    """Every tool result must be a real VirtualWorkspace return value, not
    teacher text: a write says how many bytes, a read returns the exact code."""
    comp = KnowledgeComposer(ITEMS, random.Random(3))
    msgs = comp.compose_artifact(max_turns=16)
    tools = [m for m in msgs if m["role"] == "tool"]
    assert tools, "no tool calls in an artifact arc"
    wrote = [m for m in tools if m["name"] == "write_file"]
    assert wrote and re.fullmatch(r"wrote \d+ bytes to \S+", wrote[0]["content"])
    for m in tools:
        if m["name"] == "read_file":
            assert "argparse" in m["content"]


def test_every_assistant_message_is_valid_json():
    traces = generate_knowledge_traces(ITEMS, 40, seed=1)
    assert len(traces) == 40
    for t in traces:
        for raw in re.findall(r"<assistant>(.*?)</assistant>", t, re.S):
            obj = json.loads(raw)
            assert "thought" in obj
            assert ("response" in obj) != ("tool_call" in obj)


def test_some_traces_have_no_tool_call_at_all():
    traces = generate_knowledge_traces(ITEMS, 60, seed=2, explain_fraction=0.5)
    no_tool = [t for t in traces if "tool_call" not in t]
    assert len(no_tool) > 10, f"only {len(no_tool)}/60 traces avoid tools"
