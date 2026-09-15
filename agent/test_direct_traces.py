"""No-tool conversations: no call anywhere, and half must still carry a schema
so "tools offered but not needed" is learnable."""

import json
import random
import re

from agent.chatml import render, trace_to_messages
from agent.direct_traces import generate_direct_traces
from agent.tool_schema import ToolSchemaSampler, randomize_trace

STORY = {"title": "The Glass Harbor", "genre": "sea adventure",
         "premise": "A salvage crew finds a map to a drowned city.",
         "protagonist": {"name": "Ines Varga", "trait": "reckless"},
         "characters": [{"name": "Tomas Reel", "role": "the navigator"}],
         "setting": "a storm-lashed archipelago",
         "chapters": [{"title": "One", "summary": "s", "text": "t"}]}


def test_traces_contain_no_tool_call_at_all():
    for t in generate_direct_traces(40, [STORY], seed=1):
        assert '"tool_call"' not in t and "<tool name=" not in t
        assert t.count("<user>") >= 2
        for m in re.findall(r"<assistant>(\{.*?\})</assistant>", t):
            obj = json.loads(m)
            assert "response" in obj and "tool_call" not in obj and obj["thought"]
        assert t.endswith("\x03")


def test_force_schema_attaches_tools_that_are_never_called():
    rng = random.Random(0)
    t = generate_direct_traces(1, [STORY], seed=2)[0]
    plain = randomize_trace(t, rng, ToolSchemaSampler(rng))
    assert plain == t                       # no tools used -> no schema
    forced = randomize_trace(t, rng, ToolSchemaSampler(rng), force_schema=True)
    assert forced.startswith('<system>{"tools":[')
    schema = json.loads(re.match(r"<system>(.*?)</system>", forced, re.S).group(1))
    assert len(schema["tools"]) >= 2
    assert '"tool_call"' not in forced      # offered, still unused


def test_renders_to_chatml_with_and_without_tools():
    rng = random.Random(4)
    t = generate_direct_traces(1, [STORY], seed=3)[0]
    for forced in (False, True):
        text, spans = render(*trace_to_messages(
            randomize_trace(t, rng, ToolSchemaSampler(rng), force_schema=forced)))
        assert spans and "<tool_call>" not in text.split("<|im_end|>", 1)[1]
        assert ("# Tools" in text) == forced


def test_story_answers_are_grounded_in_what_was_already_said():
    """A later answer must only use facts the assistant stated earlier."""
    for t in generate_direct_traces(30, [STORY], seed=5):
        if "Glass Harbor" not in t:
            continue
        answers = " ".join(json.loads(m)["response"] for m in
                           re.findall(r"<assistant>(\{.*?\})</assistant>", t))
        for claim in ("Ines Varga", "reckless", "storm-lashed archipelago",
                      "sea adventure", "the navigator"):
            if claim.lower() in answers.lower():
                assert claim.lower() in (STORY["premise"] + STORY["setting"]
                                         + STORY["genre"] + "ines varga reckless"
                                         + "tomas reel the navigator").lower()
