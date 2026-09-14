"""Project arcs must be grounded in real workspace observations, render
through the normal pipeline, and read files back when they answer."""

import json
import random
import re

from agent.chatml import render, trace_to_messages
from agent.project_traces import generate_project_traces
from agent.system_prompts import apply_format_rule, apply_persona
from agent.format_rules import BY_ID
from agent.tool_schema import ToolSchemaSampler, randomize_trace

STORY = {"title": "The Glass Harbor", "genre": "sea adventure",
         "premise": "A salvage crew finds a map to a drowned city. They race a rival fleet to it.",
         "protagonist": {"name": "Ines Varga", "trait": "reckless"},
         "characters": [{"name": "Tomas Reel", "role": "the ship's cautious navigator"},
                        {"name": "Captain Orlo", "role": "the rival fleet's commander"}],
         "setting": "a storm-lashed archipelago",
         "chapters": [{"title": f"Chapter {i}", "summary": f"Summary of events {i}.",
                       "text": f"Opening line {i}. " + ("Prose. " * 120)} for i in range(1, 5)]}
PERSONA = {"name": "Zed", "system_prompt": "You are Zed, a gruff engineer. Grumble.",
           "wrappers": ["Grumble. {answer}. Back to work.", "Fine: {answer}."],
           "greetings": ["Hmph."], "signoffs": ["- Zed"]}


def test_arcs_are_grounded_and_use_read_back():
    traces = generate_project_traces([STORY], 6, seed=3, max_turns=30, min_turns=12)
    for t in traces:
        assert t.count("<user>") >= 12
        # every write is followed by the real observation from the workspace
        for m in re.finditer(r'"name":"write_file","arguments":\{"path":"([^"]+)","content":"', t):
            path = m.group(1)
            assert re.search(rf"<tool name=write_file>wrote \d+ bytes to {re.escape(path)}</tool>", t)
        assert "<tool name=read_file>" in t          # answers come from reading files
        assert t.endswith("\x03")


def test_arc_survives_randomization_and_renders_to_chatml():
    rng = random.Random(0)
    t = generate_project_traces([STORY], 1, seed=1, max_turns=16, min_turns=10)[0]
    t = randomize_trace(t, rng, ToolSchemaSampler(rng))
    messages, tools = trace_to_messages(t)
    names = {x["function"]["name"] for x in tools}
    assert len(names) >= 2
    text, spans = render(messages, tools)
    assert len(spans) >= 10 and "<tool_response>" in text


def test_persona_and_format_decoration():
    rng = random.Random(0)
    t = generate_project_traces([STORY], 1, seed=2, max_turns=8, min_turns=6)[0]
    t = randomize_trace(t, rng, ToolSchemaSampler(rng))
    p = apply_persona(t, PERSONA, rng)
    assert p.startswith('<system>{"tools":') and "<system>You are Zed" in p
    responses = [json.loads(m)["response"] for m in re.findall(r"<assistant>(\{.*?\})</assistant>", p)
                 if '"response"' in m]
    assert any("Grumble." in r or "Fine:" in r or "Hmph." in r or "- Zed" in r for r in responses)
    f = apply_format_rule(t, BY_ID["uppercase"], rng)
    assert f is not None and "<system>" in f
    for m in re.findall(r"<assistant>(\{.*?\})</assistant>", f):
        obj = json.loads(m)
        if "response" in obj:
            assert obj["response"] == obj["response"].upper()


def test_decorate_never_drops_a_trace():
    """A format rule that cannot be satisfied must leave the trace alone, not
    remove it - the dataset builder counts on one output per input."""
    from agent.system_prompts import decorate_traces
    rng = random.Random(5)
    traces = generate_project_traces([STORY], 12, seed=4, max_turns=10, min_turns=6)
    for _ in range(5):
        out = decorate_traces(traces, rng, [PERSONA], 0.4, 0.4)
        assert len(out) == len(traces)
        assert all(isinstance(t, str) and t for t in out)
