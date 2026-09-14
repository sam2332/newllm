"""Schema randomization must rename parameters consistently: in the schema
block's properties/required AND in every tool call's arguments, never in
observations - and must leave a trace valid JSON throughout."""

import json
import random
import re

from agent.tool_schema import (PARAM_NAME_POOLS, ToolSchemaSampler,
                               randomize_trace)

TRACE = ('<user>What is 12 + 8, and what is the project?</user>\n'
         '<assistant>{"thought":"calc","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>\n'
         '<tool name=calc>20</tool>\n'
         '<assistant>{"thought":"memory","tool_call":{"name":"search_memory","arguments":{"key":"project"}}}</assistant>\n'
         '<tool name=search_memory>newllm</tool>\n'
         '<assistant>{"thought":"done","response":"20 and newllm"}</assistant>\x03')


def _calls(text):
    out = []
    for m in re.finditer(r"<assistant>(\{.*?\})</assistant>", text, re.S):
        obj = json.loads(m.group(1))
        if "tool_call" in obj:
            out.append(obj["tool_call"])
    return out


def test_parameters_renamed_consistently_across_schema_and_calls():
    seen_renamed = False
    for seed in range(40):
        rng = random.Random(seed)
        sampler = ToolSchemaSampler(rng, param_rename_prob=1.0)
        out = randomize_trace(TRACE, rng, sampler)
        schema = json.loads(re.match(r"<system>(.*?)</system>", out, re.S).group(1))
        by_name = {t["name"]: t for t in schema["tools"]}
        for call in _calls(out):
            tool = by_name[call["name"]]
            props = set(tool["parameters"]["properties"])
            assert set(call["arguments"]) <= props, (call, props)
            assert set(tool["parameters"]["required"]) <= props
            if set(call["arguments"]) != {"expr"} and set(call["arguments"]) != {"key"}:
                seen_renamed = True
        # observations never carry parameter names, and values are untouched
        assert "<tool name=" in out and ">20</tool>" in out and ">newllm</tool>" in out
        assert '"12 + 8"' in out and '"project"' in out
    assert seen_renamed


def test_rename_can_be_disabled():
    rng = random.Random(1)
    out = randomize_trace(TRACE, rng, ToolSchemaSampler(rng, param_rename_prob=0.0))
    assert all(set(c["arguments"]) in ({"expr"}, {"key"}) for c in _calls(out))


def test_pools_are_injective_enough_to_invert():
    for canonical, names in PARAM_NAME_POOLS.items():
        assert canonical in names and len(set(names)) == len(names)
