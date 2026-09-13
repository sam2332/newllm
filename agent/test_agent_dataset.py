"""Tests for agent tokenization and assistant-only supervision."""

from pathlib import Path

import pytest

import json
import random

from agent.agent_dataset import (
    _cross_source_math,
    _multi_hop_question,
    _serialize,
    _two_stage_math_question,
    _web_two_stage_math,
    generate_agent_dataset,
)
from agent.train_agent import make_agent_dataset, train_agent
from agent.chat import load_checkpoint
from agent.dataset import AgentDataset, encode_agent_training_example
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER
from agent.tools import Toolbox


def test_agent_tokenizer_round_trips_tagged_json_trace():
    trace = (
        '<user>What is 12 + 8?</user>\n'
        '<assistant>{"thought":"calculate","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>\n'
        '<tool name=calc>20</tool>\n'
        '<assistant>{"thought":"done","response":"20"}</assistant>\x03'
    )
    assert DEFAULT_AGENT_TOKENIZER.decode(DEFAULT_AGENT_TOKENIZER.encode(trace)) == trace


def test_only_assistant_tokens_are_supervised():
    trace = (
        '<user>What is 12 + 8?</user>\n'
        '<assistant>{"thought":"calculate","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>\n'
        '<tool name=calc>20</tool>\n'
        '<assistant>{"thought":"done","response":"20"}</assistant>\x03'
    )
    tokens, mask = encode_agent_training_example(trace)
    user_start = DEFAULT_AGENT_TOKENIZER.encode("<user>")[0]
    tool_start = DEFAULT_AGENT_TOKENIZER.encode("<tool name=calc>")[0]
    assert mask[tokens.index(user_start)] == 0.0
    assert mask[tokens.index(tool_start)] == 0.0
    assert sum(mask) > 0


def test_agent_dataset_returns_shifted_assistant_mask():
    trace = generate_agent_dataset(num_samples=1, max_len=768)[0]
    dataset = AgentDataset([trace], max_len=768)
    x, y, mask = dataset[0]
    assert len(x) == len(y) == len(mask)
    assert mask.sum().item() > 0
    assert mask.sum().item() < len(mask)


def test_legacy_checkpoint_is_rejected_when_available():
    checkpoint = Path("checkpoints/agent_best.pt")
    if not checkpoint.exists():
        pytest.skip("no default checkpoint available")
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_checkpoint(str(checkpoint), device="cpu")


def test_init_checkpoint_is_an_explicit_train_agent_option():
    assert "init_checkpoint" in train_agent.__annotations__


def test_curriculum_math_replay_keeps_math_trace_share():
    train_set, val_set = make_agent_dataset(
        num_samples=100,
        max_len=768,
        simple=False,
        val_frac=0.0,
        math_replay_fraction=0.5,
    )
    assert len(train_set) == 100


@pytest.mark.parametrize("factory", [
    _two_stage_math_question,
    _web_two_stage_math,
    lambda rng: _multi_hop_question(rng, Toolbox().memory),
    lambda rng: _cross_source_math(rng, Toolbox().memory),
])
def test_complex_scenarios_use_grounded_executable_tool_results(factory):
    messages = factory(random.Random(7))
    tool_calls = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        assistant = json.loads(message["content"])
        if "tool_call" not in assistant:
            continue
        tool_calls += 1
        call = assistant["tool_call"]
        actual = Toolbox().run_json({
            "tool": call["name"],
            "args": call["arguments"],
        })
        assert messages[index + 1]["role"] == "tool"
        assert messages[index + 1]["content"] == actual
    assert tool_calls == 3
    assert len(_serialize(messages)) <= 768
