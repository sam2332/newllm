"""The BPE tokenizer must be a drop-in for the byte tokenizer and must not
change how a trace tokenizes when it is cut at the supervised-span
boundaries - otherwise the loss mask lands on different tokens than the
model sees at inference."""

import json
import os
import tempfile

import pytest

from agent.bpe_tokenizer import (BPEAgentTokenizer, CHATML_SPECIAL_TOKENS,
                                 train_bpe)
from agent.chatml import render
from agent.dataset import encode_with_spans

CORPUS = [
    "What is 12 + 8? The answer is 20.",
    "Look up the capital of France, then multiply 3530 by 7.",
    '<tool_call>\n{"name": "evaluate_sum", "arguments": {"expr": "4710 - 10"}}\n</tool_call>',
    "You are Zed, a gruff starship engineer. Reply in one sentence.",
    "héllo € 😀 — unicode survives round trips",
] * 40


@pytest.fixture(scope="module")
def tok():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "tiny.json")
    return train_bpe(iter(CORPUS), 600, path)


def test_specials_are_single_ids_at_the_top_of_the_vocab(tok):
    ids = [tok.special_id(t) for t in CHATML_SPECIAL_TOKENS]
    assert len(set(ids)) == len(ids)
    assert min(ids) == tok.vocab_size - len(CHATML_SPECIAL_TOKENS)
    for t in CHATML_SPECIAL_TOKENS:
        assert tok.encode(t) == [tok.special_id(t)]
        assert tok.decode([tok.special_id(t)]) == t


def test_round_trip_including_unicode_and_specials(tok):
    text = "<|im_start|>user\nhéllo € 😀 4710<|im_end|>\n"
    assert tok.decode(tok.encode(text)) == text


def test_digits_are_one_token_each(tok):
    ids = tok.encode("4710")
    assert len(ids) == 4
    assert [tok.decode([i]) for i in ids] == ["4", "7", "1", "0"]


def test_eot_is_im_end(tok):
    assert tok.eot == "<|im_end|>"
    assert tok.encode(tok.eot) == [tok.special_id("<|im_end|>")]


def test_span_cut_does_not_change_tokenization(tok):
    messages = [
        {"role": "system", "content": "You are Zed."},
        {"role": "user", "content": "What is 12 + 8?"},
        {"role": "assistant", "content": "", "reasoning_content": "Add them.",
         "tool_calls": [{"function": {"name": "evaluate_sum",
                                      "arguments": {"expr": "12 + 8"}}}]},
        {"role": "tool", "content": "20"},
        {"role": "assistant", "content": "The answer is 20.",
         "reasoning_content": "Done."},
    ]
    tools = [{"type": "function", "function": {"name": "evaluate_sum",
                                               "description": "Sum.",
                                               "parameters": {"type": "object",
                                                              "properties": {}}}}]
    for style in ("hf", "ollama", "compact"):
        text, spans = render(messages, tools, style=style)
        tokens, mask = encode_with_spans(text, spans, tok, 100000)
        assert tokens == tok.encode(text), style
        supervised = tok.decode([t for t, m in zip(tokens, mask) if m > 0])
        assert supervised.endswith("The answer is 20.<|im_end|>")
        assert "<|im_start|>user" not in supervised
        assert supervised.count("<|im_end|>") == 2


def test_spec_round_trips_through_the_registry(tok):
    from agent.tokenizer_registry import load_tokenizer, spec_for, tokenizer_key
    spec = spec_for(tok)
    assert spec["kind"] == "bpe" and spec["vocab_size"] == tok.vocab_size
    again = load_tokenizer(spec)
    assert isinstance(again, BPEAgentTokenizer)
    assert again.encode("12 + 8") == tok.encode("12 + 8")
    assert tokenizer_key(spec) != tokenizer_key("byte")
    assert load_tokenizer("byte").vocab_size == 273


def test_chatml_tracker_deltas_match_the_parser(tok):
    from agent.chatml import parse_assistant
    from agent.turn_stream import ChatMLTracker
    turns = [
        "<think>\nAdd them first.\n</think>\n\n The answer is 20. \n<|im_end|>",
        "<think>\nplan\n</think>\n\n<tool_call>\n{\"name\": \"evaluate_sum\", "
        "\"arguments\": {\"expr\": \"12 + 8\"}}\n</tool_call><|im_end|>",
        "Hello there, no thinking.<|im_end|>",
    ]
    for turn in turns:
        tracker = ChatMLTracker(tok)
        got = {"thinking": "", "content": ""}
        for tid in tok.encode(turn):
            for stream, delta in tracker.feed(tid):
                got[stream] += delta
        parsed = parse_assistant(turn)
        assert got["thinking"] == parsed["thought"], turn
        assert got["content"] == parsed["content"], turn
        assert tracker.remainder("content", parsed["content"]) == ""
        if parsed["tool_calls"]:
            assert tracker.kind == "tool_call" and got["content"] == ""
