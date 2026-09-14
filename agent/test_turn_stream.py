"""Streamed deltas must concatenate to exactly the parsed field values."""

import json

from agent.ollama_context import to_model_text
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from agent.turn import parse_json_turn, sampler_from_options
from agent.turn_stream import JsonStringUnescaper, TurnTracker

HARD = 'q "quoted" back\\slash nl\n tab\t é € 😀 end'


def _stream(internal: dict, ensure_ascii: bool, allowed=None):
    text = ("<assistant>" + json.dumps(internal, ensure_ascii=ensure_ascii, separators=(",", ":"))
            + "</assistant>")
    tracker = TurnTracker(TOK, allowed)
    got = {"thinking": "", "content": ""}
    for tid in TOK.encode(to_model_text(text)):
        for stream, delta in tracker.feed(tid):
            got[stream] += delta
    return tracker, got, text


def test_response_deltas_match_parsed_values_escaped_and_raw():
    internal = {"thought": HARD, "response": "R " + HARD}
    for ensure_ascii in (True, False):
        tracker, got, text = _stream(internal, ensure_ascii)
        assert not tracker.desynced
        assert tracker.kind == "response"
        assert got["thinking"] == internal["thought"]
        assert got["content"] == internal["response"]
        parsed = parse_json_turn(to_model_text(text))
        assert parsed.kind == "response"
        assert parsed.response == internal["response"]
        assert tracker.remainder("content", parsed.response) == ""


def test_tool_call_turn_streams_thought_only():
    internal = {"thought": "use it", "tool_call": {"name": "add_numbers",
                                                  "arguments": {"a": 1, "b": 2}}}
    tracker, got, text = _stream(internal, True, allowed=["add_numbers"])
    assert tracker.kind == "tool_call"
    assert got["thinking"] == "use it"
    assert got["content"] == ""
    parsed = parse_json_turn(to_model_text(text), allowed_names=["add_numbers"])
    assert parsed.kind == "tool_call"
    assert parsed.tool_call == {"name": "add_numbers", "arguments": {"a": 1, "b": 2}}


def test_unknown_tool_is_malformed_not_an_error():
    internal = {"thought": "t", "tool_call": {"name": "memo_get", "arguments": {}}}
    text = "<assistant>" + json.dumps(internal, separators=(",", ":")) + "</assistant>"
    parsed = parse_json_turn(text, allowed_names=["add_numbers"])
    assert parsed.kind == "malformed"
    assert parsed.fallback == "unknown_tool"
    assert parsed.tool_call["name"] == "memo_get"


def test_invalid_json_and_truncation_are_reported():
    assert parse_json_turn("<assistant>{not json</assistant>").fallback == "invalid_json"
    assert parse_json_turn('<assistant>{"thought":"x","response":"unfinished').fallback == "length"


def test_desync_stops_deltas_and_remainder_recovers_everything():
    # A special token where the grammar expects a string byte leaves the grammar.
    text = '<assistant>{"thought":"ab' + '"name":' + 'cd","response":"zz"}</assistant>'
    tracker = TurnTracker(TOK)
    got = ""
    for tid in TOK.encode(text):
        for stream, delta in tracker.feed(tid):
            if stream == "content":
                got += delta
    assert tracker.desynced
    assert got == ""
    assert tracker.remainder("content", "zz") == "zz"


def test_unescaper_holds_partial_sequences():
    u = JsonStringUnescaper()
    assert u.push("a\\") == "a"
    assert u.push("u00") == ""
    assert u.push("e9") == "é"
    assert u.push("\\ud83d") == ""
    assert u.push("\\ude00") == "😀"
    raw = "é".encode("utf-8").decode("latin-1")
    assert u.push(raw[0]) == ""
    assert u.push(raw[1]) == "é"
    assert u.flush() == ""


def test_sampler_options_map_greedy_and_sampling():
    g = sampler_from_options({"temperature": 0})
    assert g.top_k == 1
    s = sampler_from_options({"temperature": 0.7, "top_p": 0.8, "top_k": 5,
                              "repeat_penalty": 1.1})
    assert (s.temperature, s.top_p, s.top_k, s.repetition_penalty) == (0.7, 0.8, 5, 1.1)
