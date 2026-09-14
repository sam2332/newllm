"""Generate exactly one assistant turn from an assembled context.

This is the unit of work behind ``POST /api/chat``: given the context, produce
the next assistant message and *return it* - a tool call for the client to
execute, or a final response. Tools are never executed here. It is also the
single code path every evaluation should go through, so that what is measured
is what is served.

Two wire protocols share it:
  * ``"json"``   - the legacy ``<assistant>{json}</assistant>`` form of the
                   byte-level checkpoints;
  * ``"chatml"`` - the Qwen3 template the retrained model uses
                   (``agent/chatml.py``).

``run_agent`` in ``agent_loop.py`` remains for the server-side loop and the
older evals; it executes tools itself and prefers the raw tool result over the
model's prose (``_select_final_answer``), neither of which is right for a
client that owns its tools.
"""

from dataclasses import dataclass

import torch

from agent.agent_loop import _extract_assistant_json
from agent.chatml import parse_assistant, IM_END, IM_START
from agent.constrained import AssistantGrammar
from agent.generate import generate_tokens, EOT, ASSISTANT_CLOSE
from agent.ollama_context import from_model_text
from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.turn_stream import TurnTracker, ChatMLTracker
from sampling import Sampler

CHATML_STOPS = (IM_END, IM_START, "<|endoftext|>")


@dataclass
class TurnResult:
    kind: str                       # "tool_call" | "response" | "malformed"
    thought: str = ""
    response: str = ""              # final text, or raw text when malformed
    tool_call: dict = None          # {"name": ..., "arguments": {...}}
    raw: str = ""                   # decoded model output (unicode)
    done_reason: str = "stop"       # "stop" | "length"
    fallback: str = None            # "unknown_tool" | "invalid_json" | "length"
    prompt_tokens: int = 0
    eval_tokens: int = 0
    prompt_eval_ns: int = 0
    eval_ns: int = 0


@dataclass
class TurnEvent:
    type: str                       # "thinking" | "content" | "tool_call" | "done"
    text: str = ""
    result: TurnResult = None


def protocol_for(tokenizer) -> str:
    """A BPE tokenizer means the ChatML model; the byte tokenizer, legacy JSON."""
    return "chatml" if getattr(tokenizer, "kind", "byte") == "bpe" else "json"


def sampler_from_options(options: dict) -> Sampler:
    """Map Ollama ``options`` onto the repo's Sampler.

    ``temperature <= 0`` means greedy, done the way ``run_agent`` does it
    (near-zero temperature with top_k=1) so results match the evals exactly.
    The non-greedy defaults mirror ``run_agent``'s sampling defaults.
    """
    options = options or {}
    temperature = float(options.get("temperature", 0.0))
    if temperature <= 0:
        return Sampler(temperature=0.01, top_k=1, top_p=1.0,
                       repetition_penalty=1.0)
    return Sampler(temperature=temperature,
                   top_k=int(options.get("top_k", 20)),
                   top_p=float(options.get("top_p", 0.9)),
                   min_p=float(options.get("min_p", 0.0)),
                   repetition_penalty=float(options.get("repeat_penalty", 1.0)))


def make_grammar(tokenizer, allowed_names, constrained: bool, protocol: str = "json"):
    # The grammar is byte-level and JSON-shaped; it does not apply to ChatML.
    if not constrained or protocol != "json":
        return None
    names = list(allowed_names or [])
    return AssistantGrammar(names, tokenizer, allow_tool_call=bool(names))


def _seed(seed):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _strip_tags(text: str) -> str:
    return (text.replace("<assistant>", "").replace("</assistant>", "")
            .replace(EOT, "").strip())


def parse_json_turn(raw_model_text: str, allowed_names=None) -> TurnResult:
    """Interpret the model's output under the legacy JSON protocol.

    Malformed output is never an error: it comes back as plain content so the
    client gets *something*, tagged with why it could not be interpreted.
    """
    text = from_model_text(raw_model_text)
    closed = ASSISTANT_CLOSE in text or EOT in text
    _, _, parsed = _extract_assistant_json(text, 0)
    if parsed is None:
        return TurnResult(kind="malformed", response=_strip_tags(text), raw=text,
                          fallback="length" if not closed else "invalid_json")
    thought = parsed.get("thought", "")
    if "tool_call" in parsed:
        call = parsed["tool_call"]
        name = call.get("name", "")
        if allowed_names is not None and name not in allowed_names:
            return TurnResult(kind="malformed", thought=thought,
                              response=_strip_tags(text), raw=text,
                              tool_call={"name": name,
                                         "arguments": call.get("arguments", {})},
                              fallback="unknown_tool")
        return TurnResult(kind="tool_call", thought=thought, raw=text,
                          tool_call={"name": name,
                                     "arguments": call.get("arguments", {})})
    return TurnResult(kind="response", thought=thought,
                      response=parsed.get("response", ""), raw=text)


def parse_chatml_turn(text: str, allowed_names=None) -> TurnResult:
    """Interpret the model's output under the Qwen3 template."""
    closed = any(s in text for s in CHATML_STOPS)
    p = parse_assistant(text)
    if p["tool_calls"]:
        call = p["tool_calls"][0]
        if allowed_names is not None and call["name"] not in allowed_names:
            return TurnResult(kind="malformed", thought=p["thought"],
                              response=p["content"] or text.split(IM_END)[0].strip(),
                              raw=text, tool_call=call, fallback="unknown_tool")
        return TurnResult(kind="tool_call", thought=p["thought"], raw=text,
                          tool_call=call)
    if p["malformed"]:
        return TurnResult(kind="malformed", thought=p["thought"],
                          response=text.split(IM_END)[0].strip(), raw=text,
                          fallback="length" if not closed else "invalid_json")
    return TurnResult(kind="response", thought=p["thought"],
                      response=p["content"], raw=text,
                      fallback=None if closed else "length")


def _finish(result: TurnResult, ids: list, max_new: int, stats: dict, closed: bool):
    result.prompt_tokens = stats.get("prompt_eval_count", 0)
    result.eval_tokens = stats.get("eval_count", len(ids))
    result.prompt_eval_ns = stats.get("prompt_eval_ns", 0)
    result.eval_ns = stats.get("eval_ns", 0)
    if result.fallback == "length" or (len(ids) >= max_new and not closed):
        result.done_reason = "length"
    return result


def _run(model, context_tokens, sampler, tokenizer, device, max_new, stop_texts,
         grammar, protocol, stats):
    if protocol == "chatml":
        return generate_tokens(model, context_tokens, sampler, max_new=max_new,
                               device=device, tokenizer=tokenizer,
                               stop_on_assistant_close=False,
                               stop_texts=tuple(stop_texts) + CHATML_STOPS,
                               grammar=None, stats=stats)
    return generate_tokens(model, context_tokens, sampler, max_new=max_new,
                           device=device, tokenizer=tokenizer,
                           stop_texts=stop_texts, grammar=grammar, stats=stats)


def _parse(raw: str, allowed_names, protocol: str):
    if protocol == "chatml":
        return parse_chatml_turn(raw, allowed_names), any(s in raw for s in CHATML_STOPS)
    result = parse_json_turn(raw, allowed_names)
    return result, (ASSISTANT_CLOSE in result.raw or EOT in result.raw)


def generate_turn(model, context_tokens: list, *, sampler: Sampler,
                  tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                  device: str = "cuda", max_new: int = 4096,
                  stop_texts: tuple = (EOT,), grammar: AssistantGrammar = None,
                  seed=None, allowed_names=None,
                  protocol: str = "json") -> TurnResult:
    """One assistant turn, no tool execution. Caller holds any GPU lock."""
    _seed(seed)
    stats = {}
    ids = list(_run(model, context_tokens, sampler, tokenizer, device, max_new,
                    stop_texts, grammar, protocol, stats))
    result, closed = _parse(tokenizer.decode(ids), allowed_names, protocol)
    return _finish(result, ids, max_new, stats, closed)


def stream_turn(model, context_tokens: list, *, sampler: Sampler,
                tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                device: str = "cuda", max_new: int = 4096,
                stop_texts: tuple = (EOT,), grammar: AssistantGrammar = None,
                seed=None, allowed_names=None, protocol: str = "json"):
    """Like ``generate_turn`` but yields ``TurnEvent``s as text arrives.

    Content and thinking deltas are emitted only while the output is inside
    the protocol's grammar; a tool call is emitted whole; the final ``done``
    event carries the complete ``TurnResult`` plus any text the deltas missed.
    """
    _seed(seed)
    stats = {}
    tracker = (ChatMLTracker(tokenizer) if protocol == "chatml"
               else TurnTracker(tokenizer, allowed_names))
    ids = []
    for token_id in _run(model, context_tokens, sampler, tokenizer, device,
                         max_new, stop_texts, grammar, protocol, stats):
        ids.append(token_id)
        for stream, text in tracker.feed(token_id):
            yield TurnEvent(type=stream, text=text)
    result, closed = _parse(tokenizer.decode(ids), allowed_names, protocol)
    result = _finish(result, ids, max_new, stats, closed)
    # Reconcile: anything the deltas did not carry goes out now.
    if result.kind in ("response", "malformed"):
        rest = tracker.remainder("content", result.response)
        if rest:
            yield TurnEvent(type="content", text=rest)
    if result.thought:
        rest = tracker.remainder("thinking", result.thought)
        if rest:
            yield TurnEvent(type="thinking", text=rest)
    if result.kind == "tool_call":
        yield TurnEvent(type="tool_call", result=result)
    yield TurnEvent(type="done", result=result)
