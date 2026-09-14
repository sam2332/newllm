"""Incremental generation for the JSON agent.

The previous loop re-ran a full forward pass over the whole sequence for every
new token, then re-decoded and re-parsed the entire string each time - O(n^2)
in both GPU work and Python. This module keeps a per-layer KV cache and only
inspects the newly generated suffix.

``generate_tokens`` is the primitive: it yields one token id at a time so a
server can stream, and reports timing so the Ollama envelope's duration and
count fields are real numbers. ``generate`` is the string convenience on top of
it and keeps its old signature.
"""

import time

import torch

from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.constrained import AssistantGrammar
from sampling import Sampler

EOT = "\x03"
ASSISTANT_CLOSE = "</assistant>"


def _single_id(tokenizer, text: str):
    """The id of ``text`` when it is exactly one token, else None."""
    ids = tokenizer.encode(text)
    return ids[0] if len(ids) == 1 else None


def _now_ns(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter_ns()


@torch.no_grad()
def generate_tokens(model, context_tokens: list, sampler: Sampler,
                    max_new: int = 120, device: str = "cuda",
                    tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                    stop_on_assistant_close: bool = True,
                    stop_texts: tuple = (EOT,),
                    grammar: AssistantGrammar = None,
                    stats: dict = None):
    """Yield newly generated token ids, one per step.

    The stop token itself is yielded before the generator returns, so a caller
    can see whether it stopped on ``</assistant>``, on EOT, or ran out of
    budget. ``stats`` (if given) receives ``prompt_eval_count``,
    ``prompt_eval_ns``, ``eval_count`` and ``eval_ns``.

    Stop detection is by token id wherever a stop string is a single token
    (``</assistant>`` is atomic, EOT is byte 3). Multi-token stop strings are
    matched against a rolling decoded tail, so the cost per step no longer
    grows with the length of the output.
    """
    if grammar is not None:
        grammar.reset()
    model.eval()
    use_cache = getattr(model, "arch_version", 1) >= 2

    # Fail loudly in Python rather than as a CUDA device-side assert, which
    # would corrupt the process for every later request.
    model_vocab = getattr(model, "vocab_size", None)
    if model_vocab is not None and context_tokens:
        worst = max(context_tokens)
        if worst >= model_vocab:
            raise ValueError(
                f"token id {worst} exceeds the model vocabulary ({model_vocab}). "
                f"Build the tokenizer with AgentTokenizer(max_vocab={model_vocab}) "
                f"for this checkpoint."
            )

    stop_ids = set()
    tail_stops = []
    for text in stop_texts:
        sid = _single_id(tokenizer, text)
        if sid is not None:
            stop_ids.add(sid)
        else:
            tail_stops.append(text)
    if stop_on_assistant_close:
        sid = _single_id(tokenizer, ASSISTANT_CLOSE)
        if sid is not None:
            stop_ids.add(sid)
        else:
            tail_stops.append(ASSISTANT_CLOSE)
    tail_len = max((len(s) for s in tail_stops), default=0)
    tail = ""

    t0 = _now_ns(device)
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    caches = model.new_caches() if use_cache else None
    if use_cache:
        out = model(input_ids, caches=caches)
    else:
        out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() == 4:
        logits = logits[:, :, 0, :]
    t1 = _now_ns(device)
    if stats is not None:
        stats["prompt_eval_count"] = len(context_tokens)
        stats["prompt_eval_ns"] = t1 - t0

    max_len = getattr(model, "max_len", 768)
    # Repetition penalty needs the running token history; keep it on-device
    # instead of rebuilding a tensor from a Python list every step.
    history = torch.tensor(context_tokens, dtype=torch.long, device=device)
    produced = 0

    for _ in range(max_new):
        step_logits = logits[:, -1, :]
        if grammar is not None:
            step_logits = grammar.mask(step_logits)
        next_token = sampler.sample(step_logits, history)
        if grammar is not None:
            grammar.advance(next_token)
        history = torch.cat(
            [history, torch.tensor([next_token], device=device)])
        produced += 1
        yield next_token

        if grammar is not None and grammar.done:
            break
        if next_token in stop_ids:
            break
        if tail_stops:
            tail = (tail + tokenizer.decode([next_token]))[-tail_len:]
            if any(s in tail for s in tail_stops):
                break

        step_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        if use_cache:
            if caches[0]["k"].size(2) >= max_len:
                # Context is full. Slide the window by re-prefilling on the
                # most recent max_len//2 tokens rather than silently erroring.
                keep = history[-(max_len // 2):]
                caches = model.new_caches()
                out = model(keep.unsqueeze(0), caches=caches)
            else:
                out = model(step_input, caches=caches)
        else:
            seq = history[-max_len:].unsqueeze(0)
            out = model(seq)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.dim() == 4:
            logits = logits[:, :, 0, :]

    if stats is not None:
        stats["eval_count"] = produced
        stats["eval_ns"] = _now_ns(device) - t1


def generate(model, context_tokens: list, sampler: Sampler,
             max_new: int = 120, device: str = "cuda",
             tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
             stop_on_assistant_close: bool = True,
             stop_texts: tuple = (EOT,),
             grammar: AssistantGrammar = None) -> str:
    """Generate a continuation and return only the newly generated text.

    Stops early at ``</assistant>`` so the caller can execute a tool without
    burning the rest of the token budget.

    If ``grammar`` is given, logits are masked to the tokens the assistant-JSON
    grammar permits, so the output is structurally valid by construction.
    """
    return tokenizer.decode(list(generate_tokens(
        model, context_tokens, sampler, max_new=max_new, device=device,
        tokenizer=tokenizer, stop_on_assistant_close=stop_on_assistant_close,
        stop_texts=stop_texts, grammar=grammar)))
