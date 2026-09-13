"""Incremental generation for the JSON agent.

The previous loop re-ran a full forward pass over the whole sequence for every
new token, then re-decoded and re-parsed the entire string each time - O(n^2)
in both GPU work and Python. This module keeps a per-layer KV cache and only
inspects the newly generated suffix.
"""

import torch

from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.constrained import AssistantGrammar
from sampling import Sampler

EOT = "\x03"
ASSISTANT_CLOSE = "</assistant>"


@torch.no_grad()
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

    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    caches = model.new_caches() if use_cache else None

    if use_cache:
        out = model(input_ids, caches=caches)
    else:
        out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() == 4:
        logits = logits[:, :, 0, :]

    max_len = getattr(model, "max_len", 768)
    produced = []
    # Repetition penalty needs the running token history; keep it on-device
    # instead of rebuilding a tensor from a Python list every step.
    history = torch.tensor(context_tokens, dtype=torch.long, device=device)

    for _ in range(max_new):
        step_logits = logits[:, -1, :]
        if grammar is not None:
            step_logits = grammar.mask(step_logits)
        next_token = sampler.sample(step_logits, history)
        if grammar is not None:
            grammar.advance(next_token)
        produced.append(next_token)
        history = torch.cat(
            [history, torch.tensor([next_token], device=device)])

        if grammar is not None:
            if grammar.done:
                break
        else:
            text = tokenizer.decode(produced)
            if any(stop in text for stop in stop_texts):
                break
            if stop_on_assistant_close and ASSISTANT_CLOSE in text:
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

    return tokenizer.decode(produced)
