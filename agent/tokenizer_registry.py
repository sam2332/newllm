"""Pick the tokenizer a checkpoint or a run was built with.

A checkpoint records ``config["tokenizer"]`` (see ``Transformer._get_config``).
Absent key = the byte tokenizer, restricted to the checkpoint's vocabulary,
which keeps every checkpoint trained before this module loadable.
"""

import os

from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER


def load_tokenizer(spec, checkpoint_dir: str = None):
    """``spec`` is None, "byte", a path to an HF tokenizer.json, or a dict
    ``{"kind": "bpe", "path": ..., "vocab_size": ...}``."""
    if spec is None or spec == "byte":
        return DEFAULT_AGENT_TOKENIZER
    if isinstance(spec, dict):
        if spec.get("kind", "bpe") == "byte":
            return AgentTokenizer(max_vocab=spec.get("vocab_size"))
        path = spec.get("path")
    else:
        path = str(spec)
    # A checkpoint directory carries its own copy of the tokenizer, so the
    # original path going stale does not strand the weights.
    candidates = [path]
    if checkpoint_dir:
        candidates.insert(0, os.path.join(checkpoint_dir, "tokenizer.json"))
    from agent.bpe_tokenizer import BPEAgentTokenizer
    for p in candidates:
        if p and os.path.exists(p):
            return BPEAgentTokenizer(p)
    raise FileNotFoundError(f"tokenizer not found at any of {candidates}")


def spec_for(tokenizer) -> dict:
    if hasattr(tokenizer, "spec"):
        return tokenizer.spec()
    return {"kind": "byte", "vocab_size": tokenizer.vocab_size}


def tokenizer_key(spec) -> str:
    """Short stable string for cache keys."""
    if spec is None or spec == "byte":
        return "byte"
    if isinstance(spec, dict):
        return f"{spec.get('kind','bpe')}:{os.path.basename(str(spec.get('path')))}:{spec.get('vocab_size')}"
    return f"bpe:{os.path.basename(str(spec))}"
