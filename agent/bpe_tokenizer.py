"""Byte-level BPE tokenizer with the same interface as ``AgentTokenizer``.

Built on Hugging Face ``tokenizers`` and configured exactly like Qwen2's
``tokenizer.json`` - GPT-2 byte-level BPE behind Qwen's pre-tokenization
regex - because that is a configuration llama.cpp reproduces bit for bit
(``tokenizer.ggml.pre = "qwen2"``). Qwen's regex also splits every digit into
its own token, which is what keeps BPE from merging ``4710`` into one symbol
and losing the arithmetic.

Special tokens are single ids, act as hard boundaries for merges, and decode
back to their literal text, so a template like ``<|im_start|>assistant`` is
always the same tokens no matter what surrounds it.
"""

import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers import Regex

# Qwen2 / Qwen3 pre-tokenization regex, verbatim from their tokenizer.json.
QWEN2_PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# ChatML / Qwen3 protocol markers. Order is the id order after the BPE vocab.
CHATML_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


def build_empty(vocab_size: int) -> tuple:
    """A trainer + untrained tokenizer with the Qwen2 configuration."""
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_PRETOKENIZER_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(CHATML_SPECIAL_TOKENS),
        special_tokens=[], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    return tok, trainer


class BPEAgentTokenizer:
    """Drop-in for ``AgentTokenizer``: ``encode``, ``decode``, ``vocab_size``,
    ``eot``, ``SPECIAL_TOKENS``."""

    SPECIAL_TOKENS = CHATML_SPECIAL_TOKENS
    kind = "bpe"

    def __init__(self, path: str):
        self.path = path
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size(with_added_tokens=True)
        self._special_ids = {t: self._tok.token_to_id(t) for t in self.SPECIAL_TOKENS}
        missing = [t for t, i in self._special_ids.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer at {path} lacks special tokens {missing}")
        self._id_to_special = {i: t for t, i in self._special_ids.items()}

    # -- the AgentTokenizer interface -------------------------------------
    def encode(self, text: str) -> list:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list) -> str:
        return self._tok.decode(list(tokens), skip_special_tokens=False)

    @property
    def eot(self) -> str:
        """End-of-turn marker in ChatML."""
        return "<|im_end|>"

    def special_id(self, token: str) -> int:
        return self._special_ids[token]

    def is_special(self, token_id: int) -> bool:
        return token_id in self._id_to_special

    def spec(self) -> dict:
        return {"kind": "bpe", "path": self.path, "vocab_size": self.vocab_size}


def train_bpe(texts, vocab_size: int, out_path: str) -> BPEAgentTokenizer:
    """Train on an iterable of strings and save an HF ``tokenizer.json``."""
    tok, trainer = build_empty(vocab_size)
    tok.train_from_iterator(texts, trainer)
    # Added last, so ids 0..N-1 are the BPE vocab and the specials sit at the
    # top - the layout every GGUF exporter expects.
    tok.add_special_tokens(list(CHATML_SPECIAL_TOKENS))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    meta = {"vocab_size": tok.get_vocab_size(with_added_tokens=True),
            "special_tokens": list(CHATML_SPECIAL_TOKENS),
            "pre_tokenizer": "qwen2", "model": "gpt2-bytelevel-bpe"}
    json.dump(meta, open(out_path.replace(".json", ".meta.json"), "w"), indent=1)
    return BPEAgentTokenizer(out_path)
