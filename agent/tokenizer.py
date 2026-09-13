"""Tokenizer for the tagged JSON agent protocol.

Normal ASCII remains byte-level so arbitrary tool arguments can be copied.
Frequent protocol markers are atomic tokens, reducing the structural burden on
the model without changing the externally visible JSON format.
"""


class AgentTokenizer:
    """Byte tokenizer with atomic tokens for stable agent-message structure."""

    # Vocabulary size of the original instruct-only protocol, kept so older
    # checkpoints can be identified and upgraded instead of rejected.
    LEGACY_VOCAB_SIZE = 271

    # NOTE: append-only. Existing ids 0-270 must keep their meaning so that
    # checkpoints trained on the 271-token vocabulary can be grown into this
    # one by appending embedding rows rather than retraining from scratch.
    SPECIAL_TOKENS = (
        "<assistant>",
        "</assistant>",
        "<user>",
        "</user>",
        "<tool name=calc>",
        "<tool name=now>",
        "<tool name=search_memory>",
        "<tool name=web_search>",
        "<tool name=finish>",
        "</tool>",
        '"thought":',
        '"tool_call":',
        '"response":',
        '"name":',
        '"arguments":',
        # --- added for the chat / multi-turn protocol (ids 271+) ---
        "<system>",
        "</system>",
    )

    def __init__(self, max_vocab: int = None):
        """``max_vocab`` restricts the tokenizer to a model's actual vocabulary.

        Because the special-token list is append-only, a checkpoint trained
        before ``<system>`` was added has only 271 embedding rows. Emitting id
        271 against it is an out-of-bounds gather, which surfaces as a CUDA
        device-side assert that also poisons the process. Passing the model's
        vocabulary size makes any newer special token fall back to plain byte
        encoding, so an older checkpoint still handles the text correctly - it
        just spends more tokens on it.
        """
        full = {token: 256 + index
                for index, token in enumerate(self.SPECIAL_TOKENS)}
        if max_vocab is None:
            self._token_to_id = full
            self.vocab_size = 256 + len(self.SPECIAL_TOKENS)
        else:
            if max_vocab < 256:
                raise ValueError("max_vocab must cover the 256 byte values")
            self._token_to_id = {t: i for t, i in full.items() if i < max_vocab}
            self.vocab_size = max_vocab
        self._id_to_token = {token_id: token for token, token_id in self._token_to_id.items()}
        self._ordered_special_tokens = sorted(self._token_to_id, key=len, reverse=True)

    def encode(self, text: str) -> list[int]:
        tokens = []
        position = 0
        while position < len(text):
            matched = None
            for special in self._ordered_special_tokens:
                if text.startswith(special, position):
                    matched = special
                    break
            if matched is not None:
                tokens.append(self._token_to_id[matched])
                position += len(matched)
            else:
                tokens.append(min(ord(text[position]), 255))
                position += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        parts = []
        for token in tokens:
            if token in self._id_to_token:
                parts.append(self._id_to_token[token])
            else:
                parts.append(chr(min(token, 255)))
        return "".join(parts)


    @property
    def eot(self) -> str:
        """End-of-turn marker: the assistant stops and yields to the user."""
        return "\x03"


DEFAULT_AGENT_TOKENIZER = AgentTokenizer()
