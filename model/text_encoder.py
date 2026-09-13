"""Text encoder."""

import math
import torch
import torch.nn as nn
from model.rope import RoPE


class TextEncoder(nn.Module):
    """Encodes token indices into continuous representations.

    In ``arch_version=2`` this is just a scaled embedding lookup: position is
    supplied by RoPE inside each attention layer, which is where it belongs.
    ``arch_version=1`` keeps the old whole-embedding rotation.
    """

    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 32768, dropout: float = 0.1,
                 use_rope: bool = True, arch_version: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope
        self.arch_version = arch_version
        self.max_len = max_len
        if use_rope:
            self.rope = RoPE(d_model, max_len)
        elif arch_version < 2:
            self.pos_encoder = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        if arch_version < 2:
            self.scale = nn.Parameter(torch.tensor(float(d_model) ** 0.5))
        else:
            # Fixed sqrt(d_model) scale. A *learnable* global scale on the
            # embedding is redundant once RMSNorm follows it, and it drifts.
            self.register_buffer("scale",
                                 torch.tensor(math.sqrt(float(d_model))),
                                 persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            offset: starting position, used when decoding with a KV cache.
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = self.embedding(x) * self.scale
        if self.use_rope:
            x = self.rope(x)
        elif self.arch_version < 2:
            x = x + self.pos_encoder[offset:offset + seq_len]
        return self.dropout(x)
