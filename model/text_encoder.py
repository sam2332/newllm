"""Text encoder."""

import torch
import torch.nn as nn
from model.rope import RoPE


class TextEncoder(nn.Module):
    """Encodes token indices into continuous representations."""

    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 32768, dropout: float = 0.1,
                 use_rope: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(d_model, max_len)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = self.embedding(x) * self.scale
        if self.use_rope:
            x = self.rope(x)
        else:
            x = x + self.pos_encoder[:seq_len]
        return self.dropout(x)
