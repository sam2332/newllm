"""Rotary Position Embedding (RoPE).

RoPE let brain understand order and also scale to very long story.
Each pair of number get rotated by angle based on position.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32768, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        positions = torch.arange(max_len).float()
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (max_len, d_model/2)
        self.register_buffer("cos", angles.cos().repeat_interleave(2, dim=-1))
        self.register_buffer("sin", angles.sin().repeat_interleave(2, dim=-1))

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with RoPE applied
        """
        seq_len = x.size(1)
        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin
