"""Rotary Position Embedding (RoPE).

RoPE let brain understand order and also scale to very long story.
Each pair of number get rotated by angle based on position.

Two entry points:
  * ``RoPE``        - legacy whole-embedding rotation (``arch_version=1``).
  * ``apply_rope``  - correct per-head rotation of Q/K inside attention
                      (``arch_version=2``). This is what real RoPE does; the
                      legacy path rotates the embedding once at the input and
                      the first LayerNorm then scrubs most of the signal.
"""

import torch
import torch.nn as nn


class RoPECache(nn.Module):
    """Precomputed cos/sin tables for per-head rotation of Q and K."""

    def __init__(self, head_dim: int, max_len: int = 32768, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (max_len, head_dim/2)
        # Interleaved layout: matches ``rotate_half`` below, which pairs
        # (x[0], x[1]), (x[2], x[3]), ... rather than splitting in halves.
        self.register_buffer("cos", angles.cos().repeat_interleave(2, dim=-1),
                             persistent=False)
        self.register_buffer("sin", angles.sin().repeat_interleave(2, dim=-1),
                             persistent=False)

    def get(self, seq_len: int, offset: int = 0, device=None, dtype=None):
        """Return (cos, sin) broadcastable to (batch, heads, seq_len, head_dim)."""
        end = offset + seq_len
        if end > self.cos.size(0):
            raise ValueError(
                f"RoPE cache holds {self.cos.size(0)} positions, need {end}"
            )
        cos = self.cos[offset:end].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:end].unsqueeze(0).unsqueeze(0)
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved rotation partner: (x0, x1, x2, x3) -> (-x1, x0, -x3, x2)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Rotate query and key tensors of shape (batch, heads, seq_len, head_dim).

    Rotating Q and K (rather than the input embedding) is what makes attention
    scores depend only on *relative* position, which is the whole point of RoPE
    and what exact-span copying needs.
    """
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


class RoPE(nn.Module):
    """Legacy whole-embedding RoPE, kept so ``arch_version=1`` still runs."""

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
        return rotate_half(x)

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
