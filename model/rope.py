"""Rotary Position Embedding (RoPE).

RoPE let brain understand order and also scale to very long story.
Each pair of number get rotated by angle based on position.

Two entry points:
  * ``RoPE``        - legacy whole-embedding rotation (``arch_version=1``).
  * ``apply_rope``  - correct per-head rotation of Q/K inside attention
                      (``arch_version=2+``). This is what real RoPE does; the
                      legacy path rotates the embedding once at the input and
                      the first LayerNorm then scrubs most of the signal.

Two pair layouts exist and they are NOT interchangeable:
  * interleaved  - pairs (x0,x1), (x2,x3), ...  (``arch_version=2``)
  * half-split   - pairs (x0,x_{d/2}), (x1,x_{d/2+1}), ...  (``arch_version=3``)
The half-split layout is what llama.cpp calls NEOX rope and what Qwen/Llama
checkpoints use, so a model trained with it exports to GGUF as a plain tensor
copy. Mixing layouts silently destroys position information.
"""

import torch
import torch.nn as nn


class RoPECache(nn.Module):
    """Cos/sin tables for per-head rotation of Q and K, grown on demand."""

    def __init__(self, head_dim: int, max_len: int = 32768, base: float = 10000.0,
                 interleaved: bool = True):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        self.head_dim = head_dim
        self.base = base
        self.interleaved = interleaved
        self.register_buffer("inv_freq",
                             1.0 / (base ** (torch.arange(0, head_dim, 2).float()
                                             / head_dim)),
                             persistent=False)
        self._build(max_len)

    def _build(self, length: int):
        """(Re)build the cos/sin tables to cover ``length`` positions.

        The tables are a pure function of position, so growing them changes no
        result - a sequence longer than the current table produces exactly the
        logits it would have with a table preallocated to that size. This is
        what makes the context limit an allocation detail rather than an
        architectural one: nothing here is learned, and the buffers are
        non-persistent so they never enter a checkpoint.
        """
        positions = torch.arange(length, device=self.inv_freq.device).float()
        angles = torch.einsum("i,j->ij", positions, self.inv_freq)
        if self.interleaved:
            cos = angles.cos().repeat_interleave(2, dim=-1)
            sin = angles.sin().repeat_interleave(2, dim=-1)
        else:
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get(self, seq_len: int, offset: int = 0, device=None, dtype=None):
        """Return (cos, sin) broadcastable to (batch, heads, seq_len, head_dim)."""
        end = offset + seq_len
        if end > self.cos.size(0):
            # Grow geometrically rather than refusing. The model has no learned
            # position parameters, so the only thing a longer context costs is
            # this table plus the attention itself. Whether the model is any
            # GOOD that far out is a training-data question, not this one.
            self._build(max(end, self.cos.size(0) * 2))
        cos = self.cos[offset:end].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:end].unsqueeze(0).unsqueeze(0)
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved rotation partner: (x0, x1, x2, x3) -> (-x1, x0, -x3, x2)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def rotate_half_split(x: torch.Tensor) -> torch.Tensor:
    """Half-split (NEOX) rotation partner: (a, b) -> (-b, a) over halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# Kept under the old name for the legacy whole-embedding path.
rotate_half = rotate_half_interleaved


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor,
               interleaved: bool = True) -> tuple:
    """Rotate query and key tensors of shape (batch, heads, seq_len, head_dim).

    Rotating Q and K (rather than the input embedding) is what makes attention
    scores depend only on *relative* position, which is the whole point of RoPE
    and what exact-span copying needs.
    """
    rot = rotate_half_interleaved if interleaved else rotate_half_split
    q_out = q * cos + rot(q) * sin
    k_out = k * cos + rot(k) * sin
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
