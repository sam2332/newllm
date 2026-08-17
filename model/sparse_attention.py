"""Sparse Attention.

Full attention look at every word pair with every other word — cost grow fast when story long.
Sparse attention only look at few important word, skip rest, like hunter only track fresh footprint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 window_size: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch = q.size(0)
        seq_len = q.size(1)
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        # grug: build sliding window mask so each token only see nearby word
        positions = torch.arange(seq_len, device=q.device)
        window_mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        window_mask = (window_mask <= self.window_size).unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(window_mask == 0, -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch, -1, self.d_model)
        return self.W_o(out)

    def flops_for_length(self, seq_len: int) -> int:
        """Rough math rock count for attention."""
        # grug: each token only compare with window, not whole story
        return seq_len * self.window_size * self.d_model
