"""Multi-Head Latent Attention (MLA).

Old attention need store big key-value memory for every past word.
MLA squish that memory into small "latent" shape before store, then unsquish when need.
Same smart thinking, less memory rock carried.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 latent_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.latent_dim = latent_dim

        # grug: squish query/key/value down to small latent rock
        self.W_dkv = nn.Linear(d_model, latent_dim)
        self.W_dq = nn.Linear(d_model, latent_dim)

        # grug: unsquish latent back to full size for smart compare
        self.W_uk = nn.Linear(latent_dim, d_model)
        self.W_uv = nn.Linear(latent_dim, d_model)
        self.W_uq = nn.Linear(latent_dim, d_model)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: (batch, seq_len, d_model)
            k: (batch, seq_len, d_model)
            v: (batch, seq_len, d_model)
            mask: optional (1, 1, seq_len, seq_len)
        Returns:
            (batch, seq_len, d_model)
        """
        batch = q.size(0)

        # squish inputs into small latent rock
        c_kv = self.W_dkv(v)  # value + key share one squished rock
        c_q = self.W_dq(q)

        # unsquish back for attention math
        q = self.W_uq(c_q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_uk(c_kv).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_uv(c_kv).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch, -1, self.d_model)
        return self.W_o(out)

    def kv_cache_size(self, seq_len: int) -> int:
        """Return memory rocks needed to store one sequence in cache."""
        # grug: store only latent vector per position, not full d_model
        return seq_len * self.latent_dim
