"""Transformer block with toggles for modern LLM ideas."""

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.mla_attention import MLAttention
from model.sparse_attention import SlidingWindowAttention
from model.feed_forward import FeedForward
from model.moe_layer import MoELayer


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1,
                 attention_type: str = "standard",
                 latent_dim: int = 64,
                 sparse_window: int = 128,
                 use_moe: bool = False,
                 num_experts: int = 4,
                 top_k: int = 2):
        super().__init__()
        if attention_type == "mla":
            self.attn = MLAttention(d_model, n_heads, latent_dim, dropout)
        elif attention_type == "sparse":
            self.attn = SlidingWindowAttention(d_model, n_heads,
                                               sparse_window, dropout)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        if use_moe:
            self.ff = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm1(x)
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        ff_out = self.ff(x)
        return x + self.dropout(ff_out)
