"""Transformer block with toggles for modern LLM ideas."""

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.mla_attention import MLAttention
from model.sparse_attention import SlidingWindowAttention
from model.feed_forward import FeedForward
from model.moe_layer import MoELayer
from model.norm import make_norm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1,
                 attention_type: str = "standard",
                 latent_dim: int = 64,
                 sparse_window: int = 128,
                 use_moe: bool = False,
                 num_experts: int = 4,
                 top_k: int = 2,
                 use_rope: bool = False,
                 max_len: int = 32768,
                 arch_version: int = 2,
                 n_kv_heads: int = None,
                 qk_norm: bool = True,
                 rope_base: float = 10000.0):
        super().__init__()
        self.arch_version = arch_version
        if attention_type == "mla":
            self.attn = MLAttention(d_model, n_heads, latent_dim, dropout)
        elif attention_type == "sparse":
            self.attn = SlidingWindowAttention(d_model, n_heads,
                                               sparse_window, dropout)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout,
                                           use_rope=use_rope, max_len=max_len,
                                           arch_version=arch_version,
                                           n_kv_heads=n_kv_heads,
                                           qk_norm=qk_norm,
                                           rope_base=rope_base)

        if use_moe:
            self.ff = MoELayer(d_model, d_ff, num_experts, top_k, dropout,
                               arch_version=arch_version)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout,
                                  arch_version=arch_version)

        norm_kind = "rms" if arch_version >= 2 else "layer"
        self.norm1 = make_norm(d_model, norm_kind)
        self.norm2 = make_norm(d_model, norm_kind)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                cache: dict = None) -> torch.Tensor:
        if self.arch_version < 2:
            return self._forward_legacy(x, mask)

        # True pre-LN: the residual stream is never overwritten, so gradients
        # reach layer 0 undamped. The old code did ``x = norm1(x)`` first,
        # which re-normalized the highway at every block and made depth close
        # to useless.
        normed = self.norm1(x)
        if cache is not None:
            attn_out = self.attn(normed, normed, normed, mask, cache=cache)
        else:
            attn_out = self.attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        if isinstance(ff_out, tuple):  # MoE may return (output, aux_loss)
            ff_out, aux = ff_out
            return x + self.dropout(ff_out), aux
        return x + self.dropout(ff_out)

    def _forward_legacy(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.norm1(x)
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        ff_out = self.ff(x)
        return x + self.dropout(ff_out)
