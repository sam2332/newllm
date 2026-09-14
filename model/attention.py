"""Multi-head attention.

``arch_version=2`` is the modern stack:
  * fused scaled-dot-product attention (Flash / mem-efficient kernels)
  * RoPE applied per head to Q and K, not to the input embedding
  * QK-Norm (Henry et al. 2020; Chameleon 2024; Gemma-3, OLMo-2) - RMS-normalize
    queries and keys before the dot product. This is the single most effective
    known fix for attention-logit blow-up in low precision.
  * Grouped-Query Attention (Ainslie et al. 2023) - fewer KV heads than query
    heads, shrinking the KV cache with no measurable quality loss.
  * incremental KV cache for O(1)-per-token generation

``arch_version=1`` reproduces the original hand-rolled math so old checkpoints
stay runnable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RoPECache, apply_rope
from model.norm import RMSNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, max_len: int = 32768,
                 arch_version: int = 2, n_kv_heads: int = None,
                 qk_norm: bool = True, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout
        self.arch_version = arch_version
        self.use_rope = use_rope and arch_version >= 2

        if arch_version >= 2:
            self.n_kv_heads = n_kv_heads or n_heads
            assert n_heads % self.n_kv_heads == 0, \
                "n_heads must be divisible by n_kv_heads for GQA"
        else:
            self.n_kv_heads = n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        kv_dim = self.n_kv_heads * self.d_k

        # v3 drops the projection biases: Qwen3 has none, and a bias-free
        # attention block exports to GGUF as a plain tensor copy.
        bias = arch_version < 3
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, kv_dim, bias=bias)
        self.W_v = nn.Linear(d_model, kv_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.qk_norm = qk_norm and arch_version >= 2
        if self.qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)
        # v2 pairs adjacent dims; v3 pairs halves (llama.cpp NEOX layout).
        self.rope_interleaved = arch_version < 3
        if self.use_rope:
            self.rope = RoPECache(self.d_k, max_len, base=rope_base,
                                  interleaved=self.rope_interleaved)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None, cache: dict = None) -> torch.Tensor:
        """
        Args:
            q, k, v: (batch, seq_len, d_model)
            mask: optional (1, 1, seq_len, seq_len); unused in v2, which relies
                  on the fused kernel's causal flag.
            cache: optional dict holding "k"/"v" for incremental decoding.
                   Mutated in place; pass the same dict across steps.
        Returns:
            (batch, seq_len, d_model)
        """
        if self.arch_version < 2:
            return self._forward_legacy(q, k, v, mask)

        batch, seq_len, _ = q.shape
        offset = cache["k"].size(2) if cache is not None and "k" in cache else 0

        q = self.W_q(q).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # QK-Norm before RoPE: normalize magnitude, let rotation supply phase.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            cos, sin = self.rope.get(seq_len, offset=offset,
                                     device=q.device, dtype=q.dtype)
            q, k = apply_rope(q, k, cos, sin, interleaved=self.rope_interleaved)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # GQA: broadcast each KV head across its group of query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # A single new query attending over cached keys needs no causal mask;
        # a full prefill does. is_causal requires matching q/k lengths.
        is_causal = q.size(2) == k.size(2) and q.size(2) > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)

    def _forward_legacy(self, q, k, v, mask):
        batch = q.size(0)
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch, -1, self.d_model)
        return self.W_o(out)
