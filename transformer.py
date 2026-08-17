#!/usr/bin/env python3
"""Minimal Transformer from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
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
        # Project
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch, -1, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
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


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, vocab_size)
        """
        x = self.embedding(x) * math.sqrt(self.embedding.weight.shape[1])
        x = self.pos_encoder(self.dropout(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.norm(x))


if __name__ == "__main__":
    # Quick smoke test
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4)
    tokens = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
    out = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {out.shape}")
    print("Transformer works.")
