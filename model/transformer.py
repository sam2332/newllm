"""Main Transformer model combining encoder, transformer blocks, decoder."""

import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from model.text_encoder import TextEncoder
from model.text_decoder import TextDecoder
from model.multi_token_head import MultiTokenHead
from model.ssm_layer import SimpleSSMLayer


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, max_len: int = 32768,
                 dropout: float = 0.1,
                 attention_type: str = "standard",
                 latent_dim: int = 64,
                 sparse_window: int = 128,
                 use_moe: bool = False,
                 num_experts: int = 4,
                 top_k: int = 2,
                 use_rope: bool = True,
                 hybrid_pattern: list = None,
                 predict_n_tokens: int = 1):
        super().__init__()
        self.predict_n_tokens = predict_n_tokens
        self.encoder = TextEncoder(vocab_size, d_model, max_len, dropout,
                                   use_rope=use_rope)

        # grug: decide each layer type from hybrid pattern
        if hybrid_pattern is None:
            hybrid_pattern = ["attn"] * n_layers
        self.layers = nn.ModuleList()
        for layer_type in hybrid_pattern:
            if layer_type == "ssm":
                self.layers.append(SimpleSSMLayer(d_model, state_dim=d_model // 4,
                                                  dropout=dropout))
            else:
                self.layers.append(
                    TransformerBlock(d_model, n_heads, d_ff, dropout,
                                     attention_type=attention_type,
                                     latent_dim=latent_dim,
                                     sparse_window=sparse_window,
                                     use_moe=use_moe,
                                     num_experts=num_experts,
                                     top_k=top_k)
                )

        self.norm = nn.LayerNorm(d_model)

        if predict_n_tokens > 1:
            self.decoder = MultiTokenHead(d_model, vocab_size, predict_n_tokens)
        else:
            self.decoder = TextDecoder(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            (batch, seq_len, vocab_size) if predict_n_tokens == 1
            else (batch, seq_len, predict_n_tokens, vocab_size)
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.decoder(x)
