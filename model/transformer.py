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
                 predict_n_tokens: int = 1,
                 tie_weights: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.attention_type = attention_type
        self.use_rope = use_rope
        self.dropout_p = dropout
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

        self.tie_weights = tie_weights
        if tie_weights and hasattr(self.decoder, "hidden_to_logits"):
            self.decoder.hidden_to_logits.weight = self.encoder.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Scaled Xavier init for stable training of deep/wide models."""
        for name, p in self.named_parameters():
            if "embedding" in name or "hidden_to_logits" in name or "W_o" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "W_q" in name or "W_k" in name or "W_v" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)
            elif isinstance(p, nn.LayerNorm):
                if p.weight is not None:
                    nn.init.ones_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)

    def _causal_mask(self, size: int):
        """Lower-triangular boolean mask for causal (left-to-right) attention."""
        return torch.tril(torch.ones(size, size, device=self.encoder.embedding.weight.device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            mask: optional external mask; if None, a causal mask is built.
        Returns:
            (batch, seq_len, vocab_size) if predict_n_tokens == 1
            else (batch, seq_len, predict_n_tokens, vocab_size)
        """
        if mask is None:
            mask = self._causal_mask(x.size(1))
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # Keep final logits in full precision to avoid fp16 overflow on the
        # large output projection (especially important for untied 1024-dim
        # decoder weights).
        return self.decoder(x.to(torch.float32))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        torch.save({"model": self.state_dict(), "config": self._get_config()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model"])

    def _get_config(self):
        return {
            "vocab_size": self.decoder.hidden_to_logits.weight.size(0)
            if hasattr(self.decoder, "hidden_to_logits")
            else 256,
            "d_model": self.d_model,
            "n_layers": len(self.layers),
            "n_heads": self.n_heads if hasattr(self, "n_heads") else 8,
            "d_ff": self.d_ff if hasattr(self, "d_ff") else self.d_model * 4,
            "max_len": self.max_len,
            "attention_type": self.attention_type if hasattr(self, "attention_type") else "standard",
            "use_rope": self.use_rope if hasattr(self, "use_rope") else True,
            "dropout": self.dropout_p if hasattr(self, "dropout_p") else 0.1,
            "tie_weights": self.tie_weights,
        }
