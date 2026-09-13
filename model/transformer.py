"""Main Transformer model combining encoder, transformer blocks, decoder.

``arch_version`` selects the architecture generation:

  1 - the original code path (kept so existing checkpoints still load):
      LayerNorm that overwrites the residual stream, RoPE applied once to the
      input embedding, GELU MLP, hand-rolled attention.

  2 - the modern stack (default for new models):
      * true pre-norm residuals              (Xiong et al. 2020)
      * RMSNorm                              (Zhang & Sennrich 2019)
      * RoPE on Q/K per head                 (Su et al. 2021)
      * QK-Norm                              (Chameleon 2024, Gemma-3, OLMo-2)
      * SwiGLU feed-forward                  (Shazeer 2020)
      * Grouped-Query Attention              (Ainslie et al. 2023)
      * fused Flash attention + KV cache
      * depth-scaled residual init           (GPT-2 / Llama practice)
      * tied embeddings by default           (Press & Wolf 2017)
"""

import math
import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from model.text_encoder import TextEncoder
from model.text_decoder import TextDecoder
from model.multi_token_head import MultiTokenHead
from model.ssm_layer import SimpleSSMLayer
from model.norm import make_norm


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
                 tie_weights: bool = False,
                 arch_version: int = 2,
                 n_kv_heads: int = None,
                 qk_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.attention_type = attention_type
        self.use_rope = use_rope
        self.dropout_p = dropout
        self.arch_version = arch_version
        self.n_kv_heads = n_kv_heads or n_heads
        self.qk_norm = qk_norm
        self.vocab_size = vocab_size

        # In v2 RoPE lives inside attention, so the encoder must not also
        # rotate the embedding (that was the original bug).
        encoder_rope = use_rope and arch_version < 2
        self.encoder = TextEncoder(vocab_size, d_model, max_len, dropout,
                                   use_rope=encoder_rope,
                                   arch_version=arch_version)

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
                                     top_k=top_k,
                                     use_rope=use_rope,
                                     max_len=max_len,
                                     arch_version=arch_version,
                                     n_kv_heads=n_kv_heads,
                                     qk_norm=qk_norm)
                )

        self.norm = make_norm(d_model, "rms" if arch_version >= 2 else "layer")

        if predict_n_tokens > 1:
            self.decoder = MultiTokenHead(d_model, vocab_size, predict_n_tokens)
        else:
            self.decoder = TextDecoder(d_model, vocab_size,
                                       arch_version=arch_version)

        self.tie_weights = tie_weights
        if tie_weights and hasattr(self.decoder, "hidden_to_logits"):
            self.decoder.hidden_to_logits.weight = self.encoder.embedding.weight

        self._init_weights(n_layers=len(self.layers))

    def _init_weights(self, n_layers: int):
        """Normal(0, 0.02) init with depth-scaled residual projections.

        Every projection that writes *into* the residual stream (attention W_o,
        FFN down-projection) is scaled by 1/sqrt(2 * n_layers) so the residual
        variance does not grow with depth. The previous implementation tried to
        branch on ``isinstance(p, nn.Linear)`` while iterating
        ``named_parameters()``, where ``p`` is always a Tensor - so those
        branches never ran and no depth scaling was applied.
        """
        if self.arch_version < 2:
            return self._init_weights_legacy()

        residual_scale = 0.02 / math.sqrt(2 * max(1, n_layers))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("w_down.weight") \
                    or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_scale)

    def _init_weights_legacy(self):
        for name, p in self.named_parameters():
            if "embedding" in name or "hidden_to_logits" in name or "W_o" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "W_q" in name or "W_k" in name or "W_v" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _causal_mask(self, size: int):
        """Lower-triangular boolean mask for causal (left-to-right) attention."""
        device = self.encoder.embedding.weight.device
        return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None,
                caches: list = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            mask: optional external mask. v2 uses the fused kernel's causal
                  flag and does not need one.
            caches: optional list of per-layer KV cache dicts for incremental
                    decoding. Mutated in place.
        Returns:
            (batch, seq_len, vocab_size) if predict_n_tokens == 1
            else (batch, seq_len, predict_n_tokens, vocab_size)
        """
        if self.arch_version < 2 and mask is None:
            mask = self._causal_mask(x.size(1))

        x = self.encoder(x, offset=self._cache_offset(caches))
        aux_total = None
        for index, layer in enumerate(self.layers):
            cache = caches[index] if caches is not None else None
            if isinstance(layer, TransformerBlock):
                out = layer(x, mask, cache=cache)
            else:
                out = layer(x, mask)
            if isinstance(out, tuple):
                x, aux = out
                aux_total = aux if aux_total is None else aux_total + aux
            else:
                x = out
        x = self.norm(x)
        # Keep final logits in fp32 to avoid overflow on the output projection.
        logits = self.decoder(x.to(torch.float32))
        if aux_total is not None:
            return logits, aux_total
        return logits

    @staticmethod
    def _cache_offset(caches) -> int:
        if not caches:
            return 0
        first = caches[0]
        return first["k"].size(2) if isinstance(first, dict) and "k" in first else 0

    def new_caches(self) -> list:
        """Allocate one empty KV cache dict per layer for generation."""
        return [{} for _ in self.layers]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        torch.save({"model": self.state_dict(), "config": self._get_config()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["model"])

    def _get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": len(self.layers),
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "max_len": self.max_len,
            "attention_type": self.attention_type,
            "use_rope": self.use_rope,
            "dropout": self.dropout_p,
            "tie_weights": self.tie_weights,
            "arch_version": self.arch_version,
            "qk_norm": self.qk_norm,
        }
