"""Smoke test entry point."""

import torch
from model.transformer import Transformer

if __name__ == "__main__":
    # Quick smoke test
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4)
    tokens = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
    out = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {out.shape}")
    print("Transformer works.")
