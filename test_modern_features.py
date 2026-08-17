"""Test all modern LLM features."""

import torch
from model.transformer import Transformer
from model.mla_attention import MLAttention
from model.sparse_attention import SlidingWindowAttention
from model.moe_layer import MoELayer
from reasoning import ReasoningTimeHelper


def test_standard():
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4)
    tokens = torch.randint(0, 1000, (2, 32))
    out = model(tokens)
    assert out.shape == (2, 32, 1000)
    print("standard OK")


def test_mla():
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4, attention_type="mla", latent_dim=32)
    tokens = torch.randint(0, 1000, (2, 32))
    out = model(tokens)
    assert out.shape == (2, 32, 1000)
    print("mla OK")


def test_sparse():
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4, attention_type="sparse",
                        sparse_window=8)
    tokens = torch.randint(0, 1000, (2, 64))
    out = model(tokens)
    assert out.shape == (2, 64, 1000)
    print("sparse OK")


def test_moe():
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4, use_moe=True, num_experts=4, top_k=2)
    tokens = torch.randint(0, 1000, (2, 16))
    out = model(tokens)
    assert out.shape == (2, 16, 1000)
    print("moe OK")


def test_hybrid():
    pattern = ["attn", "ssm", "attn", "ssm"]
    model = Transformer(vocab_size=1000, d_model=128, n_layers=4,
                        n_heads=4, hybrid_pattern=pattern)
    tokens = torch.randint(0, 1000, (2, 32))
    out = model(tokens)
    assert out.shape == (2, 32, 1000)
    print("hybrid OK")


def test_multi_token():
    model = Transformer(vocab_size=1000, d_model=128, n_layers=2,
                        n_heads=4, predict_n_tokens=3)
    tokens = torch.randint(0, 1000, (2, 32))
    out = model(tokens)
    assert out.shape == (2, 32, 3, 1000)
    print("multi-token OK")


def test_reasoning():
    model = Transformer(vocab_size=1000, d_model=64, n_layers=2,
                        n_heads=2)
    helper = ReasoningTimeHelper(model, think_steps=3)
    prompt = torch.randint(0, 1000, (1, 5))
    out = helper.think_then_answer(prompt)
    assert out.shape[0] == 1
    assert out.shape[1] > 5
    print("reasoning OK")


if __name__ == "__main__":
    test_standard()
    test_mla()
    test_sparse()
    test_moe()
    test_hybrid()
    test_multi_token()
    test_reasoning()
    print("All modern feature tests passed.")
