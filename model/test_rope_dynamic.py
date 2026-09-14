"""The context length is an allocation detail, not an architectural limit.

RoPE is a pure function of position and the model has no learned position
parameters, so a sequence longer than the declared ``max_len`` must produce
exactly the logits it would have produced had ``max_len`` been large enough.
These tests pin that, because the alternative - a hard error at a config value -
is what made context length feel like a model property.

Length *generalization* is a separate matter entirely: the model runs at any
length, but is only good near the lengths it trained on.
"""

import torch

from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from model.rope import RoPECache
from model.transformer import Transformer

SMALL = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128,
             arch_version=2, n_kv_heads=1, qk_norm=True,
             use_rope=True, dropout=0.0)


def _model(max_len):
    return Transformer(vocab_size=TOK.vocab_size, max_len=max_len, **SMALL).eval()


def test_table_grows_instead_of_raising():
    cache = RoPECache(head_dim=16, max_len=8)
    assert cache.cos.size(0) == 8
    cache.get(seq_len=100)
    assert cache.cos.size(0) >= 100


def test_grown_table_matches_preallocated():
    big = RoPECache(head_dim=16, max_len=256)
    small = RoPECache(head_dim=16, max_len=8)
    cos_b, sin_b = big.get(seq_len=200)
    cos_s, sin_s = small.get(seq_len=200)          # forces a rebuild
    assert torch.allclose(cos_b, cos_s, atol=1e-6)
    assert torch.allclose(sin_b, sin_s, atol=1e-6)


def test_sequence_longer_than_declared_max_len():
    """The case that used to raise ValueError."""
    model = _model(max_len=64)
    with torch.no_grad():
        out = model(torch.randint(0, TOK.vocab_size, (1, 512)))
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[:2] == (1, 512)


def test_declared_max_len_does_not_change_the_answer():
    torch.manual_seed(0)
    big = _model(max_len=2048)
    small = _model(max_len=32)
    small.load_state_dict(big.state_dict())
    x = torch.randint(0, TOK.vocab_size, (1, 600))
    with torch.no_grad():
        a = big(x)
        b = small(x)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    # Float32 rounding only; the tables are identical by construction.
    assert (a - b).abs().max().item() < 1e-4
