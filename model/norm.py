"""Normalization layers.

RMSNorm (Zhang & Sennrich 2019) drops the mean-centering and the bias of
LayerNorm. It is what Llama, Mistral, Qwen and Gemma all use: same quality,
fewer ops, and no bias term to destabilize fp16/bf16 training.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Scales each token vector by its RMS and applies a learned per-channel gain.
    Unlike LayerNorm there is no mean subtraction and no bias, which is both
    cheaper and more stable in bf16. The statistic is computed in fp32 and cast
    back, so autocast cannot degrade it.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 even under autocast; this is the usual
        # stability fix for low-precision training.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


def make_norm(d_model: int, kind: str = "rms") -> nn.Module:
    return RMSNorm(d_model) if kind == "rms" else nn.LayerNorm(d_model)
