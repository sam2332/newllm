"""Feed-forward layer.

``arch_version=2`` uses SwiGLU (two gated projections), the standard modern
FFN. ``arch_version=1`` keeps the original single GELU MLP so old checkpoints
load unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1,
                 arch_version: int = 2):
        super().__init__()
        self.arch_version = arch_version
        if arch_version >= 2:
            # Keep the parameter budget comparable to the GELU MLP: SwiGLU has
            # three matrices instead of two, so shrink the hidden width by 2/3
            # and round to a multiple of 64 for tensor-core friendliness.
            hidden = int(2 * d_ff / 3)
            hidden = max(64, ((hidden + 63) // 64) * 64)
            self.w_gate = nn.Linear(d_model, hidden, bias=False)
            self.w_up = nn.Linear(d_model, hidden, bias=False)
            self.w_down = nn.Linear(hidden, d_model, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch_version >= 2:
            return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
