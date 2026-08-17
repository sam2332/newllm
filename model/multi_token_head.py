"""Multi-token prediction head.

Old model guess next word one at a time, over over.
New trick teach model guess several word ahead same time during training.
"""

import torch
import torch.nn as nn


class MultiTokenHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, n_future: int = 4):
        super().__init__()
        self.n_future = n_future
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_future)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, n_future, vocab_size)
        """
        logits = [head(x) for head in self.heads]
        return torch.stack(logits, dim=2)
