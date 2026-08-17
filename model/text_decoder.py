"""Text decoder."""

import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    """Decodes hidden representations back to token logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.hidden_to_logits = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, vocab_size)
        """
        return self.dropout(self.hidden_to_logits(x))
