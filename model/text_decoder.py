"""Text decoder."""

import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    """Decodes hidden representations back to token logits."""

    def __init__(self, d_model: int, vocab_size: int, arch_version: int = 2):
        super().__init__()
        self.arch_version = arch_version
        # No bias on the output projection in v2: it is a no-op once the model
        # is trained and it blocks weight tying.
        self.hidden_to_logits = nn.Linear(d_model, vocab_size,
                                          bias=arch_version < 2)
        # Dropout immediately before the logits fights the loss for no benefit;
        # v2 drops it and relies on residual dropout instead.
        self.dropout = nn.Dropout(0.1 if arch_version < 2 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, vocab_size)
        """
        if self.training and self.arch_version < 2:
            x = self.dropout(x)
        return self.hidden_to_logits(x)
