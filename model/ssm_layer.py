"""State-space model layer (simple Mamba-like).

Instead comparing every word to every word, model carry forward a "memory state".
Like grug walking and updating mental map step by step.
"""

import torch
import torch.nn as nn


class SimpleSSMLayer(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # grug: input gate, forget gate, output gate, like tiny LSTM but simpler
        self.input_proj = nn.Linear(d_model, state_dim)
        self.forget_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(d_model, state_dim)
        self.to_model = nn.Linear(state_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: ignored, for API compatibility with transformer blocks
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        state = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            inp = torch.sigmoid(self.input_proj(x[:, t]))
            forget = torch.sigmoid(self.forget_proj(x[:, t]))
            out_gate = torch.sigmoid(self.output_proj(x[:, t]))
            state = forget * state + inp
            hidden = out_gate * state
            outputs.append(self.dropout(self.to_model(hidden)))
        out = torch.stack(outputs, dim=1)
        return x + out

    def state_memory_size(self, batch: int) -> int:
        """Memory rocks needed to keep one state."""
        return batch * self.state_dim
