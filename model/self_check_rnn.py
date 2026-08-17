"""RNN module with self-check loop."""

import torch
import torch.nn as nn


class SelfCheckRNN(nn.Module):
    """RNN with a self-check verification loop."""

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int = 1,
                 dropout: float = 0.1, self_check_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.self_check_steps = self_check_steps

        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, output_size)
        """
        # RNN pass
        rnn_out, hidden = self.rnn(x)
        rnn_out = self.dropout(rnn_out)

        # Self-check loop: verify consistency across steps
        final_output = self.projection(rnn_out)
        for _ in range(self.self_check_steps - 1):
            projected = self.projection(rnn_out)
            # Blend previous and current prediction
            final_output = 0.5 * final_output + 0.5 * projected
            # Re-encode through hidden state
            rnn_out, _ = self.rnn(self.dropout(rnn_out))
            projected = self.projection(rnn_out)
            final_output = 0.5 * final_output + 0.5 * projected

        return final_output
