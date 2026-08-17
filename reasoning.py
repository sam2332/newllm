"""Reasoning-time compute helper.

Not change transformer block itself, but change how model used.
Model allowed think many extra step before give final answer.
Extra thinking boost hard-problem solving.
"""

import torch
import torch.nn as nn


class ReasoningTimeHelper:
    """Wrap a model so it can think before answering."""

    def __init__(self, model: nn.Module, think_steps: int = 5,
                 start_thought_token: int = 1, end_thought_token: int = 2):
        self.model = model
        self.think_steps = think_steps
        self.start_thought_token = start_thought_token
        self.end_thought_token = end_thought_token

    def think_then_answer(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt: (batch, seq_len) token indices
        Returns:
            (batch, final_seq_len) token indices with reasoning and answer
        """
        generated = [prompt]
        current = prompt
        for _ in range(self.think_steps):
            logits = self.model(current)[:, -1, :]  # last token logits
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        generated.append(torch.full((prompt.size(0), 1),
                                     self.end_thought_token,
                                     dtype=prompt.dtype,
                                     device=prompt.device))
        return torch.cat(generated, dim=1)
