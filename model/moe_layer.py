"""Mixture-of-Experts (MoE).

Instead one giant brain-blob do every task, model split into many small "expert" brain-chunks.
For each word, only few expert wake up and work, rest sleep.
Model total very big but each guess only use small part — fast and cheap like small model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.feed_forward import FeedForward


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 2, dropout: float = 0.1,
                 arch_version: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # grug: many small expert blobs
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout, arch_version=arch_version)
            for _ in range(num_experts)
        ])
        # grug: router decides which expert wake up
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        flat = x.view(-1, x.size(-1))  # (batch*seq_len, d_model)
        router_logits = self.router(flat)  # (batch*seq_len, num_experts)
        weights, selected = torch.topk(F.softmax(router_logits, dim=-1),
                                        self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # grug: build big tensor of expert outputs, only use selected ones
        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = selected == expert_idx
            positions = mask.nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            expert_in = flat[positions]
            expert_out = expert(expert_in)
            # gather weight for this expert at each position
            w = weights[mask]
            output[positions] += w.unsqueeze(-1) * expert_out

        return output.view(batch, seq_len, -1)

    def active_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Return which experts fire for given input."""
        router_logits = self.router(x.view(-1, x.size(-1)))
        _, selected = torch.topk(router_logits, self.top_k, dim=-1)
        return selected.unique(sorted=True)
