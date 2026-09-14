"""Mixture-of-Experts (MoE).

Instead one giant brain-blob do every task, model split into many small "expert" brain-chunks.
For each word, only few expert wake up and work, rest sleep.
Model total very big but each guess only use small part — fast and cheap like small model.

``arch_version=3`` makes the layer isomorphic to Qwen3-MoE so it exports to
GGUF: a bias-free router, softmax -> top-k -> renormalize (Qwen's
``norm_topk_prob=True``), and no shared expert. It also returns a
load-balancing loss - without one the router collapses onto a couple of
experts within the first few hundred steps and the rest never train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.feed_forward import FeedForward


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 2, dropout: float = 0.1,
                 arch_version: int = 2, balance_alpha: float = 0.01,
                 z_alpha: float = 1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.arch_version = arch_version
        self.balance_alpha = balance_alpha
        self.z_alpha = z_alpha
        # grug: many small expert blobs
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout, arch_version=arch_version)
            for _ in range(num_experts)
        ])
        # grug: router decides which expert wake up
        self.router = nn.Linear(d_model, num_experts, bias=arch_version < 3)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model), or ``(output, aux_loss)`` for
            ``arch_version>=3`` - the block and the Transformer already
            unpack a tuple and sum the aux terms.
        """
        batch, seq_len, _ = x.shape
        flat = x.reshape(-1, x.size(-1))  # (batch*seq_len, d_model)
        router_logits = self.router(flat)  # (batch*seq_len, num_experts)
        probs = F.softmax(router_logits.float(), dim=-1)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)

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

        output = output.view(batch, seq_len, -1)
        if self.arch_version < 3:
            return output

        # Switch-Transformer load balancing: penalize the product of the
        # fraction of tokens routed to each expert and the mean probability it
        # was assigned, which is minimized by a uniform split. Scaled by E so
        # the optimum is 1 regardless of the expert count. The z-loss keeps the
        # router logits from drifting to magnitudes where bf16 softmax
        # saturates - the same reason the trainer applies it to the output.
        with torch.autocast("cuda", enabled=False):
            fraction = F.one_hot(selected, self.num_experts).float().sum(1).mean(0)
            mean_prob = probs.mean(0)
            balance = self.num_experts * (fraction * mean_prob).sum()
            z = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
        aux = self.balance_alpha * balance + self.z_alpha * z
        return output, aux

    @torch.no_grad()
    def expert_load(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of tokens routed to each expert (for utilisation logs)."""
        router_logits = self.router(x.reshape(-1, x.size(-1)))
        _, selected = torch.topk(router_logits, self.top_k, dim=-1)
        counts = torch.bincount(selected.flatten(), minlength=self.num_experts)
        return counts.float() / max(1, selected.numel())

    def active_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Return which experts fire for given input."""
        router_logits = self.router(x.view(-1, x.size(-1)))
        _, selected = torch.topk(router_logits, self.top_k, dim=-1)
        return selected.unique(sorted=True)
