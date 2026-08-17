"""Sequence-level Mixture-of-Experts.

Incoming story get embedded first.
Then router look at whole sequence embedding and pick one (or few) expert mini-minds.
Each expert is smaller transformer that only sees stories routed to it.
Good for clustering different story types: hunting tales, fire tales, sky tales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer
from model.text_encoder import TextEncoder


class SequenceMoE(nn.Module):
    """MoE where each sequence is routed as one unit to a mini-mind."""

    def __init__(self, vocab_size: int, num_experts: int = 4,
                 top_k: int = 1, d_model: int = 256,
                 expert_layers: int = 2, expert_heads: int = 4,
                 d_ff: int = 512, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        # grug: shared encoder turns token story into embedding
        self.encoder = TextEncoder(vocab_size, d_model, max_len, dropout,
                                   use_rope=True)

        # grug: router reads average embedding and decides which mini-mind
        self.router = nn.Linear(d_model, num_experts)
        # start with balanced router so no expert dominates at beginning
        nn.init.zeros_(self.router.bias)

        # grug: many small transformer mini-minds, one per expert
        self.experts = nn.ModuleList([
            Transformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=expert_layers,
                n_heads=expert_heads,
                d_ff=d_ff,
                max_len=max_len,
                attention_type="standard",
                use_rope=True,
            )
            for _ in range(num_experts)
        ])

    def _router_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Load-balance loss: encourage even expert usage."""
        # grug: router_probs shape (batch, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        # fraction of routing mass per expert
        usage = probs.mean(dim=0)
        # want each expert to get 1/num_experts mass
        target = torch.ones_like(usage) / self.num_experts
        return ((usage - target) ** 2).sum()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            logits, aux_loss tuple
        """
        batch, seq_len = x.shape

        # embed whole sequence
        emb = self.encoder(x)  # (batch, seq_len, d_model)

        # pool to one vector per sequence
        pooled = emb.mean(dim=1)  # (batch, d_model)

        # router choose top-k experts with noise for exploration
        router_logits = self.router(pooled)  # (batch, num_experts)
        # add gumbel-like noise so all experts get a chance early
        if self.training:
            noise = torch.rand_like(router_logits)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
            router_logits = router_logits + gumbel_noise
        weights, selected = torch.topk(F.softmax(router_logits, dim=-1),
                                       self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize

        # accumulate outputs from chosen experts
        # grug: all experts share same vocab head size
        sample_out = self.experts[0](x[:1])
        vocab_size = sample_out.size(-1)
        output = torch.zeros(batch, seq_len, vocab_size,
                             device=x.device, dtype=sample_out.dtype)

        for expert_idx, expert in enumerate(self.experts):
            mask = selected == expert_idx
            positions = mask.nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            expert_in = x[positions]
            expert_out = expert(expert_in)
            w = weights[mask]
            output[positions] += w.unsqueeze(-1).unsqueeze(-1) * expert_out

        aux_loss = self._router_loss(router_logits)
        return output, aux_loss

    def cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Return dominant expert index for each sequence."""
        emb = self.encoder(x)
        pooled = emb.mean(dim=1)
        router_logits = self.router(pooled)
        return router_logits.argmax(dim=-1)
