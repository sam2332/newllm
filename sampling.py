"""Sampling helpers for text generation.

Add temperature, top-k, top-p (nucleus), min-p, repetition penalty.
"""

import torch
import torch.nn.functional as F


class Sampler:
    """Configurable sampler for LLM generation."""

    def __init__(self, temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0, min_p: float = 0.0,
                 repetition_penalty: float = 1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty

    def apply_repetition_penalty(self, logits: torch.Tensor,
                                  generated: torch.Tensor) -> torch.Tensor:
        """Down-weight tokens already used."""
        if self.repetition_penalty == 1.0 or generated.numel() == 0:
            return logits
        for token in generated.unique():
            logits[..., token] /= self.repetition_penalty
        return logits

    def sample(self, logits: torch.Tensor, generated: torch.Tensor = None) -> int:
        """
        Args:
            logits: raw logits for next token, shape (..., vocab_size)
            generated: already generated token ids
        Returns:
            sampled token id
        """
        if generated is not None:
            logits = self.apply_repetition_penalty(logits, generated)

        # temperature
        logits = logits / max(self.temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        # top-k
        if self.top_k > 0:
            top_k = min(self.top_k, probs.size(-1))
            values, indices = torch.topk(probs, top_k)
            mask = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # top-p (nucleus)
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            keep_mask = cumulative <= self.top_p
            # keep at least one token
            keep_mask[..., 0] = True
            keep_indices = sorted_indices[keep_mask].unsqueeze(0)
            mask = torch.zeros_like(probs).scatter_(-1, keep_indices, 1.0)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # min-p: drop tokens below min_p * max_prob
        if self.min_p > 0:
            max_prob = probs.max(dim=-1, keepdim=True).values
            threshold = self.min_p * max_prob
            probs = probs.masked_fill(probs < threshold, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        token = torch.multinomial(probs, num_samples=1)
        return token.item()


def generate_with_sampler(model, prompt_tokens: list, sampler: Sampler,
                          max_new: int = 50, device: str = "cuda") -> list:
    """Generate token ids using a sampler."""
    model.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.dim() == 4:
                logits = logits[:, :, 0, :]
            next_logits = logits[:, -1, :]
            gen_tensor = torch.tensor(generated, dtype=torch.long, device=device)
            next_token = sampler.sample(next_logits, gen_tensor)
            generated.append(next_token)
            input_ids = torch.cat([input_ids,
                                    torch.tensor([[next_token]], device=device)], dim=1)
    return generated
