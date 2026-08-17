"""Live training runner with loss tracking and simple generation check."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.device_utils import get_best_device


class Trainer:
    """Train a tiny language model on sample stories."""

    def __init__(self, model, dataset, batch_size: int = 4,
                 lr: float = 1e-3, device: str = None,
                 max_iters: int = 500):
        if device is None:
            device = get_best_device()
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.max_iters = max_iters
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if hasattr(dataset, "collate_pad"):
            collator = dataset.collate_pad
        elif hasattr(dataset, "dataset"):
            collator = dataset.dataset.collate_pad
        else:
            collator = None
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collator)
        self.history = []
        self.best_loss = float("inf")

    def _causal_mask(self, size: int):
        return torch.tril(torch.ones(size, size, device=self.device)).unsqueeze(0).unsqueeze(0)

    def train_step(self, x, y, mask):
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(x)
        aux_loss = 0.0
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits = out
        if logits.dim() == 4:
            # multi-token head: only use first future token for simple training
            logits = logits[:, :, 0, :]
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1),
                               reduction="none")
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + 1.0 * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.model.train()
        data_iter = iter(self.loader)
        pbar = tqdm(range(self.max_iters), desc="training")
        for step in pbar:
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                x, y, mask = next(data_iter)
            loss = self.train_step(x, y, mask)
            self.history.append(loss)
            self.best_loss = min(self.best_loss, loss)
            pbar.set_postfix(loss=f"{loss:.4f}", best=f"{self.best_loss:.4f}")
        return self.history

    def generate(self, prompt: str, max_new: int = 20,
                 tokenizer=None, detokenizer=None,
                 use_reasoning: bool = False,
                 think_steps: int = 5,
                 sampler=None) -> str:
        self.model.eval()
        if tokenizer is None:
            tokenizer = lambda text: [min(ord(c), 255) for c in text]
        if detokenizer is None:
            detokenizer = lambda tokens: "".join(chr(min(t, 255)) for t in tokens)
        tokens = tokenizer(prompt)
        if use_reasoning:
            from reasoning import ReasoningTimeHelper
            helper = ReasoningTimeHelper(self.model, think_steps=think_steps)
            tokens = helper.think_then_answer(
                torch.tensor([tokens], dtype=torch.long, device=self.device)
            )[0].tolist()
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                return detokenizer(input_ids[0].tolist())
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated = tokens[:]
        if sampler is None:
            from sampling import Sampler
            sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0,
                              repetition_penalty=1.0)
        with torch.no_grad():
            for _ in range(max_new):
                out = self.model(input_ids)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                if logits.dim() == 4:
                    logits = logits[:, :, 0, :]
                next_logits = logits[:, -1, :]
                gen_tensor = torch.tensor(generated, dtype=torch.long, device=self.device)
                next_token = sampler.sample(next_logits, gen_tensor)
                generated.append(next_token)
                input_ids = torch.cat([input_ids,
                                       torch.tensor([[next_token]], device=self.device)], dim=1)
        return detokenizer(generated)
