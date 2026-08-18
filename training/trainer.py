"""Live training runner with loss tracking and simple generation check."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.device_utils import get_best_device


class Trainer:
    """Train a tiny language model on sample stories.

    Supports AMP, gradient accumulation, a validation split, and cosine LR.
    """

    def __init__(self, model, dataset, batch_size: int = 4,
                 lr: float = 1e-3, device: str | None = None,
                 max_iters: int = 500,
                 val_dataset=None,
                 grad_accum_steps: int = 1,
                 warmup_steps: int = 200,
                 use_amp: bool = True):
        if device is None:
            device = get_best_device()
        self.model = model.to(device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.max_iters = max_iters
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
        self.optimizer.zero_grad()
        if hasattr(dataset, "collate_pad"):
            collator = dataset.collate_pad
        elif hasattr(dataset, "dataset"):
            collator = dataset.dataset.collate_pad
        else:
            collator = None
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collator)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collator)
        else:
            self.val_loader = None
        self.history = []
        self.val_history = []
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_step = -1
        self._best_model_state = None

    def _lr_for_step(self, step: int, lr: float):
        """Linear warmup then cosine decay to 0.1 * lr."""
        if step < self.warmup_steps:
            return lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_iters - self.warmup_steps)
        return 0.1 * lr + 0.9 * lr * (1 + float(torch.cos(torch.tensor(progress * 3.141592653589793)))) / 2

    def _set_lr(self, step: int):
        base_lr = self.optimizer.defaults["lr"]
        new_lr = self._lr_for_step(step, base_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def _causal_mask(self, size: int):
        return torch.tril(torch.ones(size, size, device=self.device)).unsqueeze(0).unsqueeze(0)

    def _compute_loss(self, out, y, mask):
        aux_loss = 0.0
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits = out
        if logits.dim() == 4:
            logits = logits[:, :, 0, :]
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               reduction="none")
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + 1.0 * aux_loss
        return loss

    def train_step(self, x, y, mask, step: int):
        self._set_lr(step)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
            out = self.model(x)
            loss = self._compute_loss(out, y, mask)
            loss = loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def val_loss(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        total = 0.0
        count = 0
        for x, y, mask in self.val_loader:
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                out = self.model(x)
                loss = self._compute_loss(out, y, mask)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / count if count else None

    def train(self):
        data_iter = iter(self.loader)
        pbar = tqdm(range(self.max_iters), desc="training")
        for step in pbar:
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                x, y, mask = next(data_iter)
            loss = self.train_step(x, y, mask, step)
            self.history.append(loss)
            self.best_loss = min(self.best_loss, loss)
            postfix = {"loss": f"{loss:.4f}", "best": f"{self.best_loss:.4f}"}
            if step % 500 == 0 and self.val_loader is not None:
                vl = self.val_loss()
                if vl is not None:
                    self.val_history.append(vl)
                    if vl < self.best_val_loss:
                        self.best_val_loss = vl
                        self.best_val_step = step
                        self._best_model_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                    postfix["val"] = f"{vl:.4f}"
            pbar.set_postfix(postfix)
        # Restore the best validation-loss weights rather than the final ones.
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
        return self.history


    def save_best_val(self, path: str):
        """Save the tracked best-validation weights to disk."""
        if self._best_model_state is None:
            self._best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
        torch.save({
            "model": self._best_model_state,
            "config": self.model.config if hasattr(self.model, "config") else {},
        }, path)

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
