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
                 use_amp: bool = True,
                 amp_dtype: str = "bf16",
                 weight_decay: float = 0.1,
                 betas: tuple = (0.9, 0.95),
                 z_loss: float = 1e-4,
                 num_workers: int = 4,
                 checkpoint_path: str = None,
                 save_every: int = 0,
                 resume: bool = False):
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
        # bf16 needs no loss scaling and cannot overflow the way fp16 does.
        # Both an RTX 4090 (sm_89) and a 5090 (sm_120) support it natively.
        if amp_dtype == "bf16" and torch.cuda.is_available() \
                and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16)
        self.z_loss = z_loss

        # Decay only matrices. Biases, norm gains and embeddings are excluded,
        # which is standard practice and matters more at small scale.
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "embedding" in name or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, eps=1e-8, fused=True)
        self._base_lr = lr
        self.optimizer.zero_grad()
        if hasattr(dataset, "collate_pad"):
            collator = dataset.collate_pad
        elif hasattr(dataset, "dataset"):
            collator = dataset.dataset.collate_pad
        else:
            collator = None
        loader_kw = dict(num_workers=num_workers, pin_memory=True,
                         persistent_workers=num_workers > 0,
                         drop_last=True) if num_workers > 0 else {}
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collator, **loader_kw)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collator)
        else:
            self.val_loader = None
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self.start_step = 0
        self.history = []
        self.val_history = []
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_step = -1
        self._best_model_state = None
        self.last_grad_norm = 0.0
        if resume and checkpoint_path:
            self._resume_from(checkpoint_path)

    def _resume_from(self, path: str):
        """Restore model, optimizer and step from a periodic checkpoint."""
        import os
        resume_path = path + ".resume"
        if not os.path.exists(resume_path):
            print(f"no resume state at {resume_path}; starting from scratch")
            return
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        self.start_step = state.get("step", 0) + 1
        self.history = state.get("history", [])
        self.val_history = state.get("val_history", [])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.best_val_step = state.get("best_val_step", -1)
        self.best_loss = state.get("best_loss", float("inf"))
        self._best_model_state = state.get("best_model_state")
        print(f"resumed from step {self.start_step} "
              f"(best_val={self.best_val_loss:.4f})")

    def _save_resume_state(self, step: int):
        """Atomically write resume state so a crash cannot corrupt it.

        Written to a temporary file and renamed, because a power loss during a
        300MB write would otherwise leave a truncated checkpoint that is worse
        than having none.
        """
        import os
        if not self.checkpoint_path:
            return
        target = self.checkpoint_path + ".resume"
        tmp = target + ".tmp"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "step": step,
            "history": self.history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "best_loss": self.best_loss,
            "best_model_state": self._best_model_state,
            "config": self.model._get_config()
            if hasattr(self.model, "_get_config") else {},
        }, tmp)
        os.replace(tmp, target)

    def _lr_for_step(self, step: int, lr: float):
        """Linear warmup then cosine decay to 0.1 * lr."""
        if step < self.warmup_steps:
            return lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_iters - self.warmup_steps)
        return 0.1 * lr + 0.9 * lr * (1 + float(torch.cos(torch.tensor(progress * 3.141592653589793)))) / 2

    def _set_lr(self, step: int):
        base_lr = self._base_lr
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
        flat_logits = logits.reshape(-1, V)
        flat_mask = mask.reshape(-1)
        denom = flat_mask.sum().clamp(min=1.0)
        loss = F.cross_entropy(flat_logits, y.reshape(-1), reduction="none")
        loss = (loss * flat_mask).sum() / denom
        # Router z-loss (PaLM / ST-MoE): keeps logits from drifting to large
        # magnitudes, which is the main source of low-precision instability.
        if self.z_loss:
            log_z = torch.logsumexp(flat_logits.float(), dim=-1)
            loss = loss + self.z_loss * ((log_z.pow(2) * flat_mask).sum() / denom)
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + 1.0 * aux_loss
        return loss

    def train_step(self, x, y, mask, step: int):
        self._set_lr(step)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.train()

        with torch.amp.autocast("cuda", enabled=self.use_amp,
                                dtype=self.amp_dtype):
            out = self.model(x)
            loss = self._compute_loss(out, y, mask)
            loss = loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)
            self.last_grad_norm = float(grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

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
            with torch.amp.autocast("cuda", enabled=self.use_amp,
                                    dtype=self.amp_dtype):
                out = self.model(x)
                loss = self._compute_loss(out, y, mask)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / count if count else None

    def train(self):
        data_iter = iter(self.loader)
        pbar = tqdm(range(self.start_step, self.max_iters), desc="training",
                    initial=self.start_step, total=self.max_iters)
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
            if self.save_every and step and step % self.save_every == 0:
                self._save_resume_state(step)
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
            "config": self.model._get_config()
            if hasattr(self.model, "_get_config") else {},
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
