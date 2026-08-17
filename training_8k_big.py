"""Big smart 8K-context transformer trainer.

Use gradient accumulation, bigger model, longer training, learning rate schedule,
and a real validation split. Designed to actually learn long-context patterns.
"""

import torch
import time
import random
import math
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from model.transformer import Transformer
from training.dataset import StoryDataset
from training.big_dataset import generate_dataset
from training.trainer import Trainer
from training.device_utils import get_best_device


def make_8k_dataset(num_stories=4096, min_len=4096, max_len=8192):
    stories = generate_dataset(num_stories=num_stories, min_len=min_len,
                               max_len=max_len, seed=42)
    return StoryDataset(stories, max_len=max_len, max_vocab=256)


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float):
    """Cosine warmup + decay schedule."""
    if step < warmup:
        return max_lr * (step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class BigTrainer(Trainer):
    """Trainer with gradient accumulation, LR schedule, validation."""

    def __init__(self, model, train_dataset, val_dataset=None,
                 batch_size: int = 1, lr: float = 1e-3,
                 device: str = None, max_iters: int = 5000,
                 warmup: int = 500, grad_accum: int = 8,
                 eval_every: int = 500):
        super().__init__(model, train_dataset, batch_size=batch_size,
                         lr=lr, device=device, max_iters=max_iters)
        self.val_dataset = val_dataset
        self.full_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
        self.warmup = warmup
        self.grad_accum = grad_accum
        self.eval_every = eval_every
        self.val_history = []
        self.train_history = []
        # rebuild loader to use collate from underlying dataset
        self.loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=self.full_dataset.collate_pad
        )
        # rebuild optimizer with scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            betas=(0.9, 0.95), weight_decay=0.1
        )

    def train(self):
        self.model.train()
        data_iter = iter(self.loader)
        pbar = range(self.max_iters)
        accum_loss = 0.0
        for step in pbar:
            lr = get_lr(step, self.warmup, self.max_iters,
                        self.optimizer.defaults["lr"], 1e-6)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                x, y, mask = next(data_iter)

            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad(set_to_none=False)
            logits = self.model(x)
            if logits.dim() == 4:
                logits = logits[:, :, 0, :]
            B, S, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V), y.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            loss = loss / self.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                avg_loss = accum_loss
                self.train_history.append(avg_loss)
                accum_loss = 0.0
                if step % 10 == 0:
                    print(f"step={step} loss={avg_loss:.4f} lr={lr:.2e}")

            if (step + 1) % self.eval_every == 0 and self.val_dataset is not None:
                val_loss = self.evaluate()
                self.val_history.append((step, val_loss))
                print(f"  VAL step={step} val_loss={val_loss:.4f}")

        return self.train_history

    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, collate_fn=self.full_dataset.collate_pad)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for x, y, mask in loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                logits = self.model(x)
                if logits.dim() == 4:
                    logits = logits[:, :, 0, :]
                B, S, V = logits.shape
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), y.view(-1), reduction="none"
                )
                loss = (loss * mask.view(-1)).sum()
                total_loss += loss.item()
                total_tokens += mask.sum().item()
        self.model.train()
        return total_loss / max(1, total_tokens)


def run_big_8k_training(label: str, attention_type: str,
                        iters: int = 10000, num_stories: int = 4096):
    device = get_best_device()
    print(f"\n=== {label} BIG 8K on {device} ===")
    dataset = make_8k_dataset(num_stories=num_stories)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    print(f"train={len(train_ds)} val={len(val_ds)} max_len=8192")

    model = Transformer(
        vocab_size=256,
        d_model=512,           # bigger brain
        n_layers=8,            # deeper
        n_heads=8,
        d_ff=2048,
        max_len=8192,
        attention_type=attention_type,
        latent_dim=128,        # MLA small KV
        sparse_window=512,     # sparse local window
        use_rope=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params:,}")

    trainer = BigTrainer(
        model, train_ds, val_ds,
        batch_size=1,        # 8k tokens is huge
        lr=1e-3,
        max_iters=iters,
        warmup=int(iters * 0.1),
        grad_accum=8,        # effective batch size 8
        eval_every=500,
        device=device,
    )
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    final_loss = history[-1]
    best_loss = min(history)
    print(f"final={final_loss:.4f} best={best_loss:.4f} time={elapsed:.1f}s")

    prompt = "The hunter track mammoth by footprint across the wide plain."
    generated = trainer.generate(prompt, max_new=200)
    print(f"prompt='{prompt}'")
    print(f"generated='{generated}'")

    return {"loss": final_loss, "best": best_loss, "time": elapsed,
            "train_history": history, "val_history": trainer.val_history}


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    results = {}
    results["mla_8k_big"] = run_big_8k_training("MLA", "mla", iters=5000)

    print("\n=== Big 8K Summary ===")
    for name, r in results.items():
        print(f"{name:15} final={r['loss']:.4f} best={r['best']:.4f} time={r['time']:.1f}s")
