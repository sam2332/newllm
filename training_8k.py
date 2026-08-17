"""Train and evaluate on 8K context length.

Use sparse attention and MLA to fit long sequences on 16GB VRAM.
"""

import torch
import time
import random
from model.transformer import Transformer
from training.dataset import StoryDataset
from training.big_dataset import generate_dataset
from training.trainer import Trainer
from training.device_utils import get_best_device


def make_8k_dataset(num_stories=1024, min_len=2048, max_len=8192):
    stories = generate_dataset(num_stories=num_stories, min_len=min_len,
                               max_len=max_len, seed=42)
    return StoryDataset(stories, max_len=max_len, max_vocab=256)


def run_8k_training(label: str, attention_type: str, iters: int = 2000,
                    num_stories: int = 1024):
    device = get_best_device()
    print(f"\n=== {label} 8K context on {device} ===")
    dataset = make_8k_dataset(num_stories=num_stories)
    print(f"dataset size: {len(dataset)} stories, max_len=8192")

    # grug: for 8k use small model + sparse or MLA to save VRAM
    model = Transformer(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_len=8192,
        attention_type=attention_type,
        latent_dim=32,
        sparse_window=256,
        use_rope=True,
    )

    trainer = Trainer(
        model, dataset,
        batch_size=2,  # small batch for 8k on 16GB
        lr=3e-4,
        max_iters=iters,
        device=device,
    )
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    final_loss = history[-1]
    best_loss = min(history)
    avg_last_50 = sum(history[-50:]) / 50
    print(f"final={final_loss:.4f} best={best_loss:.4f} avg_last_50={avg_last_50:.4f} time={elapsed:.1f}s")

    prompt = "The hunter track mammoth by footprint across the"
    generated = trainer.generate(prompt, max_new=120)
    print(f"prompt='{prompt}'")
    print(f"generated='{generated}'")

    return {"loss": final_loss, "best": best_loss,
            "avg_last_50": avg_last_50, "time": elapsed}


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    results = {}
    results["mla_8k"] = run_8k_training("MLA", "mla", iters=1000)
    results["sparse_8k"] = run_8k_training("Sparse", "sparse", iters=1000)

    print("\n=== 8K Summary ===")
    for name, r in results.items():
        print(f"{name:15} final={r['loss']:.4f} best={r['best']:.4f} "
              f"avg_last_50={r['avg_last_50']:.4f} time={r['time']:.1f}s")
