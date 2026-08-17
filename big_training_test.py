"""Train and evaluate on big synthetic dataset with larger context."""

import torch
import time
import random
from model.transformer import Transformer
from training.dataset import StoryDataset
from training.big_dataset import generate_dataset
from training.trainer import Trainer
from training.device_utils import get_best_device


def make_big_dataset(num_stories=1024, max_len=512):
    stories = generate_dataset(num_stories=num_stories, min_len=max_len // 2,
                               max_len=max_len, seed=42)
    return StoryDataset(stories, max_len=max_len, max_vocab=256)


def run_big_training(config: dict, label: str, max_len: int = 512,
                     num_stories: int = 1024, iters: int = 2000,
                     device: str = None):
    if device is None:
        device = get_best_device()
    print(f"\n=== {label} on {device} ===")
    dataset = make_big_dataset(num_stories, max_len)
    print(f"dataset size: {len(dataset)} stories, max_len={max_len}")

    model = Transformer(
        vocab_size=256,
        d_model=config.get("d_model", 128),
        n_layers=config.get("n_layers", 4),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        max_len=max_len * 2,
        attention_type=config.get("attention_type", "standard"),
        latent_dim=config.get("latent_dim", 32),
        sparse_window=config.get("sparse_window", 64),
        use_moe=config.get("use_moe", False),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        use_rope=True,
        hybrid_pattern=config.get("hybrid_pattern", None),
        predict_n_tokens=config.get("predict_n_tokens", 1),
    )

    trainer = Trainer(
        model, dataset,
        batch_size=config.get("batch_size", 16),
        lr=config.get("lr", 3e-4),
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

    prompt = "The hunter track mammoth by footprint across"
    generated = trainer.generate(prompt, max_new=80)
    print(f"prompt='{prompt}'")
    print(f"generated='{generated}'")

    return {"loss": final_loss, "best": best_loss, "avg_last_50": avg_last_50,
            "time": elapsed, "history": history}


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    RESULTS = {}

    # phase 1: baseline at 512 tokens
    RESULTS["baseline_512"] = run_big_training(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
         "batch_size": 16, "lr": 3e-4},
        "baseline (512 context)",
        max_len=512, num_stories=1024, iters=2000,
    )

    # phase 2: MLA to save memory
    RESULTS["mla_512"] = run_big_training(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
         "attention_type": "mla", "latent_dim": 32,
         "batch_size": 16, "lr": 3e-4},
        "MLA (512 context)",
        max_len=512, num_stories=1024, iters=2000,
    )

    # phase 3: sparse attention on longer context
    RESULTS["sparse_1024"] = run_big_training(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
         "attention_type": "sparse", "sparse_window": 128,
         "batch_size": 8, "lr": 3e-4},
        "sparse (1024 context)",
        max_len=1024, num_stories=1024, iters=2000,
    )

    # phase 4: MoE bigger brain
    RESULTS["moe_512"] = run_big_training(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
         "use_moe": True, "num_experts": 8, "top_k": 2,
         "batch_size": 16, "lr": 3e-4},
        "MoE (512 context)",
        max_len=512, num_stories=1024, iters=2000,
    )

    # phase 5: hybrid SSM+attention
    RESULTS["hybrid_1024"] = run_big_training(
        {"d_model": 128, "n_layers": 6, "n_heads": 4, "d_ff": 512,
         "hybrid_pattern": ["attn", "ssm", "attn", "ssm", "attn", "ssm"],
         "batch_size": 8, "lr": 3e-4},
        "hybrid SSM+attn (1024 context)",
        max_len=1024, num_stories=1024, iters=2000,
    )

    # phase 6: multi-token prediction
    RESULTS["multitoken_512"] = run_big_training(
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
         "predict_n_tokens": 4,
         "batch_size": 16, "lr": 3e-4},
        "multi-token (512 context)",
        max_len=512, num_stories=1024, iters=2000,
    )

    print("\n=== FINAL SUMMARY ===")
    for name, r in RESULTS.items():
        print(f"{name:20} final={r['loss']:.4f} best={r['best']:.4f} "
              f"avg_last_50={r['avg_last_50']:.4f} time={r['time']:.1f}s")
