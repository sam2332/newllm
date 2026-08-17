"""Run live training test and report results."""

import torch
import time
from model.transformer import Transformer
from training.dataset import StoryDataset, SAMPLE_STORIES, collate_pad
from training.trainer import Trainer


def make_dataset():
    return StoryDataset(SAMPLE_STORIES, max_len=40, max_vocab=256)


def run_training_experiment(config: dict, label: str):
    dataset = make_dataset()
    model = Transformer(
        vocab_size=256,
        d_model=config.get("d_model", 64),
        n_layers=config.get("n_layers", 2),
        n_heads=config.get("n_heads", 2),
        d_ff=config.get("d_ff", 128),
        max_len=128,
        attention_type=config.get("attention_type", "standard"),
        latent_dim=config.get("latent_dim", 16),
        sparse_window=config.get("sparse_window", 8),
        use_moe=config.get("use_moe", False),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        predict_n_tokens=config.get("predict_n_tokens", 1),
    )
    trainer = Trainer(model, dataset, batch_size=config.get("batch_size", 4),
                      lr=config.get("lr", 5e-3),
                      max_iters=config.get("max_iters", 300))
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start
    final_loss = history[-1]
    print(f"\n[{label}] final_loss={final_loss:.4f} time={elapsed:.2f}s")
    prompt = "grug hunt"
    generated = trainer.generate(prompt, max_new=20)
    print(f"[{label}] prompt='{prompt}' -> '{generated}'")
    return final_loss, elapsed


if __name__ == "__main__":
    configs = {
        "standard": {"d_model": 64, "n_layers": 2, "n_heads": 2,
                     "d_ff": 128, "max_iters": 300},
        "mla": {"d_model": 64, "n_layers": 2, "n_heads": 2,
                "d_ff": 128, "attention_type": "mla", "latent_dim": 16,
                "max_iters": 300},
        "sparse": {"d_model": 64, "n_layers": 2, "n_heads": 2,
                   "d_ff": 128, "attention_type": "sparse",
                   "sparse_window": 8, "max_iters": 300},
        "moe": {"d_model": 64, "n_layers": 2, "n_heads": 2,
                "d_ff": 128, "use_moe": True, "num_experts": 4,
                "top_k": 2, "max_iters": 300},
        "hybrid": {"d_model": 64, "n_layers": 4, "n_heads": 2,
                   "d_ff": 128, "hybrid_pattern": ["attn", "ssm", "attn", "ssm"],
                   "max_iters": 300},
        "multi_token": {"d_model": 64, "n_layers": 2, "n_heads": 2,
                        "d_ff": 128, "predict_n_tokens": 3,
                        "max_iters": 300},
    }

    results = {}
    for name, cfg in configs.items():
        loss, elapsed = run_training_experiment(cfg, name)
        results[name] = {"loss": loss, "time": elapsed}

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:12} loss={r['loss']:.4f} time={r['time']:.2f}s")
