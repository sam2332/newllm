"""Train mini-mind MoE on clustered story data.

Each story type gets routed to its own small transformer expert.
"""

import torch
import random
import time
from collections import defaultdict
from model.sequence_moe import SequenceMoE
from training.dataset import StoryDataset
from training.big_dataset import generate_dataset
from training.trainer import Trainer
from training.device_utils import get_best_device
from results_logger import save_result


def make_typed_dataset(num_stories=4096, max_len=512):
    """Generate stories and label them by dominant topic."""
    all_stories = generate_dataset(num_stories=num_stories, min_len=max_len // 2,
                                   max_len=max_len, seed=42)
    typed = defaultdict(list)
    topic_words = {
        "hunt": ["hunt", "mammoth", "deer", "track", "hunter", "spear", "kill"],
        "fire": ["fire", "warm", "cook", "ash", "ember", "flame"],
        "river": ["river", "water", "fish", "lake", "stream", "drink"],
        "sky": ["star", "moon", "sun", "sky", "night", "storm", "thunder"],
    }
    for story in all_stories:
        scores = {topic: sum(story.count(w) for w in words)
                  for topic, words in topic_words.items()}
        topic = max(scores, key=scores.get)
        typed[topic].append(story)
    return typed


def train_sequence_moe(num_experts=4, iters=3000):
    device = get_best_device()
    print(f"\n=== Sequence MoE on {device} ===")
    typed = make_typed_dataset(4096, 512)
    print({k: len(v) for k, v in typed.items()})

    model = SequenceMoE(
        vocab_size=256,
        num_experts=num_experts,
        top_k=1,
        d_model=256,
        expert_layers=3,
        expert_heads=4,
        d_ff=512,
        max_len=512,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"SequenceMoE parameters: {total:,}")

    # grug: make a balanced dataset from all topics
    stories = []
    for topic, items in typed.items():
        stories.extend(items[: min(len(items), 600)])
    dataset = StoryDataset(stories, max_len=512, max_vocab=256)
    trainer = Trainer(model, dataset, batch_size=8, lr=1e-3,
                      max_iters=iters, device=device)
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    final_loss = history[-1]
    print(f"final_loss={final_loss:.4f} time={elapsed:.1f}s")

    # check clustering
    prompts = {
        "hunt": "The hunter track mammoth by footprint",
        "fire": "Fire warm cave and cook meat good",
        "river": "River give cold water and many fish",
        "sky": "Star show path when night dark",
    }
    from sampling import Sampler
    sampler = Sampler(temperature=0.8, top_k=20, top_p=0.9,
                      repetition_penalty=1.2)
    generations = {}
    cluster_assignments = {}
    for topic, text in prompts.items():
        tokens = torch.tensor([[min(ord(c), 255) for c in text]],
                              dtype=torch.long, device=device)
        cluster = model.cluster_assignments(tokens).item()
        generated = trainer.generate(text, max_new=80, sampler=sampler)
        generations[topic] = generated
        cluster_assignments[topic] = cluster
        print(f"[{topic}]> expert={cluster} gen='{generated}'")

    result = {
        "num_experts": num_experts,
        "model_parameters": total,
        "vocab_size": 256,
        "max_len": 512,
        "d_model": 256,
        "expert_layers": 3,
        "n_heads": 4,
        "d_ff": 512,
        "learning_rate": 1e-3,
        "iters": iters,
        "final_loss": final_loss,
        "best_loss": min(history),
        "avg_last_50": sum(history[-50:]) / max(1, len(history[-50:])),
        "elapsed_seconds": elapsed,
        "topic_counts": {k: len(v) for k, v in typed.items()},
        "cluster_assignments": cluster_assignments,
        "generations": generations,
        "history": history,
    }
    path = save_result("sequence_moe", result)
    print(f"results saved to {path}")
    return result


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    train_sequence_moe(num_experts=4, iters=3000)
