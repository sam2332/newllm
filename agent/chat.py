"""Load a trained agent checkpoint and chat or test tool use."""

import argparse
import os
import torch

from model.transformer import Transformer
from agent.agent_loop import run_agent, EOT
from agent.tools import Toolbox
from sampling import Sampler


def load_checkpoint(path: str, device: str = "cuda") -> Transformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    sd = ckpt["model"]
    vocab_size = cfg.get("vocab_size", 256)
    d_model = cfg.get("d_model", 1024)
    n_layers = cfg.get("n_layers", 12)
    max_len = cfg.get("max_len", 512)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=cfg.get("n_heads", 16),
        d_ff=cfg.get("d_ff", 4096),
        max_len=max_len,
        attention_type=cfg.get("attention_type", "standard"),
        use_rope=cfg.get("use_rope", True),
        dropout=cfg.get("dropout", 0.1),
        tie_weights=cfg.get("tie_weights", False),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {path} ({model.count_parameters():,} params)")
    return model


def _print_trace(result: dict):
    print("\n--- thinking ---")
    print(result.get("thinking", "<none>"))
    print("--- steps ---")
    for step in result.get("steps", []):
        print(f"  {step['tool']}[{step['arg']}] -> {step['result']}")
    print("--- answer ---")
    print(result.get("final_answer", "<none>"))
    print("--- raw trace ---")
    print(result.get("trace", "").replace(EOT, ""))
    print("-----------------\n")


def chat_mode(model: Transformer, toolbox: Toolbox, device: str):
    sampler = Sampler(temperature=0.5, top_k=20, top_p=0.9, repetition_penalty=1.0)
    print("\nAgent chat mode. Ask a question, or type 'exit'.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        result = run_agent(model, question, toolbox, device=device,
                           sampler=sampler, max_steps=5, max_new=400)
        _print_trace(result)


def test_mode(model: Transformer, toolbox: Toolbox, device: str, questions: list):
    print(f"\nRunning {len(questions)} test questions...")
    for q in questions:
        result = run_agent(model, q, toolbox, device=device, greedy=True,
                           max_steps=5, max_new=400)
        print(f"Q: {q}")
        print(f"A: {result['final_answer']}")
        if result.get("thinking"):
            print(f"T: {result['thinking'][:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Chat with the trained agent or test its tool use."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/agent_best.pt",
        help="Path to the model checkpoint to load",
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["chat", "test"],
        default="chat",
        help="Run in chat mode or run a fixed test battery",
    )
    parser.add_argument(
        "--questions", "-q",
        nargs="*",
        default=[
            "What is 12 + 8?",
            "Calculate 15 * 4.",
            "What is the project?",
            "Retrieve the leader.",
            "What is the current date and time?",
            "What is the capital of france?",
            "Who is the president of the united states?",
            "How many planets are there?",
            "What is the speed of light?",
            "What is the largest planet?",
            "What is the capital of japan plus 5?",
            "How many planets are there times 2?",
        ],
        help="Questions for test mode",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train an agent first with `python -m agent.train_agent`."
        )

    model = load_checkpoint(args.checkpoint, device=args.device)
    toolbox = Toolbox()

    if args.mode == "chat":
        chat_mode(model, toolbox, args.device)
    else:
        test_mode(model, toolbox, args.device, args.questions)


if __name__ == "__main__":
    main()
