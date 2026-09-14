"""Load a trained agent checkpoint and chat or test tool use."""

import argparse
import os
import torch

from model.transformer import Transformer
from agent.agent_loop import run_agent, EOT
from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.tools import Toolbox
from sampling import Sampler


def load_checkpoint(path: str, device: str = "cuda") -> Transformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    sd = ckpt["model"]
    vocab_size = cfg.get("vocab_size", 256)
    current_vocab = DEFAULT_AGENT_TOKENIZER.vocab_size
    if vocab_size < DEFAULT_AGENT_TOKENIZER.LEGACY_VOCAB_SIZE:
        raise ValueError(
            f"Checkpoint vocabulary is {vocab_size}, but the JSON agent requires "
            f"at least {DEFAULT_AGENT_TOKENIZER.LEGACY_VOCAB_SIZE}. This is a "
            "legacy checkpoint; train a fresh JSON-agent checkpoint in a "
            "separate directory."
        )
    # A checkpoint from before the chat tokens were appended is still usable:
    # ids 0-270 kept their meaning, so the model is built at its own vocabulary
    # size rather than being rejected.
    if vocab_size != current_vocab and not cfg.get("tokenizer"):
        print(f"note: checkpoint vocab {vocab_size} != current {current_vocab}; "
              f"use AgentTokenizer(max_vocab={vocab_size}) with this checkpoint "
              f"(see load_tokenizer_for)")
    d_model = cfg.get("d_model", 1024)
    # Derive layer count from the weights when the config predates it.
    n_layers = cfg.get("n_layers") or len(
        {k.split(".")[1] for k in sd if k.startswith("layers.")})
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
        # Default to 1: a checkpoint without this key predates arch_version=2.
        arch_version=cfg.get("arch_version", 1),
        n_kv_heads=cfg.get("n_kv_heads"),
        qk_norm=cfg.get("qk_norm", True),
        rope_base=cfg.get("rope_base", 10000.0),
        use_moe=cfg.get("use_moe", False),
        num_experts=cfg.get("num_experts", 4),
        top_k=cfg.get("top_k", 2),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    model.tokenizer_spec = cfg.get("tokenizer")
    model.checkpoint_dir = os.path.dirname(os.path.abspath(path))
    print(f"Loaded checkpoint from {path} ({model.count_parameters():,} params)")
    return model


def load_tokenizer_for(model):
    """Return the tokenizer this checkpoint was trained with.

    A checkpoint that records ``config["tokenizer"]`` gets exactly that
    (the BPE file is also copied next to the weights). One that predates the
    key is a byte-level checkpoint, restricted to its own vocabulary so the
    tokenizer cannot emit ids the model has no embedding row for.
    """
    spec = getattr(model, "tokenizer_spec", None)
    if spec:
        from agent.tokenizer_registry import load_tokenizer
        return load_tokenizer(spec, getattr(model, "checkpoint_dir", None))
    size = getattr(model, "vocab_size", DEFAULT_AGENT_TOKENIZER.vocab_size)
    if size == DEFAULT_AGENT_TOKENIZER.vocab_size:
        return DEFAULT_AGENT_TOKENIZER
    return AgentTokenizer(max_vocab=size)


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
                           sampler=sampler, max_steps=5, max_new=4096,
                           tokenizer=load_tokenizer_for(model))
        _print_trace(result)


def test_mode(model: Transformer, toolbox: Toolbox, device: str, questions: list):
    print(f"\nRunning {len(questions)} test questions...")
    for q in questions:
        result = run_agent(model, q, toolbox, device=device, greedy=True,
                           max_steps=5, max_new=4096,
                           tokenizer=load_tokenizer_for(model))
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
        default=os.environ.get("CHECKPOINT", "checkpoints_v2_M/agent_best.pt"),
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
            "How many planets are there times 2?",
            "Look up the number of planets, add 5, then multiply by 2.",
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
