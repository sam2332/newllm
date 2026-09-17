"""Print what the model actually says to simple prompts.

Scores hide this. A checkpoint can post a respectable exact-match number on
held-out traces while answering "Say hello." with drivel, because the traces
it does well on are formulaic and the metric only ever compares strings.
This asks plain questions and shows the raw reply, which is the fastest way
to tell a model that is learning from one that is pattern-matching.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/coherence_probe.py \\
        checkpoints_moe_v2/agent_best.pt
"""

import argparse
import sys

sys.path.insert(0, "/home/lmeadows/llm")

PROMPTS = [
    ("greeting", [{"role": "user", "content": "Hello!"}]),
    ("identity", [{"role": "user", "content": "Who are you?"}]),
    ("recall", [
        {"role": "user", "content": "For later: the access code is 7391. Remember it."},
        {"role": "assistant", "content": "Noted - the access code is 7391."},
        {"role": "user", "content": "What is the access code? Reply with just the code."}]),
    ("arithmetic", [{"role": "user", "content": "What is 2 + 2?"}]),
    ("knowledge-python", [{"role": "user", "content": "What is a Python generator?"}]),
    ("knowledge-bash", [{"role": "user", "content": "What does set -e do in a bash script?"}]),
    ("knowledge-science", [{"role": "user", "content": "What is photosynthesis?"}]),
    # The target the project is steered by: right answer is Rayleigh
    # scattering - shorter (blue) wavelengths scatter more off air molecules.
    ("sky", [{"role": "user", "content": "Why is the sky blue?"}]),
    ("principle", [{"role": "user", "content": "What does DRY mean in programming?"}]),
    ("instruction", [{"role": "user", "content": "List three fruits."}]),
    ("system-prompt", [
        {"role": "system", "content": "You always answer in exactly one word."},
        {"role": "user", "content": "What is the capital of France?"}]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from agent.chat import load_checkpoint, load_tokenizer_for
    from agent.local_client import chat_turn
    model = load_checkpoint(args.checkpoint, device=args.device)
    tok = load_tokenizer_for(model)
    print(f"{args.checkpoint}  (temperature {args.temperature})\n")
    for name, msgs in PROMPTS:
        turn = chat_turn(model, tok, msgs, [], device=args.device,
                         num_ctx=4096, max_new=args.max_new,
                         options={"temperature": args.temperature})
        msg = turn["message"]
        out = (msg.get("content") or "").strip()
        calls = msg.get("tool_calls")
        print(f"--- {name} ---")
        print(f"  Q: {msgs[-1]['content']}")
        if calls:
            print(f"  TOOL_CALL: {calls}")
        print(f"  A: {out[:400]!r}\n", flush=True)


if __name__ == "__main__":
    main()
