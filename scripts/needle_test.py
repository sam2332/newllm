"""Needle-in-a-haystack: can the model retrieve a fact from deep in a long
context? Decides whether the 64k target is *earned*, not just enabled.

A conversation of filler turns is built to a target token length, one fact
("the access code is 7391") is planted at 25/50/75% depth, and the final user
turn asks for it. Scored by exact presence of the value in the reply. Runs
through the same client path as serving (``agent/local_client.py``) or, with
``--host``, through a real Ollama server for the GGUF.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/needle_test.py checkpoints_X/agent_best.pt \\
        --lengths 8192 16384 32768 49152 65536
"""

import argparse
import json
import random
import sys

sys.path.insert(0, "/home/lmeadows/llm")

FILLER_Q = ["Tell me something about {topic}.", "What do you know about {topic}?",
            "Give me a short note on {topic}.", "Describe {topic} briefly."]
TOPICS = ["tidal pools", "bread baking", "cargo ships", "chess openings", "old radios",
          "glacier travel", "market gardens", "brass instruments", "desert roads",
          "lighthouses", "typewriters", "river ferries", "night trains", "beekeeping",
          "clockmaking", "salt marshes", "mountain huts", "city buses", "paper mills"]
FILLER_A = ["{topic} reward patience; most of what matters happens slowly and out of sight.",
            "People who work with {topic} tend to keep careful notes, because small details compound.",
            "The history of {topic} is mostly a history of small practical improvements.",
            "If you spend a season around {topic} you learn to read the weather in a new way."]


def build_conversation(rng, tokenizer, target_tokens, depth, code):
    """Filler turns up to ~target_tokens with the needle at the given depth."""
    msgs = [{"role": "system", "content": "You are a helpful assistant with a good memory."}]
    needle_user = f"For later: the access code for the archive room is {code}. Please remember it."
    needle_asst = f"Noted - the archive room access code is {code}."
    total = 0
    turns = []
    while total < target_tokens:
        topic = rng.choice(TOPICS)
        q = rng.choice(FILLER_Q).format(topic=topic)
        a = " ".join(rng.choice(FILLER_A).format(topic=topic.capitalize()) for _ in range(3))
        turns.append((q, a))
        total += len(tokenizer.encode(q + a)) + 12
    k = max(1, int(len(turns) * depth))
    for i, (q, a) in enumerate(turns):
        if i == k:
            msgs += [{"role": "user", "content": needle_user},
                     {"role": "assistant", "content": needle_asst}]
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "What is the access code for the archive room? Reply with just the code."})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--host", default=None, help="score a running Ollama server instead")
    ap.add_argument("--model", default="newllm")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 49152, 65536])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.host:
        import ollama
        from agent.bpe_tokenizer import BPEAgentTokenizer
        client = ollama.Client(host=args.host)
        tok_path = None
        for cand in ("data/tokenizer/bpe8k.json", "data/tokenizer/bpe16k.json"):
            try:
                tokenizer = BPEAgentTokenizer(cand); tok_path = cand; break
            except Exception:                                   # noqa: BLE001
                continue
        if tok_path is None:
            sys.exit("need a BPE tokenizer to size the haystack; pass a checkpoint instead")

        def answer(msgs, n):
            r = client.chat(model=args.model, messages=msgs, stream=False,
                            options={"temperature": 0, "num_ctx": int(n * 1.15) + 512,
                                     "num_predict": 32})
            return r.message.content or ""
    else:
        from agent.chat import load_checkpoint, load_tokenizer_for
        from agent.local_client import chat_turn
        model = load_checkpoint(args.checkpoint, device=args.device)
        tokenizer = load_tokenizer_for(model)

        def answer(msgs, n):
            turn = chat_turn(model, tokenizer, msgs, [], device=args.device,
                             num_ctx=int(n * 1.15) + 512, max_new=32)
            return turn["message"].get("content", "")

    print(f"{'tokens':>7s} " + " ".join(f"d={d:<4}" for d in args.depths))
    grid = {}
    for n in args.lengths:
        row = []
        for d in args.depths:
            hits = 0
            for t in range(args.trials):
                code = str(rng.randint(1000, 9999))
                msgs = build_conversation(rng, tokenizer, n, d, code)
                hits += code in answer(msgs, n)
            row.append(hits)
            grid[(n, d)] = hits
        print(f"{n:7,d} " + " ".join(f"{h}/{args.trials:<4}" for h in row), flush=True)
    json.dump({f"{n}@{d}": h for (n, d), h in grid.items()},
              open("results/needle_latest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
