"""Tokenize a large text dataset into packed uint16 shards for pretraining.

The coherence probe found a model that recites its training set and cannot
repeat a four-digit code from two turns earlier. The corpus explains it: 260k
traces composed from a few thousand library items, 78.8% of sentences
repeats of another sentence, distinct-8 of 0.288. No amount of instruction
data fixes that, because the missing thing is *language* - the model never
saw enough ordinary text to learn how sentences work, only enough of ours to
learn how ours look.

This streams a Hugging Face text dataset (FineWeb-Edu by default), tokenizes
with the project's BPE, and packs the ids end to end into flat uint16 shards
with the EOT token as a document separator. Packed shards rather than padded
examples because pretraining wants every token to be a training token: at
sequence 2048 a padded corpus of short documents wastes most of the batch.

    .venv/bin/python scripts/build_pretrain_shards.py \\
        --tokenizer data/tokenizer/bpe8192_v2.json \\
        --target-tokens 10_000_000_000 --out data/pretrain

uint16 holds a vocabulary up to 65,535, which our 8,192 comfortably fits, and
halves both the disk and the read bandwidth against int32.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "/home/lmeadows/llm")

_TOK = None
_EOT_ID = None


def _eot_id(tok):
    """The id, not the string. ``tok.eot`` is "<|im_end|>" for the BPE
    tokenizer, and writing a str into a uint16 array raises - or worse,
    silently writes the wrong thing across ten billion tokens."""
    eot = getattr(tok, "eot", None)
    if isinstance(eot, int):
        return eot
    ids = tok.encode(eot if isinstance(eot, str) else "<|im_end|>")
    if len(ids) != 1:
        raise SystemExit(f"EOT did not encode to a single token: {ids}")
    return ids[0]


def _init(tokenizer_path):
    global _TOK, _EOT_ID
    from agent.tokenizer_registry import load_tokenizer
    _TOK = load_tokenizer(tokenizer_path)
    _EOT_ID = _eot_id(_TOK)


def _encode(texts):
    """Encode a batch in a worker; returns one flat id list per document."""
    out = []
    for t in texts:
        if not t:
            continue
        ids = _TOK.encode(t)
        ids.append(_EOT_ID)
        out.append(ids)
    return out


def batched(stream, n, key="text"):
    buf = []
    for row in stream:
        buf.append(row.get(key) or "")
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe8192_v2.json")
    ap.add_argument("--target-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--shard-tokens", type=int, default=500_000_000,
                    help="tokens per shard file (~1 GB at uint16)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 8))
    ap.add_argument("--batch", type=int, default=256, help="documents per worker task")
    ap.add_argument("--out", default="data/pretrain")
    args = ap.parse_args()

    from datasets import load_dataset

    from agent.notify import notify

    from agent.tokenizer_registry import load_tokenizer
    probe = load_tokenizer(args.tokenizer)
    if probe.vocab_size > 65535:
        raise SystemExit(f"vocab {probe.vocab_size} does not fit uint16 shards")
    print(f"tokenizer {args.tokenizer}: vocab {probe.vocab_size}, "
          f"eot id {_eot_id(probe)}")
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)

    def _tok_str(n):
        return f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.0f}M"

    notify(f"pretrain shard build started: {args.dataset} `{args.config}`, "
           f"target {_tok_str(args.target_tokens)} tokens, {args.workers} workers",
           tag="pretrain")
    t0 = time.time()
    total = 0
    docs = 0
    shard_idx = 0
    buf = np.empty(args.shard_tokens + 1_000_000, dtype=np.uint16)
    fill = 0
    manifest = []

    def flush(final=False):
        nonlocal fill, shard_idx
        if fill == 0:
            return
        path = os.path.join(args.out, f"shard_{shard_idx:04d}.bin")
        buf[:fill].tofile(path)
        manifest.append({"path": os.path.basename(path), "tokens": int(fill)})
        print(f"  wrote {path} ({fill/1e6:.1f}M tokens, "
              f"{total/1e9:.2f}B total, {(time.time()-t0)/60:.1f} min)", flush=True)
        notify(f"shard {shard_idx} written: {fill/1e6:.0f}M tokens "
               f"({total/1e9:.2f}B / {args.target_tokens/1e9:.1f}B, "
               f"{(time.time()-t0)/3600:.1f}h elapsed)", tag="pretrain")
        shard_idx += 1
        fill = 0

    # ProcessPoolExecutor, not multiprocessing.Pool. When a Pool worker dies
    # the parent blocks in imap_unordered forever waiting for a result that
    # will never come: this build hung for 1h50m with every child a zombie
    # and the parent in futex_do_wait, having written 9.5B perfectly good
    # tokens and no manifest. An Executor raises BrokenProcessPool instead,
    # so the shards and the manifest survive a dead worker.
    stalled = False
    try:
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(args.tokenizer,)) as pool:
            stream = batched(ds, args.batch, key=args.text_key)
            # A bounded window of futures, submitted by hand. Executor.map
            # consumes its whole input iterable up front, which against an
            # endless stream queues the entire dataset before yielding one
            # result; Pool.imap is lazy but deadlocks when a worker dies.
            # This keeps laziness AND gets an exception instead of a hang.
            inflight = deque()
            depth = args.workers * 3
            exhausted = False
            while True:
                while not exhausted and len(inflight) < depth:
                    try:
                        inflight.append(pool.submit(_encode, next(stream)))
                    except StopIteration:
                        exhausted = True
                if not inflight:
                    break
                for ids in inflight.popleft().result():
                    n = len(ids)
                    if fill + n > len(buf):
                        flush()
                    buf[fill:fill + n] = np.asarray(ids, dtype=np.uint16)
                    fill += n
                    total += n
                    docs += 1
                if fill >= args.shard_tokens:
                    flush()
                if total >= args.target_tokens:
                    for f in inflight:
                        f.cancel()
                    break
    except (BrokenExecutor, OSError) as exc:                 # noqa: BLE001
        # Keep what was tokenized: a partial corpus is still a corpus, and
        # 9.5B of 10B tokens is not worth throwing away over a dead worker.
        stalled = True
        print(f"\n  worker pool broke ({type(exc).__name__}: {exc}); "
              f"keeping {total/1e9:.2f}B tokens", flush=True)
        notify(f":warning: shard build lost its worker pool at "
               f"{total/1e9:.2f}B tokens - keeping what was written",
               tag="pretrain")
    flush(final=True)

    del ds
    meta = {"dataset": args.dataset, "config": args.config,
            "tokenizer": args.tokenizer, "tokens": int(total),
            "documents": int(docs), "shards": manifest, "dtype": "uint16",
            "complete": not stalled}
    json.dump(meta, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    mins = (time.time() - t0) / 60
    print(f"\n{total/1e9:.2f}B tokens from {docs:,} documents in {mins:.1f} min "
          f"({total/max(1e-9, mins*60)/1e6:.1f}M tokens/s) -> {args.out}")
    notify(f":white_check_mark: **pretrain corpus ready** {total/1e9:.2f}B tokens, "
           f"{docs:,} docs, {len(manifest)} shards in {mins/60:.1f}h -> `{args.out}`",
           tag="pretrain", blocking=True)


if __name__ == "__main__":
    main()
