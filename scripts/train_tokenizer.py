"""Train the BPE tokenizer on a text corpus and report compression.

The corpus should be a sample of the mix the model will train on - the
compression sweep this session found ~3x on tool traces, and it saturates at
about 4k merges there, but prose (personas, chapters) benefits from more.
Train on the mix, then choose the size from the numbers this prints.

    .venv/bin/python scripts/train_tokenizer.py --corpus data/corpus/*.txt \
        --vocab 8192 --out data/tokenizer/bpe8k.json
"""

import argparse
import glob
import random
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.bpe_tokenizer import train_bpe, BPEAgentTokenizer
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER


RS = "\x1e"   # record separator between whole traces (scripts/dump_corpus.py)


def iter_lines(paths, limit=None):
    """Yield one record per trace when the file uses RS, else per line."""
    n = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            blob = fh.read()
        records = blob.split(RS) if RS in blob else blob.split("\n")
        for rec in records:
            rec = rec.strip("\n")
            if rec:
                yield rec
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="text files / globs")
    ap.add_argument("--vocab", type=int, nargs="+", default=[8192])
    ap.add_argument("--out", default="data/tokenizer/bpe{vocab}.json")
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="max training lines")
    args = ap.parse_args()

    paths = sorted(p for g in args.corpus for p in glob.glob(g))
    if not paths:
        sys.exit(f"no files match {args.corpus}")
    lines = list(iter_lines(paths, args.limit or None))
    random.Random(0).shuffle(lines)
    hold, train = lines[:args.holdout], lines[args.holdout:]
    byte_total = sum(len(DEFAULT_AGENT_TOKENIZER.encode(l)) for l in hold)
    print(f"{len(train):,} training lines, {len(hold):,} held out; "
          f"byte-level baseline {byte_total/len(hold):,.0f} tokens/line")
    print(f"{'vocab':>7s} {'tok/line':>9s} {'vs bytes':>9s} {'out':>6s}")
    for vocab in args.vocab:
        out = args.out.format(vocab=vocab)
        tok = train_bpe(iter(train), vocab, out)
        n = sum(len(tok.encode(l)) for l in hold)
        print(f"{vocab:7,d} {n/len(hold):9,.0f} {byte_total/n:8.2f}x  {out}")
        sample = hold[0][:80]
        assert tok.decode(tok.encode(sample)) == sample, "round trip failed"


if __name__ == "__main__":
    main()
