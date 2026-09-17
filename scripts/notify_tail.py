"""Post a running job's tqdm progress to the webhook, from outside the job.

A process already running cannot pick up a new notify schedule, so this
reads its log instead: every --every seconds it posts the latest progress
line, and exits once the log stops advancing or reaches 100%.

    screen -dmS sft-notify .venv/bin/python scripts/notify_tail.py logs/sft.log --every 60
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.notify import notify  # noqa: E402

LINE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([^,]+),\s*(.*)\]")


def last_progress(path: str, tail_bytes: int = 4096):
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - tail_bytes))
        text = fh.read().decode("utf-8", "replace")
    for chunk in reversed(re.split(r"[\r\n]", text)):
        m = LINE.search(chunk)
        if m:
            return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--every", type=float, default=60.0, help="seconds between posts")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--stale", type=float, default=300.0,
                    help="exit when the log has not grown for this many seconds")
    args = ap.parse_args()
    last_step = None
    while True:
        if time.time() - os.path.getmtime(args.log) > args.stale:
            notify(f"{args.log} went quiet; watcher exiting", tag=args.tag, blocking=True)
            return
        m = last_progress(args.log)
        if m and m.group(2) != last_step:
            last_step = m.group(2)
            pct, step, total, elapsed, eta, rate, post = m.groups()
            notify(f"step {int(step):,}/{int(total):,} ({pct}%) {post}, "
                   f"{rate.strip()}, elapsed {elapsed.strip()}, ETA {eta.strip()}",
                   tag=args.tag, blocking=True)
            if step == total:
                return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
