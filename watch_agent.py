"""Tail the latest agent training log and print progress every few seconds."""

import argparse
import os
import sys
import time


def tail_log(path: str, n: int = 10, interval: float = 10.0):
    if not os.path.exists(path):
        print(f"log file not found: {path}")
        return
    print(f"watching {path} (press Ctrl+C to stop)")
    last_size = 0
    try:
        while True:
            size = os.path.getsize(path)
            if size != last_size:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                print("\033[2J\033[H", end="")  # clear screen
                print("".join(lines[-n:]), end="")
                last_size = size
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch agent training log.")
    parser.add_argument("--log", default="agent_run_v3f.log",
                        help="Path to the training log file")
    parser.add_argument("--lines", type=int, default=20,
                        help="Number of trailing lines to display")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Refresh interval in seconds")
    args = parser.parse_args()
    tail_log(args.log, args.lines, args.interval)
