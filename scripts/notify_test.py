"""Check that the Discord webhook in config.json actually works.

Worth running once before trusting a nine-hour unattended run to it.

    .venv/bin/python scripts/notify_test.py
    .venv/bin/python scripts/notify_test.py --message "hello from the 5090"
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.notify import DEFAULT_CONFIG, enabled, load_config, notify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="test message from scripts/notify_test.py")
    args = ap.parse_args()

    path = os.environ.get("NEWLLM_CONFIG") or DEFAULT_CONFIG
    cfg = load_config(path)
    if not cfg:
        print(f"no config at {path}\n"
              f"  cp config.example.json config.json   # then paste your webhook URL")
        return 1
    d = cfg.get("discord") or {}
    url = os.environ.get("DISCORD_WEBHOOK_URL") or d.get("webhook_url", "")
    # Never print the URL: it is a credential, and this output tends to end
    # up pasted into issues and chat.
    shown = f"{url[:36]}..." if len(url) > 36 else ("(none)" if not url else "(set)")
    print(f"config      : {path}")
    print(f"webhook     : {shown}")
    print(f"username    : {d.get('username', 'newllm')}")
    print(f"enabled     : {enabled()}")
    if not enabled():
        print("\nnothing to send: set discord.webhook_url, or enabled is false")
        return 1
    ok = notify(args.message, tag="test", blocking=True)
    print(f"\ndelivered   : {ok}")
    if not ok:
        print("the POST failed - see the [notify] line above for the reason")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
