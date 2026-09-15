"""Post progress to a Discord webhook, so long runs report from anywhere.

Generation, training and evaluation all run detached for hours at a time and
across two machines' worth of GPUs; the only way to know where they are is to
ssh in and tail a log. This puts the same milestones in a chat room.

Configuration lives in ``config.json`` at the repo root (override with
``$NEWLLM_CONFIG``)::

    {"discord": {"webhook_url": "https://discord.com/api/webhooks/...",
                 "username": "newllm", "enabled": true,
                 "min_seconds_between": 2.0}}

The file is gitignored because a webhook URL is a credential: anyone holding
it can post to the channel. ``config.example.json`` is the committed copy.

Three rules this module keeps, in order of importance:

1. **It cannot break the run.** Every failure - no config, bad URL, DNS
   down, Discord 500, rate limit - is swallowed and logged to stderr at
   most once. A training run must not die because a chat room is
   unreachable.
2. **It cannot slow the run.** Posts go out on a daemon thread; the caller
   is never blocked on the network. A step loop calling ``notify`` every
   500 steps must not wait on an HTTP round trip.
3. **It says which machine is talking.** Every message is prefixed with the
   hostname, because "training finished" from one of several boxes is not
   useful on its own.
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "config.json")
MAX_CONTENT = 1900          # Discord's limit is 2000; leave room for the prefix.

_lock = threading.Lock()
_state = {"loaded": False, "cfg": {}, "warned": False, "worker": None,
          "queue": None, "last_sent": 0.0}


def load_config(path: str = None) -> dict:
    """The parsed config, or ``{}`` if there is none. Never raises."""
    path = path or os.environ.get("NEWLLM_CONFIG") or DEFAULT_CONFIG
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _discord_config() -> dict:
    with _lock:
        if not _state["loaded"]:
            _state["cfg"] = load_config()
            _state["loaded"] = True
        cfg = _state["cfg"]
    d = cfg.get("discord") or {}
    # The environment wins, so a one-off run can post somewhere else without
    # editing a file that other processes are reading.
    url = os.environ.get("DISCORD_WEBHOOK_URL") or d.get("webhook_url")
    if not url or d.get("enabled") is False:
        return {}
    return {"url": url,
            "username": d.get("username", "newllm"),
            "min_seconds_between": float(d.get("min_seconds_between", 2.0))}


def _warn_once(msg: str):
    with _lock:
        if _state["warned"]:
            return
        _state["warned"] = True
    print(f"[notify] {msg} (further notify errors are silent)",
          file=sys.stderr, flush=True)


def _post(url: str, payload: dict, timeout: float = 10.0):
    body = json.dumps(payload).encode()
    req = urlrequest.Request(url, data=body,
                             headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _worker_loop(q: "queue.Queue"):
    while True:
        item = q.get()
        if item is None:
            return
        url, payload, gap = item
        # Discord rate-limits webhooks; pace them rather than earning a 429.
        wait = gap - (time.time() - _state["last_sent"])
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                _post(url, payload)
                _state["last_sent"] = time.time()
                break
            except urlerror.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                _warn_once(f"discord POST failed: HTTP {exc.code}")
                break
            except Exception as exc:                        # noqa: BLE001
                _warn_once(f"discord POST failed: {type(exc).__name__}: {exc}")
                break


def _ensure_worker() -> "queue.Queue":
    with _lock:
        if _state["worker"] is None:
            q = queue.Queue(maxsize=256)
            t = threading.Thread(target=_worker_loop, args=(q,), daemon=True,
                                 name="notify-discord")
            t.start()
            _state["worker"], _state["queue"] = t, q
        return _state["queue"]


def notify(message: str, *, tag: str = None, host: bool = True,
           blocking: bool = False) -> bool:
    """Post one line. Returns True if it was accepted for sending.

    Never raises. ``blocking=True`` sends inline, for the last message of a
    process that is about to exit and would otherwise kill the daemon thread
    before it flushes.
    """
    d = _discord_config()
    if not d:
        return False
    text = str(message)
    prefix = f"**{socket.gethostname()}**" if host else ""
    if tag:
        prefix = f"{prefix} `{tag}`" if prefix else f"`{tag}`"
    content = f"{prefix} {text}".strip() if prefix else text
    if len(content) > MAX_CONTENT:
        content = content[:MAX_CONTENT - 3] + "..."
    payload = {"content": content, "username": d["username"]}
    if blocking:
        try:
            _post(d["url"], payload)
            _state["last_sent"] = time.time()
            return True
        except Exception as exc:                            # noqa: BLE001
            _warn_once(f"discord POST failed: {type(exc).__name__}: {exc}")
            return False
    try:
        _ensure_worker().put_nowait((d["url"], payload, d["min_seconds_between"]))
        return True
    except queue.Full:
        return False


def notify_exception(context: str, exc: BaseException) -> bool:
    """Report a crash. Sent blocking: the process is usually on its way out."""
    return notify(f":rotating_light: **{context} failed**\n"
                  f"`{type(exc).__name__}: {exc}`", blocking=True)


def enabled() -> bool:
    return bool(_discord_config())
