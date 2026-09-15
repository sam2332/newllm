"""The notifier must never break or block the run it is reporting on."""

import json
import threading
import time

import agent.notify as notify


def _reset(cfg):
    notify._state.update({"loaded": True, "cfg": cfg, "warned": False,
                          "worker": None, "queue": None, "last_sent": 0.0})


def test_no_config_is_silent_and_false(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({})
    assert notify.enabled() is False
    assert notify.notify("anything") is False


def test_disabled_flag_wins_over_a_present_url(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({"discord": {"webhook_url": "https://example.invalid/hook",
                        "enabled": False}})
    assert notify.notify("quiet please") is False


def test_environment_overrides_the_file(monkeypatch):
    _reset({"discord": {"webhook_url": "https://file.invalid/hook"}})
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://env.invalid/hook")
    assert notify._discord_config()["url"] == "https://env.invalid/hook"


def test_message_carries_host_and_tag(monkeypatch):
    sent = []
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({"discord": {"webhook_url": "https://example.invalid/hook"}})
    monkeypatch.setattr(notify, "_post", lambda url, payload, timeout=10.0:
                        sent.append((url, payload)))
    assert notify.notify("step 500", tag="train", blocking=True) is True
    content = sent[0][1]["content"]
    assert "step 500" in content and "`train`" in content
    assert sent[0][1]["username"] == "newllm"


def test_long_messages_are_truncated_not_rejected(monkeypatch):
    sent = []
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({"discord": {"webhook_url": "https://example.invalid/hook"}})
    monkeypatch.setattr(notify, "_post", lambda url, payload, timeout=10.0:
                        sent.append(payload))
    notify.notify("x" * 5000, blocking=True)
    assert len(sent[0]["content"]) <= notify.MAX_CONTENT
    assert sent[0]["content"].endswith("...")


def test_a_failing_webhook_does_not_raise(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({"discord": {"webhook_url": "https://example.invalid/hook"}})

    def boom(url, payload, timeout=10.0):
        raise OSError("network is down")

    monkeypatch.setattr(notify, "_post", boom)
    assert notify.notify("still fine", blocking=True) is False
    assert notify.notify_exception("training", ValueError("bad")) is False


def test_async_send_does_not_block_the_caller(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    _reset({"discord": {"webhook_url": "https://example.invalid/hook",
                        "min_seconds_between": 0}})
    started = threading.Event()

    def slow(url, payload, timeout=10.0):
        started.set()
        time.sleep(1.0)

    monkeypatch.setattr(notify, "_post", slow)
    t0 = time.time()
    assert notify.notify("async") is True
    assert time.time() - t0 < 0.3, "notify blocked the caller"
    assert started.wait(2.0), "the worker never ran"


def test_bad_config_file_is_treated_as_absent(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text("{not json")
    monkeypatch.setenv("NEWLLM_CONFIG", str(p))
    assert notify.load_config(str(p)) == {}
