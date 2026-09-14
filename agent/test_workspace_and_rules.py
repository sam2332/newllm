"""Workspace tools: real and virtual backends must behave identically, and
neither may escape its root. Format rules: apply() must satisfy check()."""

import os
import tempfile

import pytest

from agent.format_rules import RULES, apply, verify
from agent.tools import Toolbox
from agent.virtual_workspace import VirtualWorkspace
from agent.workspace_tools import (MAX_READ_CHARS, RealWorkspace, WorkspaceError,
                                   attach_workspace_tools)


@pytest.fixture(params=["real", "virtual"])
def ws(request):
    if request.param == "real":
        return RealWorkspace(tempfile.mkdtemp())
    return VirtualWorkspace()


def test_write_read_append_list_round_trip(ws):
    assert ws.write_file("ch/one.md", "héllo") == "wrote 6 bytes to ch/one.md"
    assert ws.read_file("ch/one.md") == "héllo"
    assert ws.append_file("ch/one.md", " world") == \
        "appended 6 bytes to ch/one.md (now 12 bytes)"
    assert ws.list_files("ch/*.md") == "ch/one.md  12 bytes"
    assert ws.list_files("*.txt") == "(no files)"


def test_backends_produce_identical_observations():
    real, virt = RealWorkspace(tempfile.mkdtemp()), VirtualWorkspace()
    ops = [("write_file", {"path": "a.txt", "content": "x" * 50}),
           ("append_file", {"path": "a.txt", "content": "y"}),
           ("read_file", {"path": "a.txt", "start": 40}),
           ("read_file", {"path": "missing.txt"}),
           ("list_files", {}),
           ("write_file", {"path": "../escape.txt", "content": "no"})]
    for name, args in ops:
        assert real.specs()[name]["execute"](args) == virt.specs()[name]["execute"](args), name


def test_paths_cannot_escape(ws):
    for bad in ("../x", "/etc/passwd", "~/x", "a/../../x", ""):
        with pytest.raises(WorkspaceError):
            ws.write_file(bad, "no")


def test_real_workspace_never_writes_outside_root():
    root = tempfile.mkdtemp()
    ws = RealWorkspace(root)
    ws.write_file("deep/er/file.txt", "ok")
    assert os.path.isfile(os.path.join(root, "deep", "er", "file.txt"))
    assert not os.path.exists(os.path.join(os.path.dirname(root), "escape.txt"))


def test_long_reads_are_paged(ws):
    ws.write_file("big.txt", "a" * (MAX_READ_CHARS + 500))
    first = ws.read_file("big.txt")
    assert first.startswith("a" * 100) and "truncated: 500 more" in first
    assert f"start={MAX_READ_CHARS}" in first
    assert ws.read_file("big.txt", start=MAX_READ_CHARS) == "a" * 500


def test_attach_to_toolbox_and_run_json():
    tb = attach_workspace_tools(Toolbox(), VirtualWorkspace())
    assert {"write_file", "append_file", "read_file", "list_files"} <= set(tb.tools)
    out = tb.run_json({"tool": "write_file", "args": {"path": "n.md", "content": "hi"}})
    assert out == "wrote 2 bytes to n.md"
    assert tb.run_json({"tool": "read_file", "args": {"path": "n.md"}}) == "hi"
    assert tb.run_json({"tool": "read_file", "args": {}}).lower().startswith("error")


def test_every_rule_apply_satisfies_check():
    plain = "The river is 3530 km long. It flows north. Many cities sit on it."
    for rule in RULES:
        out = apply(rule["id"], plain)
        assert verify(rule["id"], out), (rule["id"], out)
        assert len(rule["prompts"]) >= 3


def test_plain_text_fails_the_strict_rules():
    plain = "The river is 3530 km long. It flows north."
    for rid in ("json_only", "markdown_heading", "bullet_list", "numbered_steps",
                "uppercase", "prefix_answer", "suffix_signoff", "brackets"):
        assert not verify(rid, plain), rid
