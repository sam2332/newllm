"""Compose subject-knowledge traces from the teacher's knowledge library.

Everything else in this repo teaches *mechanics*: route a tool, hold a
persona, keep a 52-turn project straight. None of it teaches what a
generator is, why ``set -e`` matters, what entropy means, or why you would
prefer composition to inheritance. A model that routes tools perfectly and
cannot answer those is not a coding assistant.

Two shapes are composed from ``data/knowledge.json``
(``scripts/gen_knowledge_ollama.py``):

* **explain** - a multi-turn Q&A conversation with **no tool call at all**,
  wandering between related topics and following up. Together with
  ``agent/direct_traces.py`` this is what stops the model believing a turn
  must begin with a tool call.
* **artifact** - the user asks for a script, the assistant writes it to the
  workspace, and later turns ask about the file *after it has scrolled out
  of context*, so the answer requires a ``read_file``. The code and the
  answers are the teacher's; every observation comes from
  ``VirtualWorkspace``, so no tool result is fabricated.

Mixed arcs interleave the two: a file is written, then a general question
about the same domain is answered directly with no tool. That contrast -
same conversation, tool for the file, no tool for the concept - is the
discrimination the model actually has to learn.
"""

import json
import random

from agent.chat_dataset import serialize_chat
from agent.virtual_workspace import VirtualWorkspace

THOUGHTS = {
    "direct": ["This is a knowledge question; no tool can answer it better than I can.",
               "Nothing to look up - I can explain this directly.",
               "No file or tool involved here, just an explanation.",
               "I know this; answering without a tool.",
               "A tool would add nothing - this is a concept question."],
    "followup": ["A follow-up on the same idea; still no lookup needed.",
                 "Continuing the explanation.",
                 "This builds on what I just said.",
                 "Same topic, so I can answer straight away."],
    "write": ["I'll write the {lang} file to {path} rather than paste it into the chat.",
              "Saving this to {path} so it can be run and edited.",
              "Writing {path} into the workspace.",
              "This belongs in a file; creating {path}."],
    "read": ["{path} is no longer in view - I'll read it back before answering.",
             "I don't want to misquote the code; reading {path}.",
             "Let me look at {path} so the answer matches what's actually there.",
             "Checking {path} rather than trusting my memory of it."],
    "answer": ["The file tells me what I need.", "That settles it.",
               "Now I can answer precisely.", "I have the code in front of me."],
    "saved": ["Saved - I'll tell the user what it does and where it is.",
              "Written. Summarising it for the user.",
              "The file is on disk; reporting back.",
              "Done - confirming the path and size."],
    "list": ["I'll list the workspace to see what we've built.",
             "Checking which files exist.",
             "Let me enumerate the files first."],
}

ASK_FILE_PHRASINGS = [
    "{request}", "{request} Save it to a file.",
    "{request} Put it in the workspace so I can run it.",
    "Can you {lower} Write it to a file.",
]
LIST_PHRASINGS = ["What files have we got?", "List the workspace.",
                  "Show me what's been saved so far."]
RUN_PHRASINGS = ["How do I run it?", "What's the command to run this?"]


def _aj(thought, response=None, tool_call=None):
    obj = {"thought": thought}
    if tool_call is not None:
        obj["tool_call"] = tool_call
    else:
        obj["response"] = response
    return {"role": "assistant", "content": json.dumps(obj, separators=(",", ":"))}


def _lower_request(text: str) -> str:
    """"Write me a script that ..." -> "write me a script that ...", so it can
    be dropped into "Can you {lower}"."""
    t = text.strip()
    if not t:
        return t
    t = t[0].lower() + t[1:]
    return t if t.endswith(("?", ".", "!")) else t + "."


class KnowledgeComposer:
    def __init__(self, items, rng: random.Random, char_budget=40000):
        self.rng = rng
        self.char_budget = char_budget
        self.explain = [i for i in items if i.get("kind") == "explain"
                        and i.get("question") and i.get("answer")]
        self.artifact = [i for i in items if i.get("kind") == "artifact"
                         and i.get("code") and i.get("followups")]
        # Topic neighbours: a conversation that stays in one domain reads like
        # a real session, and it is the case where remembering earlier turns
        # actually matters.
        self.by_domain = {}
        for it in self.explain:
            self.by_domain.setdefault(it["domain"], []).append(it)

    def _t(self, kind, **kw):
        return self.rng.choice(THOUGHTS[kind]).format(**kw)

    # -- shape 1: no tools at all -------------------------------------------
    def compose_explain(self, max_turns=12):
        rng = self.rng
        if not self.explain:
            return []
        first = rng.choice(self.explain)
        pool = self.by_domain.get(first["domain"]) or self.explain
        msgs, chars, turns = [], 0, 0
        item = first
        while turns < max_turns and chars < self.char_budget:
            msgs.append({"role": "user", "content": item["question"]})
            msgs.append(_aj(self._t("direct"), response=item["answer"]))
            chars += len(item["question"]) + len(item["answer"]) + 80
            turns += 1
            for f in item.get("followups", []):
                if turns >= max_turns or chars >= self.char_budget:
                    break
                msgs.append({"role": "user", "content": f["question"]})
                msgs.append(_aj(self._t("followup"), response=f["answer"]))
                chars += len(f["question"]) + len(f["answer"]) + 80
                turns += 1
            if rng.random() < 0.45:
                break
            item = rng.choice(pool)
        return msgs

    # -- shape 2: write a file, then answer about it from disk --------------
    def compose_artifact(self, max_turns=24):
        rng = self.rng
        if not self.artifact:
            return []
        ws = VirtualWorkspace()
        msgs, chars, turns = [], 0, 0
        written = []             # (item, path)
        in_context = None        # the path whose text is still recent

        def call(name, args, kind, **kw):
            nonlocal chars
            msgs.append(_aj(self._t(kind, **kw), tool_call={"name": name, "arguments": args}))
            obs = ws.specs()[name]["execute"](args)
            msgs.append({"role": "tool", "name": name, "content": obs})
            chars += len(json.dumps(args)) + len(obs) + 80
            return obs

        def respond(text, kind="answer", **kw):
            nonlocal chars
            msgs.append(_aj(self._t(kind, **kw), response=text))
            chars += len(text) + 60

        def add_file(item):
            nonlocal in_context, turns
            path = item["filename"]
            if any(p == path for _, p in written):
                path = f"{len(written)+1:02d}_{path}"
            msgs.append({"role": "user", "content": rng.choice(ASK_FILE_PHRASINGS).format(
                request=item["request"], lower=_lower_request(item["request"]))})
            call("write_file", {"path": path, "content": item["code"]}, "write",
                 lang=item.get("language", "code"), path=path)
            lines = item["code"].count("\n") + 1
            summary = item.get("summary") or "Done."
            respond(f"{summary} Saved to {path} ({lines} lines).", "saved")
            written.append((item, path))
            in_context = path
            turns += 1

        add_file(rng.choice(self.artifact))
        while turns < max_turns and chars < self.char_budget:
            options = ["ask"] * 5 + ["concept"] * 2 + ["list", "another"]
            action = rng.choice(options)
            if action == "another" and len(written) < 4:
                add_file(rng.choice(self.artifact))
                continue
            if action == "list":
                msgs.append({"role": "user", "content": rng.choice(LIST_PHRASINGS)})
                obs = call("list_files", {}, "list")
                respond("The workspace holds: " + obs.replace("\n", ", ") + ".")
                turns += 1
                in_context = None
                continue
            if action == "concept" and self.explain:
                # Same conversation, a question no file can answer: the model
                # has to NOT reach for read_file here.
                dom = written[-1][0]["domain"]
                pool = self.by_domain.get(dom) or self.explain
                it = rng.choice(pool)
                msgs.append({"role": "user", "content": it["question"]})
                respond(it["answer"], "direct")
                chars += len(it["answer"])
                turns += 1
                continue
            item, path = rng.choice(written)
            fu = rng.choice(item["followups"])
            msgs.append({"role": "user", "content": fu["question"]})
            # The file is only still "in context" if it was the last thing
            # written; otherwise the honest move is to read it back.
            if path != in_context or rng.random() < 0.35:
                call("read_file", {"path": path}, "read", path=path)
                in_context = path
            respond(fu["answer"])
            turns += 1
        return msgs


def generate_knowledge_traces(items, n, seed=0, explain_fraction=0.5,
                              max_turns=24, char_budget=40000):
    """``n`` legacy-format traces: half pure Q&A, half workspace code arcs."""
    rng = random.Random(seed)
    comp = KnowledgeComposer(items, rng, char_budget=char_budget)
    out = []
    for _ in range(n):
        if rng.random() < explain_fraction or not comp.artifact:
            msgs = comp.compose_explain(max_turns=min(max_turns, 12))
        else:
            msgs = comp.compose_artifact(max_turns=max_turns)
        if msgs:
            out.append(serialize_chat(msgs))
    return out
