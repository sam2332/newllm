"""Compose long multi-turn "project" traces over a virtual workspace.

A project is a conversation of up to ~52 user turns in which the assistant
writes a multi-chapter story to files, answers questions about it, revises
chapters, and - the skill that matters - **reads a file back** when the
answer is no longer in the context. No context window holds 26 chapters; the
filesystem is the model's long-term memory, and this is where it learns to
use it.

Everything the assistant says is grounded in a real observation from
``VirtualWorkspace`` (write/read/list results come from the same code the
live tool runs) or in the outline/chapter library text itself. Prose comes
from the teacher-written library (``scripts/gen_chapters_ollama.py``); it is
composed, never generated here, so tens of thousands of arcs cost no teacher
time.

Output is the legacy tagged trace text, so the rest of the pipeline
(``randomize_trace`` -> ``trace_to_messages`` -> ``chatml.render``) applies
unchanged, including tool/parameter-name randomization.
"""

import json
import random

from agent.chat_dataset import serialize_chat
from agent.virtual_workspace import VirtualWorkspace

THOUGHTS = {
    "outline": ["The user wants to start a story; I'll save the outline to a file so we can refer back to it.",
                "Best to persist the plan first. Writing outline.md.",
                "I'll store the outline in the workspace before drafting anything.",
                "Saving the outline now so later chapters stay consistent with it."],
    "write": ["Chapter {i} goes into its own file so the conversation stays short.",
              "I'll write chapter {i} to {path} rather than paste it here.",
              "Drafting chapter {i} and saving it to the workspace.",
              "Time to write chapter {i}; it belongs in a file."],
    "read_outline": ["I no longer have the outline in view; I'll read outline.md.",
                     "Let me check the outline file before answering.",
                     "The plan is in outline.md - reading it back.",
                     "I should confirm this against the saved outline."],
    "read_chapter": ["That detail is in {path}; I'll read the file rather than guess.",
                     "I don't want to misremember - reading {path}.",
                     "Let me look at {path} to answer precisely.",
                     "The text of that chapter is on disk; reading it back."],
    "revise": ["I'll read the current chapter, then save the revised version.",
               "To revise safely I need the existing text first.",
               "Reading {path} so the edit keeps everything else intact."],
    "list": ["I'll list the workspace to see what exists.",
             "Checking which files we have so far.",
             "Let me enumerate the files before answering."],
    "answer": ["I have what I need to answer.", "The file gave me the answer.",
               "That settles it.", "Now I can respond accurately."],
    "direct": ["This needs no tool; I can answer from the conversation.",
               "No file access needed for this one.",
               "I can reply directly."],
}

START_PHRASINGS = [
    "Let's write a {genre} story called \"{title}\". {premise} Save an outline first.",
    "I want to work on a {genre} novel, \"{title}\". Here's the idea: {premise} Put the outline in a file.",
    "New project: \"{title}\" ({genre}). {premise} Start by writing the plan to outline.md.",
    "Help me draft \"{title}\", a {genre} story. {premise} Save the chapter plan before we begin.",
]
CHAPTER_PHRASINGS = [
    "Write chapter {i}.", "Next chapter please.", "Go ahead with chapter {i}: {summary}",
    "Draft chapter {i} now.", "Let's do chapter {i}.", "Continue with the next chapter.",
    "Chapter {i} - keep it consistent with the outline.",
]
QUESTION_KINDS = ["character_role", "chapter_summary", "protagonist", "setting", "chapter_opening"]
QUESTION_PHRASINGS = {
    "character_role": ["Remind me, what is {name}'s role?", "Who is {name} again?",
                       "What role does {name} play in the story?"],
    "chapter_summary": ["What happens in chapter {i}?", "Summarize chapter {i} for me.",
                        "Remind me what chapter {i} was about."],
    "protagonist": ["Who is the protagonist and what defines them?",
                    "What's our main character's name and trait?"],
    "setting": ["Where is the story set?", "Remind me of the setting."],
    "chapter_opening": ["How does chapter {i} open?", "Quote the first sentence of chapter {i}.",
                        "What's the opening line of chapter {i}?"],
}
REVISE_PHRASINGS = ["Add a final line to chapter {i}: \"{line}\"",
                    "Append this closing sentence to chapter {i}: {line}",
                    "End chapter {i} with the sentence: {line}"]
CLOSING_LINES = ["The night held its breath.", "Nothing would be the same after this.",
                 "Somewhere, a door closed.", "It was only the beginning.",
                 "The lights went out one by one."]
LIST_PHRASINGS = ["What files do we have so far?", "List the workspace.", "Show me what's been saved."]
RECAP_PHRASINGS = ["Give me a recap of everything so far.", "Summarize the whole story to this point.",
                   "Write a few paragraphs recapping the chapters we have."]


def _aj(thought, response=None, tool_call=None):
    obj = {"thought": thought}
    if tool_call is not None:
        obj["tool_call"] = tool_call
    else:
        obj["response"] = response
    return {"role": "assistant", "content": json.dumps(obj, separators=(",", ":"))}


def _outline_text(story):
    lines = [f"Title: {story['title']}", f"Genre: {story['genre']}",
             f"Premise: {story['premise']}",
             f"Protagonist: {story['protagonist']['name']} - {story['protagonist']['trait']}",
             f"Setting: {story.get('setting', '')}", "Characters:"]
    lines += [f"- {c['name']}: {c['role']}" for c in story.get("characters", [])]
    lines.append("Chapters:")
    lines += [f"{i+1}. {c['title']}: {c['summary']}" for i, c in enumerate(story["chapters"])]
    return "\n".join(lines)


class ProjectComposer:
    def __init__(self, stories, rng: random.Random, char_budget=60000):
        self.stories = [s for s in stories if s.get("chapters")
                        and all(c.get("text") for c in s["chapters"])]
        self.rng = rng
        self.char_budget = char_budget

    def _t(self, kind, **kw):
        return self.rng.choice(THOUGHTS[kind]).format(**kw)

    def compose(self, max_turns=52) -> list:
        """Return the message list for one arc (user/assistant/tool dicts)."""
        rng = self.rng
        story = rng.choice(self.stories)
        ws = VirtualWorkspace()
        msgs = []
        chars = 0
        outline = _outline_text(story)
        chapter_paths = {}
        written = []             # chapter indices already on disk
        in_context = set()       # chapter files whose text is still "recent"

        def add_user(text):
            nonlocal chars
            msgs.append({"role": "user", "content": text})
            chars += len(text)

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

        # turn 1: outline
        add_user(rng.choice(START_PHRASINGS).format(genre=story["genre"], title=story["title"],
                                                    premise=story["premise"]))
        call("write_file", {"path": "outline.md", "content": outline}, "outline")
        respond(f"Saved the outline for \"{story['title']}\" to outline.md with "
                f"{len(story['chapters'])} planned chapters.")
        turns = 1
        n_ch = len(story["chapters"])
        next_chapter = 0
        while turns < max_turns and chars < self.char_budget:
            turns += 1
            # weight actions: write chapters early, questions/revisions later
            options = []
            if next_chapter < n_ch:
                options += ["chapter"] * 4
            if written:
                options += ["question"] * 3 + ["revise", "list", "recap"]
            if not options:
                break
            action = rng.choice(options)
            if action == "chapter":
                i = next_chapter
                ch = story["chapters"][i]
                path = f"ch{i+1:02d}.md"
                add_user(rng.choice(CHAPTER_PHRASINGS).format(i=i + 1, summary=ch["summary"]))
                # After a few turns the outline has scrolled away; read it back
                # before writing so continuity is a tool habit, not a memory trick.
                if i > 0 and rng.random() < 0.6:
                    call("read_file", {"path": "outline.md"}, "read_outline")
                call("write_file", {"path": path, "content": ch["text"]}, "write", i=i + 1, path=path)
                words = len(ch["text"].split())
                respond(f"Chapter {i+1}, \"{ch['title']}\", is written and saved to {path} "
                        f"({words} words).", "answer")
                chapter_paths[i] = path
                written.append(i)
                in_context = {path}
                next_chapter += 1
            elif action == "question":
                kind = rng.choice(QUESTION_KINDS)
                if kind == "character_role" and story.get("characters"):
                    c = rng.choice(story["characters"])
                    add_user(rng.choice(QUESTION_PHRASINGS[kind]).format(name=c["name"]))
                    obs = call("read_file", {"path": "outline.md"}, "read_outline")
                    respond(f"{c['name']} is {c['role']}." if f"- {c['name']}: {c['role']}" in obs
                            else f"According to the outline, {c['name']}: {c['role']}.")
                elif kind in ("protagonist", "setting"):
                    add_user(rng.choice(QUESTION_PHRASINGS[kind]))
                    call("read_file", {"path": "outline.md"}, "read_outline")
                    if kind == "protagonist":
                        p = story["protagonist"]
                        respond(f"The protagonist is {p['name']}, defined by being {p['trait']}.")
                    else:
                        respond(f"The story is set in {story.get('setting', 'the outlined setting')}.")
                elif kind == "chapter_summary":
                    i = rng.choice(written)
                    add_user(rng.choice(QUESTION_PHRASINGS[kind]).format(i=i + 1))
                    call("read_file", {"path": "outline.md"}, "read_outline")
                    respond(f"Chapter {i+1}, \"{story['chapters'][i]['title']}\": "
                            f"{story['chapters'][i]['summary']}")
                else:  # chapter_opening: must read the chapter file itself
                    i = rng.choice(written)
                    path = chapter_paths[i]
                    add_user(rng.choice(QUESTION_PHRASINGS[kind]).format(i=i + 1))
                    obs = call("read_file", {"path": path, "start": 0}, "read_chapter", path=path)
                    first = ws.files[path].strip().split("\n")[0]
                    first_sentence = first.split(". ")[0].strip()
                    if not first_sentence.endswith("."):
                        first_sentence += "."
                    respond(f"Chapter {i+1} opens with: \"{first_sentence}\"")
            elif action == "revise":
                i = rng.choice(written)
                path = chapter_paths[i]
                line = rng.choice(CLOSING_LINES)
                add_user(rng.choice(REVISE_PHRASINGS).format(i=i + 1, line=line))
                call("read_file", {"path": path, "start": max(0, len(ws.files[path]) - 400)},
                     "revise", path=path)
                new_text = ws.files[path].rstrip() + "\n\n" + line
                call("write_file", {"path": path, "content": new_text}, "write", i=i + 1, path=path)
                respond(f"Chapter {i+1} now ends with \"{line}\" - saved to {path}.")
            elif action == "list":
                add_user(rng.choice(LIST_PHRASINGS))
                obs = call("list_files", {"pattern": "*"}, "list")
                names = [l.split("  ")[0] for l in obs.splitlines()]
                respond("So far we have: " + ", ".join(names) + ".")
            elif action == "recap":
                add_user(rng.choice(RECAP_PHRASINGS))
                call("read_file", {"path": "outline.md"}, "read_outline")
                parts = [f"Chapter {i+1} ({story['chapters'][i]['title']}): "
                         f"{story['chapters'][i]['summary']}" for i in written]
                recap = (f"\"{story['title']}\" is a {story['genre']} story. {story['premise']}\n\n"
                         + "\n\n".join(parts))
                respond(recap)
        return msgs


def generate_project_traces(stories, n, seed=0, max_turns=52, char_budget=60000,
                            min_turns=6, system_prompts=None):
    """``n`` legacy-format traces. ``system_prompts`` (optional list of str)
    are attached to a random half of the traces as free-text system blocks."""
    rng = random.Random(seed)
    comp = ProjectComposer(stories, rng, char_budget)
    out = []
    for _ in range(n):
        turns = rng.randint(min_turns, max_turns)
        msgs = comp.compose(max_turns=turns)
        system = rng.choice(system_prompts) if system_prompts and rng.random() < 0.5 else None
        out.append(serialize_chat(msgs, system=system))
    return out
