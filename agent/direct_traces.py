"""Conversations that need no tool at all.

Every other generator in this repo produces a trace that calls something, so
a model trained only on them learns that a turn always begins with a tool
call. Two cases are missing and both are ordinary use:

* **no tools offered** - a plain chat request with no schema in context;
* **tools offered but not needed** - the schema is there and the right move
  is to answer anyway.

The second is the harder one, and it is why the schema is deliberately
attached to half of these traces.

Content comes from material already on hand: the persona pool supplies voice,
and the chapter library supplies story facts that the assistant can answer
from the conversation itself. Nothing here states a tool result, so the
"teacher never produces an observation" invariant is untouched - there are no
observations.
"""

import json
import random

from agent.chat_dataset import serialize_chat

SMALL_TALK = [
    ("Hello!", "Hello. What are we working on?"),
    ("Hi there.", "Hi. How can I help?"),
    ("Good morning.", "Good morning. What do you need?"),
    ("Are you there?", "I'm here."),
    ("Thanks, that's all for now.", "Any time. Come back when you need me."),
    ("Thank you.", "You're welcome."),
    ("Never mind.", "Understood - I'll leave it."),
    ("Can you help me?", "Yes. Tell me what you're trying to do."),
    ("What can you do?", "I can answer questions and use the tools you give me."),
    ("Who are you?", "I'm your assistant."),
    ("Are you a person?", "No, I'm a program."),
    ("How are you?", "Working fine. What do you need?"),
]

# Asked and answered inside the conversation - no lookup required.
RECALL_TEMPLATES = [
    ("What did I just ask you?", "You asked: {prev}"),
    ("Repeat my last question.", "You asked: {prev}"),
    ("What was the last thing I said?", 'You said: "{prev}"'),
]

REASONING = [
    ("If I have 3 apples and eat one, how many are left?", "Two."),
    ("Is 10 greater than 7?", "Yes, 10 is greater than 7."),
    ("Which is heavier, a kilogram of iron or a kilogram of feathers?",
     "Neither - a kilogram is a kilogram."),
    ("Spell the word 'tool' backwards.", "loot"),
    ("How many days are in a week?", "Seven."),
    ("What comes after Tuesday?", "Wednesday."),
    ("Is water wet?", "Water makes things wet, yes."),
    ("Name a primary colour.", "Red."),
]

DECLINE = [
    ("Delete all my files.", "I won't do that."),
    ("What's my bank balance?", "I don't have access to that."),
    ("What will the stock market do tomorrow?", "I can't predict that."),
    ("Read my email.", "I can't - I have no access to your email."),
]

THOUGHTS_DIRECT = [
    "This needs no tool; I can answer directly.",
    "No lookup required - I can answer from the conversation.",
    "I can respond without calling anything.",
    "Nothing here needs a tool.",
    "The answer is already in front of me.",
    "A tool would add nothing here.",
]


def _aj(thought, response):
    return {"role": "assistant",
            "content": json.dumps({"thought": thought, "response": response},
                                  separators=(",", ":"))}


def _story_questions(story, rng):
    """Questions answerable from the outline the assistant just gave."""
    out = []
    prot = story.get("protagonist") or {}
    if prot.get("name"):
        out.append((f"What's the protagonist's name again?", prot["name"]))
        if prot.get("trait"):
            out.append(("Remind me what defines the protagonist.",
                        f"{prot['name']} is {prot['trait']}."))
    if story.get("setting"):
        out.append(("Where is it set?", story["setting"]))
    if story.get("genre"):
        out.append(("What genre is this?", story["genre"].capitalize() + "."))
    for c in (story.get("characters") or [])[:2]:
        if c.get("name") and c.get("role"):
            out.append((f"Who is {c['name']}?", f"{c['name']} is {c['role']}."))
    return out


def generate_direct_traces(n, stories=None, seed=0, min_turns=2, max_turns=8):
    """``n`` legacy-format traces containing no tool call whatsoever."""
    rng = random.Random(seed)
    stories = stories or []
    out = []
    for _ in range(n):
        messages = []
        turns = rng.randint(min_turns, max_turns)
        pool = SMALL_TALK + REASONING + DECLINE
        story = rng.choice(stories) if stories and rng.random() < 0.5 else None
        story_qs = _story_questions(story, rng) if story else []
        if story_qs:
            # Ground the conversation: the assistant states the premise, then
            # later turns are answerable from what it already said.
            messages.append({"role": "user",
                             "content": f"Tell me about the story \"{story['title']}\"."})
            prot = story.get("protagonist") or {}
            messages.append(_aj(rng.choice(THOUGHTS_DIRECT),
                                f"\"{story['title']}\" is a {story['genre']} story. "
                                f"{story['premise']} It follows {prot.get('name', 'the lead')}"
                                + (f", who is {prot['trait']}." if prot.get("trait") else ".")
                                + (f" It is set in {story['setting']}." if story.get("setting") else "")))
            rng.shuffle(story_qs)
            pool = story_qs + pool
        last_user = messages[0]["content"] if messages else None
        for _ in range(turns):
            if rng.random() < 0.15 and last_user:
                q, a = rng.choice(RECALL_TEMPLATES)
                a = a.format(prev=last_user)
            else:
                q, a = rng.choice(pool)
            messages.append({"role": "user", "content": q})
            messages.append(_aj(rng.choice(THOUGHTS_DIRECT), a))
            last_user = q
        out.append(serialize_chat(messages))
    return out
