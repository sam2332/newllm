"""Turn an Ollama message list into the model's context.

The model was trained on one serialization (see ``agent/chat_dataset.py``):

    <system>{"tools":[...]}</system>        schema block, when tools exist
    <system>free text</system>               client system prompt, stacked after
    <user>...</user>
    <assistant>{"thought":...,"tool_call":{...}}</assistant>
    <tool name=X>observation</tool>
    <assistant>{"thought":...,"response":"..."}</assistant>\x03

Everything here reproduces that form from what an Ollama client sends. Two
things are easy to get wrong and were wrong in the previous server:

* A client system prompt must be *added after* the schema block, not put in
  its place. The schema is how the model knows the tool names; dropping it is
  the 88% -> 0% failure ``tool_schema.py`` exists to prevent.
* The last message in a client-executes-tools loop is a ``role: "tool"``
  message. It must be kept - it is the observation the model is about to
  answer from.

Truncation keeps the system blocks whole and drops the oldest *turns*, never
the head of the context.
"""

import json

from agent.ollama_format import assistants_from_ollama
from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from agent.tool_schema import schema_block_from_tools

EOT = "\x03"


# ------------------------------------------------------------- text transport

def to_model_text(text: str) -> str:
    """UTF-8 bytes as one char each, which is what the byte tokenizer expects.

    ``AgentTokenizer.encode`` maps every char to ``min(ord(c), 255)``, so any
    code point above 255 would collapse to byte 255. Sending the UTF-8 bytes
    keeps the text recoverable.
    """
    return str(text).encode("utf-8").decode("latin-1")


def from_model_text(text: str) -> str:
    """Inverse of ``to_model_text``; invalid sequences become U+FFFD."""
    return str(text).encode("latin-1", "replace").decode("utf-8", "replace")


# ------------------------------------------------------------ system blocks

def system_texts_from_messages(messages: list) -> list:
    return [str(m.get("content") or "") for m in messages
            if m.get("role") == "system" and m.get("content")]


def system_blocks(tools: list, system_texts: list,
                  system_mode: str = "stacked") -> list:
    """Schema block first, then the free-text system prompt.

    That order is the only combined form in the training data. With
    ``system_mode="drop"`` the free text is discarded (an escape hatch for a
    checkpoint that has never seen one).
    """
    blocks = []
    schema = schema_block_from_tools(tools) if tools else ""
    if schema:
        blocks.append(schema)
    if system_mode == "stacked":
        text = "\n".join(t for t in system_texts if t)
        if text:
            blocks.append(f"<system>{to_model_text(text)}</system>")
    return blocks


# --------------------------------------------------------------- turn blocks

def _assistant_block(internal: dict) -> str:
    block = ("<assistant>" + json.dumps(internal, separators=(",", ":"))
             + "</assistant>")
    # A final response ends the turn; the model must learn to yield here.
    if "response" in internal:
        block += EOT
    return block


def _tool_block(name: str, content) -> str:
    return f"<tool name={name}>{to_model_text(content)}</tool>"


def turn_blocks_from_messages(messages: list, thought_cache=None) -> list:
    """Group non-system messages into turns of tagged blocks.

    A turn is a user message plus everything up to the next user message. An
    assistant message carrying N tool calls is rendered as N call/observation
    pairs, matched to the following ``role: "tool"`` messages by ``tool_name``
    when present, else in order - the model's protocol is one call per turn.
    """
    turns = []
    current = None
    pending = []  # tool calls awaiting their observation

    def flush_pending():
        # Calls that never got an observation still go into the context,
        # otherwise the model cannot see what it already tried.
        for internal in pending:
            current.append(_assistant_block(internal))
        pending.clear()

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            if current is not None:
                flush_pending()
            current = [f"<user>{to_model_text(m.get('content') or '')}</user>"]
            turns.append(current)
            continue
        if current is None:
            current = []
            turns.append(current)
        if role == "assistant":
            flush_pending()
            for internal in assistants_from_ollama(m, thought_cache):
                if "tool_call" in internal:
                    pending.append(internal)
                else:
                    current.append(_assistant_block(internal))
        elif role == "tool":
            name = m.get("tool_name") or m.get("name") or ""
            match = None
            for internal in pending:
                if internal["tool_call"]["name"] == name:
                    match = internal
                    break
            if match is None and pending:
                match = pending[0]
            if match is not None:
                pending.remove(match)
                current.append(_assistant_block(match))
                current.append(_tool_block(match["tool_call"]["name"],
                                           m.get("content") or ""))
            else:
                current.append(_tool_block(name, m.get("content") or ""))
    if current is not None:
        flush_pending()
    return turns


# ----------------------------------------------------------------- assembly

def assemble_context(messages: list, tools: list, *, num_ctx: int,
                     reserve: int,
                     tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                     system_mode: str = "stacked",
                     thought_cache=None) -> tuple:
    """Return ``(context_text, token_ids, info)``.

    ``num_ctx - reserve`` is the prompt budget. The system head is never
    truncated; whole turns are dropped oldest-first; the last turn is always
    kept, minus its oldest call/observation pairs if it alone overflows. As a
    last resort the body is cut from the front by tokens, which is logged in
    ``info`` because it means the model is answering from a torn context.
    """
    head = system_blocks(tools, system_texts_from_messages(messages),
                         system_mode=system_mode)
    turns = turn_blocks_from_messages(messages, thought_cache)

    def ntok(blocks):
        return len(tokenizer.encode("\n".join(blocks) + "\n")) if blocks else 0

    budget = max(0, num_ctx - reserve - ntok(head))
    kept = list(turns)
    dropped = 0
    while len(kept) > 1 and ntok([b for t in kept for b in t]) > budget:
        kept.pop(0)
        dropped += 1

    body = [b for t in kept for b in t]
    trimmed_pairs = 0
    if kept and ntok(body) > budget:
        last = list(kept[-1])
        # Keep the user block (index 0) and shed the oldest pairs after it.
        while len(last) > 1 and ntok(last) > budget:
            last.pop(1)
            trimmed_pairs += 1
        body = last

    context = "\n".join(head + body) + "\n"
    tokens = tokenizer.encode(context)
    hard_cut = 0
    limit = num_ctx - reserve
    if limit > 0 and len(tokens) > limit:
        head_tokens = tokenizer.encode("\n".join(head) + "\n") if head else []
        keep = max(0, limit - len(head_tokens))
        body_tokens = tokens[len(head_tokens):]
        hard_cut = len(body_tokens) - keep
        tokens = head_tokens + body_tokens[len(body_tokens) - keep:]
        context = tokenizer.decode(tokens)

    info = {"prompt_tokens": len(tokens), "dropped_turns": dropped,
            "trimmed_pairs": trimmed_pairs, "hard_cut_tokens": hard_cut,
            "system_blocks": len(head)}
    return context, tokens, info
