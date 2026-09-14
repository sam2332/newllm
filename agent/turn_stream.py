"""Incremental decoding of one assistant turn, for streaming.

The model emits ``<assistant>{"thought":"...","response":"..."}</assistant>``
one token at a time. A streaming client wants the *response text* as it is
written, not the JSON envelope - which means knowing, per token, whether the
byte belongs to the thought string, the response string, or the structure.

``AssistantGrammar`` already is that state machine. Here it runs passively:
it is never used to mask logits, only asked ``accepts(token)`` and then
advanced. The moment the model emits something the grammar would not have
allowed, the tracker marks itself desynced and stops emitting deltas, because
a client cannot retract text; the final chunk reconciles whatever was missed.

JSON string bodies arrive with escapes split across tokens (``\\u`` now, the
hex digits later) and non-ASCII text arrives as UTF-8 bytes one per token, so
the unescaper works in bytes and only releases what is complete.
"""

import codecs

from agent.constrained import (AssistantGrammar, S_IN_STR, S_ACTION_KEY,
                               S_AFTER_THOUGHT)

_SIMPLE_ESCAPES = {'"': b'"', "\\": b"\\", "/": b"/", "b": b"\b",
                   "f": b"\f", "n": b"\n", "r": b"\r", "t": b"\t"}


def _is_hex(s: str) -> bool:
    return len(s) == 4 and all(c in "0123456789abcdefABCDEF" for c in s)


class JsonStringUnescaper:
    """Feed the raw chars of a JSON string body; get decoded text back.

    Input chars are the byte tokenizer's chars (code points 0-255, i.e. raw
    UTF-8 bytes) plus JSON escapes. Output is real text. Anything that is not
    yet decodable - a lone backslash, ``\\u`` with fewer than four digits, a
    high surrogate without its pair, a UTF-8 lead byte without its
    continuation bytes - is held until it is.
    """

    def __init__(self):
        self.pending = ""
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, chars: str) -> str:
        self.pending += chars
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        s = self.pending
        out = bytearray()
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c != "\\":
                out.append(ord(c) & 0xFF)
                i += 1
                continue
            if i + 1 >= n:
                break                                   # lone backslash
            e = s[i + 1]
            if e != "u":
                out += _SIMPLE_ESCAPES.get(e, e.encode("utf-8"))
                i += 2
                continue
            if i + 6 > n:
                break                                   # \u with <4 digits
            hexs = s[i + 2:i + 6]
            if not _is_hex(hexs):
                out += s[i:i + 6].encode("utf-8", "replace")
                i += 6
                continue
            cp = int(hexs, 16)
            if 0xD800 <= cp <= 0xDBFF:                  # high surrogate
                if i + 12 > n:
                    if final:
                        out += "�".encode("utf-8")
                        i += 6
                        continue
                    break
                low = s[i + 8:i + 12]
                if s[i + 6:i + 8] == "\\u" and _is_hex(low) \
                        and 0xDC00 <= int(low, 16) <= 0xDFFF:
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (int(low, 16) - 0xDC00)
                    out += chr(cp).encode("utf-8")
                    i += 12
                else:
                    out += "�".encode("utf-8")
                    i += 6
                continue
            if 0xDC00 <= cp <= 0xDFFF:                  # stray low surrogate
                out += "�".encode("utf-8")
            else:
                out += chr(cp).encode("utf-8")
            i += 6
        if final:
            # Whatever is left cannot be completed; emit it literally.
            out += s[i:].encode("utf-8", "replace")
            self.pending = ""
        else:
            self.pending = s[i:]
        return self._utf8.decode(bytes(out), final=final)


class TurnTracker:
    """Classify each generated token and emit safe text deltas.

    ``feed(token_id)`` returns a list of ``(stream, text)`` pairs where
    ``stream`` is ``"thinking"`` or ``"content"``. ``kind`` becomes
    ``"tool_call"`` or ``"response"`` as soon as the action key is seen.
    """

    def __init__(self, tokenizer, allowed_names=None):
        names = list(allowed_names or [])
        self.tok = tokenizer
        self.grammar = AssistantGrammar(names, tokenizer,
                                        allow_tool_call=bool(names))
        self.kind = None
        self.desynced = False
        self.streamed = {"thinking": "", "content": ""}
        self._un = {"thinking": JsonStringUnescaper(),
                    "content": JsonStringUnescaper()}
        self._closed = set()

    def feed(self, token_id: int) -> list:
        if self.desynced:
            return []
        g = self.grammar
        if not g.accepts(token_id):
            self.desynced = True
            return []
        events = []
        state = g.state
        text = self.tok.decode([token_id])
        if state == S_IN_STR:
            stream = "thinking" if g.after_string == S_AFTER_THOUGHT else "content"
            closing = (text == '"' and not g.escape)
            if closing:
                out = self._un[stream].flush()
                self._closed.add(stream)
            else:
                out = self._un[stream].push(text)
            if out:
                self.streamed[stream] += out
                events.append((stream, out))
        elif state == S_ACTION_KEY:
            self.kind = "tool_call" if text == '"tool_call":' else "response"
        g.advance(token_id)
        return events

    def remainder(self, stream: str, full_value: str) -> str:
        """Text the client has not seen, given the parsed final value.

        Normally the parsed value extends what was streamed. When it does not
        (a desync mid-string), the whole value is returned so nothing is lost
        - the client will see a duplicate prefix, which is the lesser evil.
        """
        seen = self.streamed.get(stream, "")
        if full_value.startswith(seen):
            return full_value[len(seen):]
        return full_value


class ChatMLTracker:
    """Delta emitter for the Qwen3 template.

    Every structural marker is a single special token, so the state machine
    runs on token ids: ``<think>`` opens the thinking stream, ``</think>``
    closes it, ``<tool_call>`` suspends deltas (the call is emitted whole at
    the end), and any end-of-turn marker finishes. Newline runs are held back
    until the next non-newline text so the template's ``\\n`` padding around
    ``</think>`` and after it never reaches the client as content.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        sid = tokenizer.special_id
        self._think, self._think_close = sid("<think>"), sid("</think>")
        self._tool, self._tool_close = sid("<tool_call>"), sid("</tool_call>")
        self._ends = {sid("<|im_end|>"), sid("<|im_start|>"), sid("<|endoftext|>")}
        self.mode = "start"
        self.kind = None
        self.desynced = False
        self.streamed = {"thinking": "", "content": ""}
        self._hold = ""

    def feed(self, token_id: int) -> list:
        if self.mode == "done":
            return []
        if token_id == self._think:
            self.mode, self._hold = "thinking", ""
            return []
        if token_id == self._think_close:
            self.mode, self._hold = "after_think", ""
            return []
        if token_id == self._tool:
            self.mode, self._hold, self.kind = "tool", "", "tool_call"
            return []
        if token_id == self._tool_close:
            self.mode = "after_tool"
            return []
        if token_id in self._ends:
            self.mode = "done"
            return []
        if self.mode in ("tool", "after_tool"):
            return []
        text = self.tok.decode([token_id])
        if self.mode == "after_think":
            if text.strip("\n") == "":
                return []                      # the template's "\n\n" padding
            self.mode = "content"
        if self.mode == "start":
            self.mode = "content"
        stream = "thinking" if self.mode == "thinking" else "content"
        # Mirror parse_assistant exactly: the thought is stripped of newlines,
        # the content of all whitespace. Otherwise the streamed text and the
        # final parsed value disagree by a leading space and the client sees
        # the whole answer twice.
        chars = "\n" if stream == "thinking" else None
        combined = self._hold + text
        if not self.streamed[stream]:
            combined = combined.lstrip(chars)
        emit = combined.rstrip(chars)
        self._hold = combined[len(emit):]
        if not emit:
            return []
        if stream == "content":
            self.kind = self.kind or "response"
        self.streamed[stream] += emit
        return [(stream, emit)]

    def remainder(self, stream: str, full_value: str) -> str:
        seen = self.streamed.get(stream, "")
        if full_value.startswith(seen):
            return full_value[len(seen):]
        return full_value
