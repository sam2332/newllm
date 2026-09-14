"""Grammar-constrained decoding for the assistant JSON protocol.

HANDOFF.md lists "constrained decoding for tool names and JSON keys" as the
top architectural experiment. This implements it as a logit mask driven by a
small state machine over the assistant message grammar:

    <assistant>{"thought":<str>,"tool_call":{"name":<tool>,"arguments":<obj>}}</assistant>
    <assistant>{"thought":<str>,"response":<str>}</assistant>

Structural positions admit exactly one token, so they cost the model nothing.
String bodies and argument objects stay free-form - the model still has to
produce the right *content*, it just can no longer produce malformed JSON or
hallucinate a tool that does not exist.
"""

import torch

from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER

# Grammar states
S_OPEN = "open"                # expect <assistant>
S_LBRACE = "lbrace"            # expect {
S_THOUGHT_KEY = "thought_key"  # expect "thought":
S_STR_OPEN = "str_open"        # expect opening quote of a string
S_IN_STR = "in_str"            # inside a string body
S_AFTER_THOUGHT = "after_th"   # expect ,
S_ACTION_KEY = "action_key"    # expect "tool_call": or "response":
S_TC_LBRACE = "tc_lbrace"      # expect { of tool_call
S_NAME_KEY = "name_key"        # expect "name":
S_NAME_OPEN = "name_open"      # expect opening quote of tool name
S_IN_NAME = "in_name"          # inside the tool name (constrained alphabet)
S_AFTER_NAME = "after_name"    # expect ,
S_ARGS_KEY = "args_key"        # expect "arguments":
S_IN_ARGS = "in_args"          # free-form balanced JSON object
S_TC_RBRACE = "tc_rbrace"      # expect } closing tool_call
S_RBRACE = "rbrace"            # expect } closing the message
S_CLOSE = "close"              # expect </assistant>
S_DONE = "done"


class AssistantGrammar:
    """Tracks grammar state and produces an allowed-token mask per step."""

    def __init__(self, tool_names, tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                 allow_tool_call: bool = True):
        self.tok = tokenizer
        self.vocab = tokenizer.vocab_size
        self.tool_names = sorted(tool_names)
        # A request that offers no tools must get a "response" turn; with an
        # empty name list the S_IN_NAME state would otherwise admit only the
        # closing quote and force an empty tool name.
        self.allow_tool_call = allow_tool_call and bool(self.tool_names)
        self.sid = {t: tokenizer.encode(t)[0] for t in tokenizer.SPECIAL_TOKENS}
        self.reset()

    def accepts(self, token_id: int) -> bool:
        """Whether ``token_id`` is admissible in the current state.

        Lets the grammar run as a passive tracker over tokens it did not mask:
        a server streaming an unconstrained model uses this to know when the
        output has left the grammar and deltas can no longer be trusted.
        """
        return token_id in self.allowed()

    def reset(self):
        self.state = S_OPEN
        self.escape = False          # previous char was a backslash
        self.depth = 0               # brace depth inside arguments
        self.in_arg_string = False
        self.name_prefix = ""        # partial tool name so far
        self.after_string = None     # state to enter when a string closes
        return self

    # -- allowed token ids for the current state -------------------------
    def _byte(self, ch):
        return ord(ch)

    def allowed(self):
        s = self.state
        if s == S_OPEN:
            return [self.sid["<assistant>"]]
        if s == S_LBRACE:
            return [self._byte("{")]
        if s == S_THOUGHT_KEY:
            return [self.sid['"thought":']]
        if s in (S_STR_OPEN, S_NAME_OPEN):
            return [self._byte('"')]
        if s == S_IN_STR:
            if self.escape:
                return list(range(32, 127))
            # any printable byte; the quote closes the string
            return list(range(32, 256))
        if s == S_AFTER_THOUGHT:
            return [self._byte(",")]
        if s == S_ACTION_KEY:
            if not self.allow_tool_call:
                return [self.sid['"response":']]
            return [self.sid['"tool_call":'], self.sid['"response":']]
        if s == S_TC_LBRACE:
            return [self._byte("{")]
        if s == S_NAME_KEY:
            return [self.sid['"name":']]
        if s == S_IN_NAME:
            # only bytes that keep the prefix on a real tool name
            nxt = set()
            for name in self.tool_names:
                if name.startswith(self.name_prefix):
                    if len(name) == len(self.name_prefix):
                        nxt.add(self._byte('"'))
                    else:
                        nxt.add(self._byte(name[len(self.name_prefix)]))
            return sorted(nxt) or [self._byte('"')]
        if s == S_AFTER_NAME:
            return [self._byte(",")]
        if s == S_ARGS_KEY:
            return [self.sid['"arguments":']]
        if s == S_IN_ARGS:
            if self.depth == 0:
                return [self._byte("{")]
            return list(range(32, 256))
        if s in (S_TC_RBRACE, S_RBRACE):
            return [self._byte("}")]
        if s == S_CLOSE:
            return [self.sid["</assistant>"]]
        return [self.sid["</assistant>"]]

    def mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Return logits with disallowed tokens set to -inf."""
        allow = self.allowed()
        out = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(allow, device=logits.device, dtype=torch.long)
        out.index_copy_(-1, idx, logits.index_select(-1, idx))
        return out

    # -- advance the machine on the chosen token -------------------------
    def advance(self, token_id: int):
        s = self.state
        text = self.tok.decode([token_id])
        is_special = token_id >= 256

        if s == S_OPEN:
            self.state = S_LBRACE
        elif s == S_LBRACE:
            self.state = S_THOUGHT_KEY
        elif s == S_THOUGHT_KEY:
            self.state, self.after_string = S_STR_OPEN, S_AFTER_THOUGHT
        elif s == S_STR_OPEN:
            self.state = S_IN_STR
        elif s == S_IN_STR:
            if self.escape:
                self.escape = False
            elif text == "\\":
                self.escape = True
            elif text == '"':
                self.state = self.after_string
        elif s == S_AFTER_THOUGHT:
            self.state = S_ACTION_KEY
        elif s == S_ACTION_KEY:
            if is_special and text == '"response":':
                self.state, self.after_string = S_STR_OPEN, S_RBRACE
            else:
                self.state = S_TC_LBRACE
        elif s == S_TC_LBRACE:
            self.state = S_NAME_KEY
        elif s == S_NAME_KEY:
            self.state = S_NAME_OPEN
        elif s == S_NAME_OPEN:
            self.state, self.name_prefix = S_IN_NAME, ""
        elif s == S_IN_NAME:
            if text == '"':
                self.state = S_AFTER_NAME
            else:
                self.name_prefix += text
        elif s == S_AFTER_NAME:
            self.state = S_ARGS_KEY
        elif s == S_ARGS_KEY:
            self.state, self.depth, self.in_arg_string = S_IN_ARGS, 0, False
        elif s == S_IN_ARGS:
            if self.escape:
                self.escape = False
            elif text == "\\":
                self.escape = True
            elif text == '"':
                self.in_arg_string = not self.in_arg_string
            elif not self.in_arg_string:
                if text == "{":
                    self.depth += 1
                elif text == "}":
                    self.depth -= 1
                    if self.depth == 0:
                        self.state = S_TC_RBRACE
        elif s == S_TC_RBRACE:
            self.state = S_RBRACE
        elif s == S_RBRACE:
            self.state = S_CLOSE
        elif s == S_CLOSE:
            self.state = S_DONE
        return self

    @property
    def done(self) -> bool:
        return self.state == S_DONE
