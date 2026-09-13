"""Observation store: keep big tool results out of the model's context.

Real tasks return large observations - a file, a search result page, a command's
output. A byte-level model pays one token per character, so putting a 3KB result
into context costs 3000 tokens. Across 60 tool calls that is ~189,000 tokens,
which no amount of GPU will make practical.

The fix is the one real agents use: the tool returns whatever it returns, the
store keeps it verbatim, and the *context* gets a bounded preview plus a handle.
If the model needs more it calls ``read_obs`` and pages through - turning a large
observation into more hops rather than more context, which is the resource that
is actually available.

    tool returns 3 KB
      -> store keeps 3 KB under id "obs_4"
      -> context sees ~180 B preview + "[obs_4: 3012 bytes, showing 0-180]"
      -> read_obs(id="obs_4", offset=180) pages in the next window
"""

DEFAULT_PREVIEW = 180
DEFAULT_WINDOW = 400
MAX_STORED = 8000


class ObservationStore:
    """Holds full tool results; hands out bounded previews for the context."""

    def __init__(self, preview_bytes: int = DEFAULT_PREVIEW,
                 window_bytes: int = DEFAULT_WINDOW):
        self.preview_bytes = preview_bytes
        self.window_bytes = window_bytes
        self._items = {}
        self._n = 0

    def add(self, content: str) -> tuple:
        """Store a result and return (obs_id, context_view).

        Small results pass through untouched so short chains stay compact and
        identical to what the model already learned.
        """
        content = str(content)
        if len(content) <= self.preview_bytes:
            return None, content
        self._n += 1
        obs_id = f"obs_{self._n}"
        self._items[obs_id] = content[:MAX_STORED]
        return obs_id, self._view(obs_id, 0)

    def _view(self, obs_id: str, offset: int) -> str:
        content = self._items.get(obs_id, "")
        if not content:
            return f"ERROR: no observation {obs_id}"
        end = min(len(content), offset + self.window_bytes)
        chunk = content[offset:end]
        return (f"{chunk}\n[{obs_id}: {len(content)} bytes, "
                f"showing {offset}-{end}]")

    def read(self, obs_id: str, offset: int = 0) -> str:
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        if obs_id not in self._items:
            known = ", ".join(sorted(self._items)) or "none"
            return f"ERROR: no observation '{obs_id}' (have: {known})"
        return self._view(obs_id, offset)

    def __len__(self) -> int:
        return len(self._items)


def compact(history: list, keep_recent: int = 6) -> list:
    """Collapse old turns so context stays bounded as a chain deepens.

    Beyond ``keep_recent`` tool results, older observations are replaced with a
    one-line note recording that the step happened and what it returned in
    summary. The assistant's own reasoning is kept - the plan is what later
    steps depend on; the raw bytes of step 3 usually are not.
    """
    tool_idx = [i for i, block in enumerate(history)
                if block.startswith("<tool ")]
    if len(tool_idx) <= keep_recent:
        return list(history)
    cutoff = tool_idx[-keep_recent]
    out = []
    for i, block in enumerate(history):
        if i >= cutoff or not block.startswith("<tool "):
            out.append(block)
            continue
        name = block.split("name=", 1)[-1].split(">", 1)[0]
        body = block.split(">", 1)[-1].rsplit("</tool>", 1)[0]
        first = body.strip().splitlines()[0] if body.strip() else ""
        if len(first) > 60:
            first = first[:60] + "..."
        elided = f"<tool name={name}>{first} [elided]</tool>"
        # Never grow. A short observation costs less verbatim than it does with
        # an elision marker bolted on, and compaction that increases the
        # context is worse than no compaction.
        out.append(elided if len(elided) < len(block) else block)
    return out


READ_OBS_SPEC = {
    "read_obs": {
        "description": ("Read more of a large observation that was truncated "
                        "in the transcript."),
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": "Observation id, e.g. obs_3."},
            "offset": {"type": "integer",
                       "description": "Byte offset to read from."}},
            "required": ["id"]},
    },
}


def attach_observation_tool(toolbox, store: ObservationStore):
    """Expose ``read_obs`` bound to a specific store."""
    spec = dict(READ_OBS_SPEC["read_obs"])
    spec["execute"] = lambda args: store.read(args.get("id", ""),
                                              args.get("offset", 0))
    toolbox.tools["read_obs"] = spec
    return toolbox
