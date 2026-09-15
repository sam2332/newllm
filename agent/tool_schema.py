"""Tool schemas in context, with randomized names, so the model reads them.

The model previously saw only "<user>question</user>". Tool names lived in its
weights, so renaming calc -> compute_expression took the battery from 88.2% to
0.0%: it still emitted "calc", which no longer existed. A model like that can
never serve a user who brings their own tools.

The fix is the one real function-calling APIs use - pass the tools per request:

    <system>{"tools":[{"name":...,"description":...,"parameters":{...}}]}</system>
    <user>What is 12 + 8?</user>
    <assistant>{"thought":"...","tool_call":{"name":"<a name from the schema>",...}}

Training randomizes the surface form every time: names are drawn from
synonymous, generic and deliberately opaque pools, descriptions are paraphrased,
unused distractor tools are included, and the order is shuffled. Memorizing a
name is therefore useless - the only way to answer is to read the schema and
copy the matching name. Opaque names like "xq7_run" matter most: they make the
description the sole signal, so the model cannot pattern-match on a plausible
identifier either.
"""

import json
import random
import re

# Several surface names per capability, spanning plausible to opaque.
NAME_POOLS = {
    "calc": ["calc", "compute_expression", "evaluate_math", "do_arithmetic",
             "math_eval", "calculator", "arith", "solve_expr", "fn_compute",
             "tool_math", "xq7_eval", "op_42", "zx_calc"],
    "search_memory": ["search_memory", "lookup_stored_value", "recall",
                      "get_memory", "read_store", "fetch_saved", "memo_get",
                      "kv_lookup", "tool_recall", "mm_fetch", "q3_store"],
    "web_search": ["web_search", "query_knowledge_base", "lookup_fact",
                   "search_web", "find_online", "kb_query", "fact_find",
                   "external_lookup", "tool_search", "wz_query", "v9_find"],
    "now": ["now", "current_timestamp", "get_time", "clock", "today",
            "read_clock", "time_now", "tool_clock", "tk_now", "d4_time"],
    "run_bash": ["run_bash", "shell", "execute_shell", "bash_exec",
                 "run_command", "sh", "terminal", "tool_shell", "rb9_exec"],
    "run_python": ["run_python", "execute_python", "py_exec", "run_snippet",
                   "python_eval", "script_run", "tool_python", "px2_run"],
    "describe_symbol": ["describe_symbol", "lookup_symbol", "find_definition",
                        "symbol_info", "get_def", "code_lookup", "sy5_info"],
    "read_file": ["read_file", "open_file", "cat_file", "file_read",
                  "get_source", "load_file", "rf3_read"],
    "search_code": ["search_code", "grep_source", "find_in_code",
                    "code_grep", "scan_source", "sc8_find"],
    "list_files": ["list_files", "ls", "enumerate_files", "file_list",
                   "dir_listing", "lf2_list"],
    "repo_stats": ["repo_stats", "codebase_summary", "project_info",
                   "repo_overview", "rs4_stats"],
    # workspace tier (agent/workspace_tools.py)
    "write_file": ["write_file", "save_file", "create_file", "put_file",
                   "store_document", "wf1_write", "write_text"],
    "append_file": ["append_file", "add_to_file", "append_text", "extend_file",
                    "af2_append"],
}

# Paraphrases so the description is not a memorizable constant either.
DESCRIPTIONS = {
    "calc": ["Evaluate an arithmetic expression.",
             "Compute the result of a mathematical expression.",
             "Perform a calculation and return the number.",
             "Solve arithmetic: addition, subtraction, multiplication, division."],
    "search_memory": ["Look up a value stored in memory by its key.",
                      "Retrieve a saved internal value.",
                      "Read a previously stored key/value pair.",
                      "Fetch a remembered value by name."],
    "web_search": ["Search for a short factual answer.",
                   "Look up a fact that is not known internally.",
                   "Query an external knowledge source for a fact.",
                   "Find factual information about a topic."],
    "now": ["Return the current date and time.",
            "Get the present timestamp.",
            "Read the system clock.",
            "Report what time it is now."],
    "run_bash": ["Run a shell command in an isolated sandbox.",
                 "Execute a shell command and return its output.",
                 "Run a command line in a sandboxed environment."],
    "run_python": ["Run a Python snippet in an isolated sandbox.",
                   "Execute Python code and return its output.",
                   "Evaluate a Python script in a sandbox."],
    "describe_symbol": ["Look up a class or function definition by name.",
                        "Return the definition and docstring of a symbol.",
                        "Find where a symbol is defined."],
    "read_file": ["Read lines from a source file.",
                  "Return the contents of part of a file.",
                  "Open a file and read a range of lines."],
    "search_code": ["Search source code for a string.",
                    "Find occurrences of text across the codebase.",
                    "Grep the source for a literal match."],
    "list_files": ["List source files, optionally filtered.",
                   "Enumerate files in the project.",
                   "Show files whose path matches a pattern."],
    "repo_stats": ["Summarize the repository structure and size.",
                   "Report file and line counts per directory.",
                   "Give an overview of the codebase."],
    "write_file": ["Create or overwrite a text file in the workspace.",
                   "Save text to a file.",
                   "Write the given content to a file, replacing it."],
    "append_file": ["Append text to the end of a workspace file.",
                    "Add more text to an existing file.",
                    "Extend a file with additional content."],
}

PARAM_SCHEMAS = {
    "calc": {"expr": "Arithmetic expression to evaluate."},
    "search_memory": {"key": "The key to look up."},
    "web_search": {"query": "A short search query."},
    "now": {},
    "run_bash": {"command": "Shell command to run."},
    "run_python": {"code": "Python source to execute."},
    "describe_symbol": {"name": "Class or function name."},
    "read_file": {"path": "Repo-relative file path.",
                  "start": "First line number.", "lines": "How many lines."},
    "search_code": {"query": "Literal text to find."},
    "list_files": {"pattern": "Substring filter."},
    "repo_stats": {},
    "write_file": {"path": "Relative file path.", "content": "Full file contents."},
    "append_file": {"path": "Relative file path.", "content": "Text to append."},
}

# Surface names for each canonical parameter. Training never varied these
# before, so given a client tool add_numbers(a, b) the model emitted
# {"expr": "12 + 8"} - the right tool, an argument shape it had memorized.
# Randomizing them makes the parameter list something the model has to read
# from the schema, exactly as it already must for the tool name.
PARAM_NAME_POOLS = {
    "expr": ["expr", "expression", "input", "formula", "math", "calculation",
             "text", "q", "arithmetic", "e"],
    "key": ["key", "name", "id", "field", "memory_key", "k", "entry", "slot"],
    "query": ["query", "q", "search", "question", "text", "terms", "lookup",
              "topic"],
    "command": ["command", "cmd", "shell", "script", "line"],
    "code": ["code", "source", "snippet", "python", "program", "src"],
    "name": ["name", "symbol", "identifier", "target", "definition"],
    "path": ["path", "file", "filename", "filepath", "location"],
    "start": ["start", "offset", "from_line", "begin", "first"],
    "lines": ["lines", "count", "n", "limit", "how_many"],
    "pattern": ["pattern", "glob", "filter", "match", "substring"],
    "content": ["content", "text", "body", "data", "contents"],
}

PARAM_DESCRIPTIONS = {
    "expr": ["Arithmetic expression to evaluate.", "The math to compute.",
             "An expression such as 12 + 8.", "Formula to calculate."],
    "key": ["The key to look up.", "Name of the stored value.",
            "Which memory entry to read."],
    "query": ["A short search query.", "What to search for.",
              "The question to look up."],
    "command": ["Shell command to run.", "The command line to execute."],
    "code": ["Python source to execute.", "The snippet to run."],
    "name": ["Class or function name.", "The symbol to describe."],
    "path": ["Repo-relative file path.", "Which file to open."],
    "start": ["First line number.", "Line to start from."],
    "lines": ["How many lines.", "Number of lines to return."],
    "pattern": ["Substring filter.", "Only paths containing this."],
    "content": ["The text to write.", "File contents.", "Body of the file."],
}

REQUIRED = {
    "calc": ["expr"], "search_memory": ["key"], "web_search": ["query"],
    "now": [], "run_bash": ["command"], "run_python": ["code"],
    "describe_symbol": ["name"], "read_file": ["path"],
    "search_code": ["query"], "list_files": ["pattern"], "repo_stats": [],
    "write_file": ["path", "content"], "append_file": ["path", "content"],
}


class ToolSchemaSampler:
    """Produces a randomized schema and the canonical -> surface name map."""

    def __init__(self, rng: random.Random, distractor_prob: float = 0.6,
                 max_distractors: int = 4, param_rename_prob: float = 0.7):
        self.rng = rng
        self.distractor_prob = distractor_prob
        self.max_distractors = max_distractors
        self.param_rename_prob = param_rename_prob
        self.last_param_maps = {}

    def sample(self, used_tools) -> tuple:
        """Return (mapping, system_block) for this trace.

        ``used_tools`` are the canonical tools the trace actually calls.
        Distractors are included so the model must SELECT, not just copy the
        only option available. Parameter names are randomized too (see
        ``PARAM_NAME_POOLS``); the per-tool ``{canonical: surface}`` maps are
        left in ``self.last_param_maps`` for ``randomize_trace`` to apply to
        the assistant's arguments.
        """
        used = [t for t in used_tools if t in NAME_POOLS]
        pool = [t for t in NAME_POOLS if t not in set(used)]
        extras = []
        # With no tools used at all every entry is a distractor, and there
        # must be at least one or the schema block would be empty.
        if pool and (not used or self.rng.random() < self.distractor_prob):
            lo = 1 if used else 2
            k = self.rng.randint(lo, min(max(self.max_distractors, lo), len(pool)))
            extras = self.rng.sample(pool, k)

        rename_params = self.rng.random() < self.param_rename_prob
        mapping, entries, param_maps = {}, [], {}
        for canonical in used + extras:
            surface = self.rng.choice(NAME_POOLS[canonical])
            mapping[canonical] = surface
            pmap, props, taken = {}, {}, set()
            for arg, desc in PARAM_SCHEMAS[canonical].items():
                choices = [c for c in PARAM_NAME_POOLS.get(arg, [arg]) if c not in taken]
                new = self.rng.choice(choices) if rename_params and choices else arg
                taken.add(new)
                pmap[arg] = new
                props[new] = {
                    "type": "integer" if arg in ("start", "lines") else "string",
                    "description": (self.rng.choice(PARAM_DESCRIPTIONS[arg])
                                    if rename_params and arg in PARAM_DESCRIPTIONS
                                    else desc)}
            param_maps[canonical] = pmap
            entries.append({
                "name": surface,
                "description": self.rng.choice(DESCRIPTIONS[canonical]),
                "parameters": {"type": "object", "properties": props,
                               "required": [pmap[a] for a in REQUIRED[canonical]]},
            })
        # Shuffle so position carries no information either.
        self.rng.shuffle(entries)
        self.last_param_maps = param_maps
        return mapping, _schema_block(entries)


def _schema_block(entries: list) -> str:
    """The one place the schema block is formatted.

    Training, the live Toolbox and client-supplied Ollama tools all go through
    here, so the minified JSON the model was trained on and the JSON it is
    served cannot drift apart.
    """
    return "<system>" + json.dumps({"tools": entries},
                                   separators=(",", ":")) + "</system>"


def schema_block_from_toolbox(toolbox) -> str:
    """Build the system block for a live Toolbox, using its real names."""
    entries = []
    for name, spec in toolbox.tools.items():
        if name == "finish":
            continue
        entries.append({"name": name,
                        "description": spec["description"],
                        "parameters": spec["parameters"]})
    return _schema_block(entries)


def schema_block_from_tools(tools: list) -> str:
    """Build the system block from Ollama/OpenAI-shaped client tools.

    Each entry is ``{"type": "function", "function": {"name", "description",
    "parameters"}}``. Entries without a name are skipped; a missing description
    becomes "" and missing parameters become an empty object schema, so a
    sloppy client still gets a block the model can read. Returns "" when there
    are no usable tools - a trace with no tool use carries no schema block, so
    emitting an empty one would be a form the model has never seen.
    """
    entries = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        entries.append({"name": name,
                        "description": str(fn.get("description") or ""),
                        "parameters": params})
    return _schema_block(entries) if entries else ""


def schema_block_from_entries(tools: list) -> str:
    """The system block for tools that came from outside this repo.

    An imported dataset (``scripts/import_hf_dataset.py``) brings its own
    function schemas - xlam alone has 3,605 distinct names - which are not in
    any Toolbox and must not be renamed: the whole value of that data is that
    the names are ones we never invented. This formats them through the same
    ``_schema_block`` the live server uses, so an imported trace and a served
    request cannot drift apart.
    """
    entries = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        entries.append({"name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters")
                        or {"type": "object", "properties": {}}})
    return _schema_block(entries) if entries else ""


def randomize_trace(text: str, rng: random.Random,
                    sampler: "ToolSchemaSampler" = None,
                    force_schema: bool = False) -> str:
    """Rewrite a trace to use randomized tool names, prefixed by its schema.

    Applied as a post-processing pass so every generator benefits without each
    one needing to know about schemas. Both the assistant's ``"name":"calc"``
    and the observation tag ``<tool name=calc>`` are rewritten, so the
    canonical name survives nowhere in the text and cannot be memorized.
    """
    sampler = sampler or ToolSchemaSampler(rng)
    used = []
    for canonical in NAME_POOLS:
        if f'"name":"{canonical}"' in text or f"<tool name={canonical}>" in text:
            used.append(canonical)
    if not used:
        # A trace that calls nothing still needs the "tools were offered and
        # the right move was to answer anyway" case, which is only learnable
        # when a schema is present and unused.
        if not force_schema:
            return text
        _, block = sampler.sample([])
        return block + "\n" + text if block else text
    mapping, block = sampler.sample(used)

    # Longest first: "search_code" must not be rewritten by the "search_memory"
    # rule, and no canonical name may be partially matched by another.
    for canonical in sorted(mapping, key=len, reverse=True):
        surface = mapping[canonical]
        text = text.replace(f'"name":"{canonical}"', f'"name":"{surface}"')
        text = text.replace(f"<tool name={canonical}>", f"<tool name={surface}>")
    text = _rename_arguments(text, mapping, sampler.last_param_maps)
    return block + "\n" + text


_ASSISTANT_JSON = re.compile(r"(<assistant>)(\{.*?\})(</assistant>)", re.S)


def _rename_arguments(text: str, name_map: dict, param_maps: dict) -> str:
    """Rewrite the argument keys of every tool call to the sampled surface
    names. The schema block already uses them; the calls must match."""
    if not param_maps or all(v == {k2: k2 for k2 in v} for v in param_maps.values()):
        return text
    surface_to_canonical = {v: k for k, v in name_map.items()}

    def fix(m):
        try:
            obj = json.loads(m.group(2))
        except json.JSONDecodeError:
            return m.group(0)
        call = obj.get("tool_call")
        if isinstance(call, dict):
            canonical = surface_to_canonical.get(call.get("name"))
            pmap = param_maps.get(canonical)
            args = call.get("arguments")
            if pmap and isinstance(args, dict):
                call["arguments"] = {pmap.get(k, k): v for k, v in args.items()}
        return m.group(1) + json.dumps(obj, separators=(",", ":")) + m.group(3)

    return _ASSISTANT_JSON.sub(fix, text)
