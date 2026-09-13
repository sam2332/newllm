"""Compose instruct and chat traces from the Ollama-generated diversity pool.

Division of labour:
  * the teacher model supplies LANGUAGE (question phrasings, reasoning
    sentences, factual key/value pairs)
  * the real ``Toolbox`` supplies every TOOL RESULT

The teacher is never trusted to compute or to state what a tool returned, so a
hallucinating teacher cannot poison the data - it can only make it read less
naturally. Every emitted trace is verified against the Toolbox before it is
kept, and traces that fail verification are dropped and counted.
"""

import collections
import json
import random

from agent.tools import Toolbox
from agent.repo_tools import attach_repo_tools, describe_symbol, REPO_TOOL_SPECS
from agent.sandbox_tools import SANDBOX_TOOL_SPECS
from agent.chat_dataset import serialize_chat, SYSTEM_PROMPTS

EOT = "\x03"


def load_pool(path: str, sandbox_path: str = "data/sandbox_pool.json") -> dict:
    with open(path) as fh:
        pool = json.load(fh)
    pool.setdefault("thoughts", {})
    pool.setdefault("paraphrases", {})
    pool.setdefault("facts", {})
    pool.setdefault("sandbox", [])
    # Sandbox entries are precomputed and already container-verified.
    if not pool["sandbox"]:
        try:
            with open(sandbox_path) as fh:
                pool["sandbox"] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pool["sandbox"] = []
    return pool


class TraceComposer:
    """Builds Toolbox-verified traces using teacher-generated language."""

    # Symbols and files the self-inspection traces ask about. Kept small and
    # concrete so the observations fit a 768-token context.
    REPO_SYMBOLS = [
        "AssistantGrammar", "MultiHeadAttention", "TransformerBlock",
        "RMSNorm", "AgentTokenizer", "Toolbox", "Trainer", "TextEncoder",
        "FeedForward", "RoPECache", "AgentDataset", "TraceComposer",
    ]
    REPO_FILES = [
        "model/attention.py", "model/rope.py", "model/norm.py",
        "model/transformer_block.py", "agent/tokenizer.py", "agent/tools.py",
        "agent/constrained.py", "training/trainer.py",
    ]
    REPO_TERMS = [
        "qk_norm", "arch_version", "n_kv_heads", "apply_rope", "z_loss",
        "scaled_dot_product_attention", "SwiGLU", "tie_weights",
    ]

    def __init__(self, pool: dict, seed: int = 42, memory: dict = None):
        self.pool = pool
        self.rng = random.Random(seed)
        self.sandbox_pool = pool.get("sandbox") or []
        self.facts = {k: v for k, v in pool["facts"].items()
                      if k and v and len(k) < 60}
        self.memory = memory or dict(Toolbox.DEFAULT_MEMORY)
        self.toolbox = attach_repo_tools(
            Toolbox(memory=self.memory, web_kb=self.facts or None))
        # Registered so tool names validate; execution is never triggered
        # during dataset generation (outputs are precomputed and verified).
        self.toolbox.tools.update(SANDBOX_TOOL_SPECS)
        self.dropped = 0
        # Repo-inspection tools walk the filesystem and their output is
        # deterministic, so identical calls are memoized during generation.
        self._repo_cache = {}
        # Facts whose value starts with a number can feed arithmetic chains.
        self.numeric_facts = {
            k: v for k, v in self.facts.items()
            if self._leading_number(v) is not None
        }

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _leading_number(value: str):
        token = str(value).split()[0].replace(",", "") if str(value).split() else ""
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def _num(value) -> str:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(round(f, 4))

    def thought(self, situation: str, fallback: str) -> str:
        options = self.pool["thoughts"].get(situation)
        return self.rng.choice(options) if options else fallback

    def phrase(self, intent: str, fallback: str, **subs) -> str:
        options = self.pool["paraphrases"].get(intent)
        template = self.rng.choice(options) if options else fallback
        for key, value in subs.items():
            template = template.replace("{" + key + "}", str(value))
        return template

    def _aj(self, thought, response=None, tool_call=None) -> str:
        data = {"thought": thought}
        if response is not None:
            data["response"] = response
        if tool_call is not None:
            data["tool_call"] = tool_call
        return json.dumps(data, separators=(",", ":"))

    _CACHEABLE = frozenset(REPO_TOOL_SPECS)

    def _call(self, name: str, args: dict) -> str:
        """Run the real tool. This is the only source of observations."""
        if name in self._CACHEABLE:
            key = (name, json.dumps(args, sort_keys=True))
            if key not in self._repo_cache:
                self._repo_cache[key] = self.toolbox.run_json(
                    {"tool": name, "args": args})
            return self._repo_cache[key]
        return self.toolbox.run_json({"tool": name, "args": args})

    def _turn(self, question, tool, args, think_call, think_done,
              final=None):
        """One user turn -> tool call -> verified observation -> answer."""
        observation = self._call(tool, args)
        answer = observation if final is None else final
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": self._aj(
                think_call, tool_call={"name": tool, "arguments": args})},
            {"role": "tool", "name": tool, "content": observation},
            {"role": "assistant", "content": self._aj(think_done,
                                                      response=answer)},
        ], observation

    # ----------------------------------------------------------- patterns
    def p_math(self):
        a, b = self.rng.randint(1, 500), self.rng.randint(1, 500)
        op = self.rng.choice(["+", "-", "*", "/"])
        expr = f"{a} {op} {b}"
        q = self.phrase("math", f"What is {expr}?", EXPR=expr)
        msgs, _ = self._turn(
            q, "calc", {"expr": expr},
            self.thought("calc_call", "I need to use the calculator."),
            self.thought("calc_done", "The calculation is complete."))
        return msgs

    def p_memory(self):
        key = self.rng.choice(list(self.memory.keys()))
        q = self.phrase("memory", f"Look up the {key}.", KEY=key)
        msgs, _ = self._turn(
            q, "search_memory", {"key": key},
            self.thought("memory_call", f"I should search memory for '{key}'."),
            self.thought("memory_done", "I found the stored value."))
        return msgs

    def p_web(self):
        if not self.facts:
            return self.p_math()
        key = self.rng.choice(list(self.facts.keys()))
        q = self.phrase("web", f"What is the {key}?", TOPIC=key)
        msgs, _ = self._turn(
            q, "web_search", {"query": key},
            self.thought("web_call", "I should look this up online."),
            self.thought("web_done", "I found the answer."))
        return msgs

    def p_time(self):
        q = self.phrase("time", "What is the current date and time?", NOW="")
        msgs, _ = self._turn(
            q, "now", {},
            self.thought("time_call", "I need the current time."),
            self.thought("time_done", "Here is the current timestamp."))
        return msgs

    def p_web_then_math(self):
        if not self.numeric_facts:
            return self.p_math()
        key = self.rng.choice(list(self.numeric_facts.keys()))
        value = self.facts[key]
        base = self._leading_number(value)
        k = self.rng.randint(2, 50)
        op, word = self.rng.choice([("+", "add"), ("-", "subtract"),
                                    ("*", "multiply it by")])
        q = f"Look up the {key} and {word} {k}."
        observation = self._call("web_search", {"query": key})
        obs_num = self._leading_number(observation)
        if obs_num is None:
            self.dropped += 1
            return None
        expr = f"{self._num(obs_num)} {op} {k}"
        result = self._call("calc", {"expr": expr})
        return [
            {"role": "user", "content": q},
            {"role": "assistant", "content": self._aj(
                self.thought("web_call", "I need the web fact first."),
                tool_call={"name": "web_search", "arguments": {"query": key}})},
            {"role": "tool", "name": "web_search", "content": observation},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_first", "Now I combine it with arithmetic."),
                tool_call={"name": "calc", "arguments": {"expr": expr}})},
            {"role": "tool", "name": "calc", "content": result},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_last", "The final answer is computed."),
                response=result)},
        ]

    def p_memory_then_math(self):
        numeric_keys = [k for k, v in self.memory.items()
                        if self._leading_number(v) is not None]
        if not numeric_keys:
            return self.p_math()
        key = self.rng.choice(numeric_keys)
        observation = self._call("search_memory", {"key": key})
        base = self._leading_number(observation)
        if base is None:
            self.dropped += 1
            return None
        k = self.rng.randint(2, 40)
        op, word = self.rng.choice([("+", "add"), ("*", "multiply by")])
        q = f"Look up the {key} and {word} {k}."
        expr = f"{self._num(base)} {op} {k}"
        result = self._call("calc", {"expr": expr})
        return [
            {"role": "user", "content": q},
            {"role": "assistant", "content": self._aj(
                self.thought("memory_call", "First retrieve the stored value."),
                tool_call={"name": "search_memory", "arguments": {"key": key}})},
            {"role": "tool", "name": "search_memory", "content": observation},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_first", "Now apply the arithmetic."),
                tool_call={"name": "calc", "arguments": {"expr": expr}})},
            {"role": "tool", "name": "calc", "content": result},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_last", "The final answer is computed."),
                response=result)},
        ]

    def p_two_stage_math(self):
        a, b = self.rng.randint(2, 99), self.rng.randint(2, 99)
        c = self.rng.randint(2, 20)
        e1 = f"{a} * {b}"
        r1 = self._call("calc", {"expr": e1})
        e2 = f"{r1} + {c}"
        r2 = self._call("calc", {"expr": e2})
        if r1.startswith("ERROR") or r2.startswith("ERROR"):
            self.dropped += 1
            return None
        q = f"Multiply {a} and {b}, then add {c}."
        return [
            {"role": "user", "content": q},
            {"role": "assistant", "content": self._aj(
                self.thought("calc_call", "First the multiplication."),
                tool_call={"name": "calc", "arguments": {"expr": e1}})},
            {"role": "tool", "name": "calc", "content": r1},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_first", "Now add the second number."),
                tool_call={"name": "calc", "arguments": {"expr": e2}})},
            {"role": "tool", "name": "calc", "content": r2},
            {"role": "assistant", "content": self._aj(
                self.thought("chain_last", "The final answer is computed."),
                response=r2)},
        ]

    # -------------------------------------------------------- chat (multi-turn)
    def p_chat_math_coref(self):
        a, b = self.rng.randint(2, 200), self.rng.randint(2, 200)
        e1 = f"{a} + {b}"
        r1 = self._call("calc", {"expr": e1})
        msgs, _ = self._turn(
            self.phrase("math", f"What is {e1}?", EXPR=e1), "calc",
            {"expr": e1}, self.thought("calc_call", "I need the calculator."),
            self.thought("calc_done", "Done."))
        k = self.rng.randint(2, 12)
        word, op = self.rng.choice([("Multiply that by", "*"),
                                    ("Add", "+"), ("Subtract", "-")])
        follow = (f"Add {k} to that." if word == "Add" else
                  f"Subtract {k} from that." if word == "Subtract" else
                  f"Multiply that by {k}.")
        e2 = f"{r1} {op} {k}"
        r2 = self._call("calc", {"expr": e2})
        if r2.startswith("ERROR"):
            self.dropped += 1
            return None
        msgs += [
            {"role": "user", "content": follow},
            {"role": "assistant", "content": self._aj(
                self.thought("coref",
                             f"'that' refers to the previous result {r1}."),
                tool_call={"name": "calc", "arguments": {"expr": e2}})},
            {"role": "tool", "name": "calc", "content": r2},
            {"role": "assistant", "content": self._aj(
                self.thought("calc_done", "Done."), response=r2)},
        ]
        return msgs

    def p_chat_topic_pair(self):
        if len(self.facts) < 2:
            return self.p_chat_math_coref()
        k1, k2 = self.rng.sample(list(self.facts.keys()), 2)
        msgs, _ = self._turn(
            self.phrase("web", f"What is the {k1}?", TOPIC=k1), "web_search",
            {"query": k1}, self.thought("web_call", "I should look this up."),
            self.thought("web_done", "Found it."))
        follow = self.rng.choice([f"And the {k2}?", f"What about the {k2}?",
                                  f"Now the {k2}."])
        more, _ = self._turn(
            follow, "web_search", {"query": k2},
            self.thought("web_call",
                         f"The user has moved on to the {k2}."),
            self.thought("web_done", "Found it."))
        return msgs + more

    def p_chat_back_reference(self):
        a, b = self.rng.randint(10, 99), self.rng.randint(10, 99)
        e1 = f"{a} + {b}"
        r1 = self._call("calc", {"expr": e1})
        msgs, _ = self._turn(
            f"What is {e1}?", "calc", {"expr": e1},
            self.thought("calc_call", "I need the calculator."),
            self.thought("calc_done", "Done."))
        key = self.rng.choice(list(self.memory.keys()))
        more, _ = self._turn(
            f"Look up the {key}.", "search_memory", {"key": key},
            self.thought("memory_call", f"Searching memory for '{key}'."),
            self.thought("memory_done", "Found it."))
        msgs += more
        msgs += [
            {"role": "user", "content": self.rng.choice([
                "What was the result of my first question?",
                "Remind me what the first answer was.",
                "What number did we get at the start?"])},
            {"role": "assistant", "content": self._aj(
                self.thought("no_tool",
                             f"The first result {r1} is already in the "
                             f"conversation, so no tool is needed."),
                response=r1)},
        ]
        return msgs


    # ------------------------------------------------- three-tool chains
    def _chain(self, question, stages, think_names):
        """Build an N-stage trace. Every observation comes from the Toolbox."""
        msgs = [{"role": "user", "content": question}]
        for i, (tool, args) in enumerate(stages):
            observation = self._call(tool, args)
            if observation.startswith("ERROR") or observation == "no results found":
                self.dropped += 1
                return None
            last = i == len(stages) - 1
            msgs.append({"role": "assistant", "content": self._aj(
                self.thought(think_names[min(i, len(think_names) - 1)],
                             "Next step."),
                tool_call={"name": tool, "arguments": args})})
            msgs.append({"role": "tool", "name": tool, "content": observation})
            if last:
                msgs.append({"role": "assistant", "content": self._aj(
                    self.thought("chain_last", "The final answer is computed."),
                    response=observation)})
        return msgs

    def p_web_two_stage(self):
        """web_search -> calc -> calc (three tool calls)."""
        if not self.numeric_facts:
            return self.p_two_stage_math()
        key = self.rng.choice(list(self.numeric_facts.keys()))
        obs = self._call("web_search", {"query": key})
        base = self._leading_number(obs)
        if base is None:
            self.dropped += 1
            return None
        a, b = self.rng.randint(2, 40), self.rng.randint(2, 12)
        e1 = f"{self._num(base)} + {a}"
        r1 = self._call("calc", {"expr": e1})
        if r1.startswith("ERROR"):
            self.dropped += 1
            return None
        e2 = f"{r1} * {b}"
        q = f"Look up the {key}, add {a}, then multiply by {b}."
        return self._chain(q, [("web_search", {"query": key}),
                               ("calc", {"expr": e1}), ("calc", {"expr": e2})],
                           ["web_call", "chain_first", "chain_first"])

    def p_cross_source(self):
        """search_memory -> web_search -> calc (three tools, two sources)."""
        numeric_keys = [k for k, v in self.memory.items()
                        if self._leading_number(v) is not None]
        if not numeric_keys or not self.numeric_facts:
            return self.p_two_stage_math()
        mkey = self.rng.choice(numeric_keys)
        wkey = self.rng.choice(list(self.numeric_facts.keys()))
        m_obs = self._call("search_memory", {"key": mkey})
        w_obs = self._call("web_search", {"query": wkey})
        m_num, w_num = self._leading_number(m_obs), self._leading_number(w_obs)
        if m_num is None or w_num is None:
            self.dropped += 1
            return None
        op = self.rng.choice(["+", "*"])
        expr = f"{self._num(m_num)} {op} {self._num(w_num)}"
        word = "add it to" if op == "+" else "multiply it by"
        q = f"Look up the stored {mkey}, then {word} the {wkey}."
        return self._chain(q, [("search_memory", {"key": mkey}),
                               ("web_search", {"query": wkey}),
                               ("calc", {"expr": expr})],
                           ["memory_call", "chain_first", "chain_first"])

    def p_memory_two_stage(self):
        """search_memory -> calc -> calc (three tool calls)."""
        numeric_keys = [k for k, v in self.memory.items()
                        if self._leading_number(v) is not None]
        if not numeric_keys:
            return self.p_two_stage_math()
        key = self.rng.choice(numeric_keys)
        obs = self._call("search_memory", {"key": key})
        base = self._leading_number(obs)
        if base is None:
            self.dropped += 1
            return None
        a, b = self.rng.randint(2, 30), self.rng.randint(2, 10)
        e1 = f"{self._num(base)} + {a}"
        r1 = self._call("calc", {"expr": e1})
        if r1.startswith("ERROR"):
            self.dropped += 1
            return None
        e2 = f"{r1} * {b}"
        q = (f"Multiply {a} and {b}, then add the stored {key}."
             if self.rng.random() < 0.5
             else f"Look up the {key}, add {a}, then multiply by {b}.")
        return self._chain(q, [("search_memory", {"key": key}),
                               ("calc", {"expr": e1}), ("calc", {"expr": e2})],
                           ["memory_call", "chain_first", "chain_first"])

    # ------------------------------------------------- self-inspection
    def _repo_turn(self, question, tool, args, think_call, think_done,
                   max_obs=260, summarize=None):
        """Repo tools can return long text; keep both halves inside budget.

        The final response must NOT echo the whole observation. Doing that
        doubles the payload and pushed ``search_code`` and ``read_file`` traces
        past the context limit, so they were dropped outright and the model
        never learned those two tools. ``summarize`` produces a short answer
        that refers to the tool output instead of repeating it.
        """
        observation = self._call(tool, args)
        if observation.startswith("ERROR") or observation in (
                "no matches", "no matching files"):
            self.dropped += 1
            return None
        observation = observation.strip()
        if len(observation) > max_obs:
            observation = observation[:max_obs].rsplit("\n", 1)[0] + "\n..."
        answer = summarize(observation) if summarize else observation
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": self._aj(
                think_call, tool_call={"name": tool, "arguments": args})},
            {"role": "tool", "name": tool, "content": observation},
            {"role": "assistant", "content": self._aj(think_done,
                                                      response=answer)},
        ]

    @staticmethod
    def _summarize_hits(observation: str) -> str:
        """Short answer for a grep-style result: count plus first location."""
        lines = [l for l in observation.splitlines() if ":" in l]
        if not lines:
            return observation[:120]
        first = lines[0].split(":")
        where = f"{first[0]}:{first[1]}" if len(first) > 1 else lines[0]
        n = len(lines)
        return (f"Found {n} match{'es' if n != 1 else ''}, "
                f"the first at {where}.")

    @staticmethod
    def _summarize_files(observation: str) -> str:
        files = [l for l in observation.splitlines() if l.strip()]
        if not files:
            return "No matching files."
        head = ", ".join(files[:3])
        extra = f" and {len(files) - 3} more" if len(files) > 3 else ""
        return f"{len(files)} file(s): {head}{extra}."

    @staticmethod
    def _summarize_read(observation: str) -> str:
        lines = [l for l in observation.splitlines() if l.strip()]
        first = lines[0].strip()[:90] if lines else ""
        return (f"That section has {len(lines)} non-empty line(s), "
                f"starting with: {first}")

    def p_repo_symbol(self):
        name = self.rng.choice(self.REPO_SYMBOLS)
        q = self.rng.choice([
            f"What does {name} do?", f"Describe the {name} class.",
            f"Look up {name} in your own source code.",
            f"Where is {name} defined?", f"Explain {name}.",
        ])
        return self._repo_turn(
            q, "describe_symbol", {"name": name},
            self.thought("repo_call",
                         f"This asks about my own code, so I should look up "
                         f"the symbol {name}."),
            self.thought("repo_done", "Here is the definition from my source."))

    def p_repo_search(self):
        term = self.rng.choice(self.REPO_TERMS)
        q = self.rng.choice([
            f"Where is {term} used in your code?",
            f"Search your source for {term}.",
            f"Which files mention {term}?",
        ])
        return self._repo_turn(
            q, "search_code", {"query": term},
            self.thought("repo_call",
                         f"I should grep my own source for '{term}'."),
            self.thought("repo_done", "These are the matching locations."),
            summarize=self._summarize_hits)

    def p_repo_read(self):
        path = self.rng.choice(self.REPO_FILES)
        start = self.rng.choice([1, 1, 1, 10, 20])
        lines = self.rng.choice([12, 15, 20])
        q = self.rng.choice([
            f"Show me lines {start} to {start + lines - 1} of {path}.",
            f"Read {path} starting at line {start}.",
            f"What is at the top of {path}?" if start == 1
            else f"Show me {path} from line {start}.",
        ])
        return self._repo_turn(
            q, "read_file", {"path": path, "start": start, "lines": lines},
            self.thought("repo_call",
                         f"I should read my own source file {path}."),
            self.thought("repo_done", "Here is that section of the file."),
            summarize=self._summarize_read)

    def p_repo_stats(self):
        q = self.rng.choice([
            "How big is your own codebase?",
            "Summarize your source code structure.",
            "How many files and lines do you consist of?",
            "What directories make up your code?",
        ])
        return self._repo_turn(
            q, "repo_stats", {},
            self.thought("repo_call",
                         "This asks about my own repository, so I should "
                         "summarize it."),
            self.thought("repo_done", "Here is the structure of my code."))

    def p_repo_list(self):
        pattern = self.rng.choice(["attention", "model/", "agent/", "train",
                                   "test", "dataset", "norm"])
        q = self.rng.choice([
            f"Which of your files relate to {pattern}?",
            f"List your source files matching {pattern}.",
            f"Find files named like {pattern}.",
        ])
        return self._repo_turn(
            q, "list_files", {"pattern": pattern},
            self.thought("repo_call",
                         f"I should list my own files matching '{pattern}'."),
            self.thought("repo_done", "These are the matching files."),
            summarize=self._summarize_files)


    # ------------------------------------------------------- sandbox
    def p_sandbox(self):
        """Trace a question answered by running code in the Docker sandbox.

        Outputs come from ``scripts/gen_sandbox_pool.py``, which executed each
        command in a real container and discarded anything that failed. No
        container is started here, so dataset generation stays fast.
        """
        if not self.sandbox_pool:
            return self.p_math()
        item = self.rng.choice(self.sandbox_pool)
        output = item["output"]
        if len(output) > 400:
            self.dropped += 1
            return None
        verb = ("write a shell command" if item["tool"] == "run_bash"
                else "write a Python snippet")
        return [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": self._aj(
                self.thought("sandbox_call",
                             f"I should {verb} and run it in the sandbox."),
                tool_call={"name": item["tool"], "arguments": item["args"]})},
            {"role": "tool", "name": item["tool"], "content": output},
            {"role": "assistant", "content": self._aj(
                self.thought("sandbox_done",
                             "The sandbox returned the result."),
                response=output)},
        ]


# Weights matter more than they look. An earlier mix with 33% multi-step and
# NO three-tool chains scored 52.9% on the battery with 0% on 3-hop questions,
# against 88.2%/100% for a mix with 57% multi-step and 32% three-tool traces.
# The model simply cannot learn a chain depth it has never seen, so keep
# three-tool patterns well represented and the new tool families modest.
INSTRUCT_PATTERNS = [
    ("p_math", 11), ("p_memory", 7), ("p_web", 11), ("p_time", 5),
    # two-tool chains
    ("p_web_then_math", 11), ("p_memory_then_math", 9),
    ("p_two_stage_math", 9),
    # three-tool chains - the capability that collapsed when these were absent
    ("p_web_two_stage", 11), ("p_cross_source", 10), ("p_memory_two_stage", 10),
    # self-inspection, kept small so it does not crowd out chaining
    ("p_repo_symbol", 2), ("p_repo_search", 2), ("p_repo_read", 1),
    ("p_repo_stats", 1), ("p_repo_list", 1),
    ("p_sandbox", 4),
]
CHAT_PATTERNS = [
    ("p_chat_math_coref", 34), ("p_chat_topic_pair", 30),
    ("p_chat_back_reference", 22), ("p_math", 7), ("p_web", 7),
]


def _serialize_instruct(messages) -> str:
    parts = []
    for m in messages:
        if m["role"] == "tool":
            parts.append(f"<tool name={m['name']}>{m['content']}</tool>")
        else:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
    return "\n".join(parts) + EOT


def generate(pool: dict, num_samples: int, mode: str = "instruct",
             max_len: int = 768, seed: int = 42,
             system_prob: float = 0.4) -> tuple:
    """Return (traces, stats). ``mode`` is 'instruct' or 'chat'."""
    comp = TraceComposer(pool, seed=seed)
    patterns = INSTRUCT_PATTERNS if mode == "instruct" else CHAT_PATTERNS
    names = [n for n, _ in patterns]
    weights = [w for _, w in patterns]
    traces, too_long = [], 0
    kept_by = collections.Counter()
    dropped_by = collections.Counter()
    while len(traces) < num_samples:
        name = comp.rng.choices(names, weights=weights, k=1)[0]
        messages = getattr(comp, name)()
        if not messages:
            continue
        if mode == "chat":
            system = (comp.rng.choice(SYSTEM_PROMPTS)
                      if comp.rng.random() < system_prob else None)
            text = serialize_chat(messages, system=system)
        else:
            text = _serialize_instruct(messages)
        if len(text) > max_len:
            too_long += 1
            dropped_by[name] += 1
            continue
        kept_by[name] += 1
        traces.append(text)

    # A pattern that is mostly or entirely dropped for length teaches the model
    # nothing, silently. This has now happened twice: once to the repo tools and
    # once to the three-tool chains, where losing the pattern took 3-hop
    # accuracy from 100% to 0%. Aggregate counts hid it both times, so report
    # per pattern and refuse to be quiet about it.
    starved = []
    for name in names:
        total = kept_by[name] + dropped_by[name]
        if total and dropped_by[name] / total > 0.20:
            starved.append((name, dropped_by[name] / total, kept_by[name]))
    if starved:
        print(f"WARNING: {len(starved)} pattern(s) losing >20% of traces to the "
              f"max_len={max_len} limit:")
        for name, rate, kept in sorted(starved, key=lambda x: -x[1]):
            print(f"    {name:22s} {rate:4.0%} dropped, only {kept} kept")
        print("    Raise --max-len or shorten these patterns; a starved "
              "pattern is a capability the model cannot learn.")

    return traces, {"dropped_invalid": comp.dropped, "too_long": too_long,
                    "facts": len(comp.facts),
                    "numeric_facts": len(comp.numeric_facts),
                    "kept_by_pattern": dict(kept_by),
                    "dropped_by_pattern": dict(dropped_by),
                    "starved_patterns": [n for n, _, _ in starved]}
