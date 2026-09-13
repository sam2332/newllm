"""Why do 4+ hop chains fail? Classify the failure instead of guessing.

Held-out accuracy is dominated by one bucket: traces needing 4+ tool calls.
This replays those traces and reports where the model departs from the
reference plan and what kind of departure it is, which is the difference
between "needs more data", "needs longer training" and "needs constrained
decoding".

The answer the first time it was run: 35 of 40 diverged at hop 0, so depth
itself was not the problem.
"""
import json, random, re, sys, collections
sys.path.insert(0, "/home/lmeadows/llm")
import torch
from torch.utils.data import random_split
from agent.agent_loop import run_agent
from agent.dataset_builder import PrebuiltDataset
from agent.tokenizer import DEFAULT_AGENT_TOKENIZER as TOK
from scripts.eval_agent import load
from scripts.eval_random import renamed_toolbox, split_trace, norm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CKPT = "checkpoints_schema_M/agent_best.pt"

data = torch.load("data/cache/instruct_1d39780d71d5d7f5.pt", weights_only=False)
ds = PrebuiltDataset(data)
val_n = max(1, int(len(ds) * 0.05))
_, val = random_split(ds, [len(ds)-val_n, val_n],
                      generator=torch.Generator().manual_seed(42))
rng = random.Random(7)
model = load(CKPT, device="cuda")

CALL = re.compile(r'"tool_call":\{"name":"([^"]+)","arguments":(\{.*?\})\}')

# collect indices whose reference needs 4+ calls
deep = []
for i in rng.sample(range(len(val)), min(4000, len(val))):
    toks, _ = val.dataset.samples[val.indices[i]]
    t = TOK.decode(toks)
    if t.count('"tool_call"') >= 4:
        deep.append((i, t))
    if len(deep) >= N:
        break

print(f"{len(deep)} deep traces; reference hop counts: "
      f"{collections.Counter(t.count(chr(34)+'tool_call'+chr(34)) for _, t in deep).most_common()}\n")

kinds = collections.Counter()
first_div = collections.Counter()
ref_hops_all, got_hops_all = [], []
for n, (i, text) in enumerate(deep, 1):
    system, q, expected = split_trace(text)
    tb = renamed_toolbox(system) if system else None
    if not tb or not q:
        kinds["unparseable reference"] += 1; continue
    ref = CALL.findall(text)
    r = run_agent(model, q, tb, device="cuda", greedy=True, tokenizer=TOK,
                  system=system, max_steps=24, max_new=256)
    got = [(s["tool"], s["arg"]) for s in r["steps"]]
    ref_hops_all.append(len(ref)); got_hops_all.append(len(got))

    # where does it first depart from the reference plan?
    div = None
    for k in range(max(len(ref), len(got))):
        if k >= len(got): div = ("stopped early", k); break
        if k >= len(ref): div = ("kept going", k); break
        if got[k][0] != ref[k][0]: div = ("wrong tool", k); break
        if json.loads(got[k][1]) != json.loads(ref[k][1]): div = ("wrong args", k); break
    if div is None:
        div = ("plan matched", len(ref))
    first_div[div[1]] += 1

    errs = sum(1 for s in r["steps"] if str(s["result"]).lower().startswith(("error", "unknown")))
    hit = bool(norm(expected)) and norm(expected) in norm(r.get("model_response") or "")
    if hit: kinds["CORRECT"] += 1
    elif not r["success"]: kinds[f"no final answer ({div[0]})"] += 1
    elif errs: kinds[f"tool error mid-chain ({div[0]})"] += 1
    else: kinds[f"finished, wrong ({div[0]})"] += 1

    if n <= 4:
        print(f"--- [{n}] ref {len(ref)} calls, model made {len(got)}, "
              f"first divergence at hop {div[1]} ({div[0]}), success={r['success']}")
        for k in range(min(6, max(len(ref), len(got)))):
            rr = f"{ref[k][0]}{ref[k][1]}" if k < len(ref) else "-"
            gg = f"{got[k][0]}{got[k][1]}" if k < len(got) else "-"
            mark = "  " if rr == gg else "<<"
            print(f"    {k}: ref {rr[:64]}\n       got {gg[:64]} {mark}")
        print()

print("failure classes:")
for k, v in kinds.most_common():
    print(f"  {v:3d}  {k}")
print("\nfirst divergence at hop:")
for k in sorted(first_div):
    print(f"  hop {k}: {first_div[k]}")
import statistics
print(f"\nreference hops mean {statistics.mean(ref_hops_all):.1f}, "
      f"model hops mean {statistics.mean(got_hops_all):.1f}")
