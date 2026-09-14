"""Long-horizon continuity: does the model read a file back instead of guessing?

The 52-turn project skill in one measurable shape. With a small ``num_ctx``
so early turns genuinely scroll out of the context:

  1. the user asks for an outline to be saved (the model must write a file);
  2. several chapter-writing turns follow (each a write_file);
  3. the user asks a fact that was only ever in the outline.

Scored per case: ``wrote_outline`` (a write_file with the outline content),
``read_back`` (a read_file call before answering step 3) and ``correct``
(the fact appears in the reply). Runs the same client loop as serving.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_longhorizon.py checkpoints_X/agent_best.pt
"""

import argparse
import json
import random
import sys

sys.path.insert(0, "/home/lmeadows/llm")

from agent.chat import load_checkpoint, load_tokenizer_for
from agent.local_client import run_tool_loop
from agent.tools import Toolbox
from agent.virtual_workspace import VirtualWorkspace
from agent.workspace_tools import attach_workspace_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--stories", default="data/chapters.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--chapters", type=int, default=3)
    ap.add_argument("--num-ctx", type=int, default=6144)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint, device=args.device)
    tok = load_tokenizer_for(model)
    stories = [s for s in json.load(open(args.stories))
               if all(c.get("text") for c in s.get("chapters", []))]
    rng = random.Random(args.seed)
    totals = {"wrote_outline": 0, "read_back": 0, "correct": 0}
    for k in range(args.cases):
        story = rng.choice(stories)
        tb = attach_workspace_tools(Toolbox(), VirtualWorkspace())
        messages = [{"role": "system", "content": "You are a writing assistant. Keep drafts in files."}]
        outline = (f"Title: {story['title']}\nProtagonist: {story['protagonist']['name']} - "
                   f"{story['protagonist']['trait']}\nSetting: {story.get('setting', '')}\n"
                   + "\n".join(f"{i+1}. {c['title']}: {c['summary']}"
                               for i, c in enumerate(story["chapters"][:args.chapters])))
        messages.append({"role": "user", "content":
                         f"We're writing \"{story['title']}\". Save this outline to outline.md:\n{outline}"})
        out = run_tool_loop(model, tok, messages, tb, device=args.device, num_ctx=args.num_ctx)
        messages = out["messages"]
        wrote = any(n == "write_file" and story["protagonist"]["name"] in str(a.get("content", ""))
                    for n, a, _ in out["steps"])
        for i in range(args.chapters):
            messages.append({"role": "user", "content": f"Write chapter {i+1} and save it to ch{i+1:02d}.md."})
            out = run_tool_loop(model, tok, messages, tb, device=args.device, num_ctx=args.num_ctx)
            messages = out["messages"]
        messages.append({"role": "user", "content":
                         "Remind me: what is the protagonist's defining trait? Check the outline file."})
        out = run_tool_loop(model, tok, messages, tb, device=args.device, num_ctx=args.num_ctx)
        read_back = any(n == "read_file" for n, _, _ in out["steps"])
        correct = story["protagonist"]["trait"].lower() in (out["content"] or "").lower()
        totals["wrote_outline"] += wrote
        totals["read_back"] += read_back
        totals["correct"] += correct
        print(f"  case {k+1}: outline={'y' if wrote else 'n'} read_back={'y' if read_back else 'n'} "
              f"correct={'y' if correct else 'n'}  {(out['content'] or '')[:70]!r}", flush=True)
        if args.verbose:
            for n, a, o in out["steps"]:
                print(f"      {n}({json.dumps(a)[:60]}) -> {str(o)[:50]!r}")
    print(f"\nTOTAL wrote_outline {totals['wrote_outline']}/{args.cases}  "
          f"read_back {totals['read_back']}/{args.cases}  correct {totals['correct']}/{args.cases}")


if __name__ == "__main__":
    main()
