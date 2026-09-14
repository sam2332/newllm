# Archived notes

Superseded working notes, kept because they record *why* decisions were made.
**Do not treat anything here as current.** Current state lives in
[HANDOFF.md](../../HANDOFF.md); how to run things lives in
[scripts/run/README.md](../../scripts/run/README.md).

Archived 2026-09-14.

| file | what it is | why it is here |
|---|---|---|
| `progress.md` | running narrative of every experiment | states superseded conclusions as current, e.g. that dataset work should stop because the model "cannot retain exact tool arguments and multi-step arithmetic together" - reached against the defective `arch_version=1` |
| `mistakes.md` | bugs found and what caused them | still the best record of past defects; the durable lessons are folded into HANDOFF.md |
| `LEADERBOARD.md` | newest-first experiment results | scores are against the pre-schema protocol and the 17-case battery, so they are not comparable to current held-out numbers |
| `DREAMS.md` | speculative directions | never acted on |
| `CLEANUP.md` | one-off tidy-up checklist | done |
| `sidetrack_ideas.md` | parked ideas | unevaluated |

## Claims in here that are now known to be wrong

- **"`checkpoints_v2_M` is the best instruct checkpoint at 88.2%."** It predates
  tool schemas in context and scores 0% when tools are renamed, so it cannot
  serve a user's own tools.
- **"Constrained decoding does not help because the model already emits 100%
  valid JSON."** The premise stopped being true. The model emits *invalid tool
  names*; masking them is worth 24/100 vs 22/100 and does not fix deep chains.
- **"Validation loss reaches 0.0000 because the model memorizes 34 sentences."**
  True of the old hand-written generator, not of the current pool.
- **Scores quoted as percentages on the 17-case battery.** One case is 5.9
  points. Quote held-out traces instead (`scripts/eval_random.py`).
