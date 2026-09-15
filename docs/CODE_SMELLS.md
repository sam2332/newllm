# Known smells and traps

Things that are wrong, or right but dangerous, with enough context to decide
whether to fix them. Most were found by being bitten. Ordered by how much time
they have cost or could cost.

Last updated 2026-09-15. Entries marked **fixed** are kept because the
reasoning is the useful part; the pattern usually recurs somewhere else.

---

## 1. Evals silently measure the harness

**Status: partly fixed, pattern still live.**

`scripts/eval_random.py`, `eval_agent.py`, `eval_renamed.py` and
`eval_chat.py` decode with the byte tokenizer and parse
`<assistant>{json}</assistant>`. Pointed at a BPE/ChatML checkpoint they do not
raise - they find nothing to parse and print `0/0 correct`, which is
indistinguishable from a broken model. This cost an afternoon, three times in
one day: once on the checkpoint's protocol, once on the *cache's* protocol
(the guard only covered the checkpoint), and once on a mispaired reference
answer.

Guards now exist (`require_legacy_protocol`, plus both evals refusing when
nothing parsed) but they are guards on a design that invites the mistake. The
real fix is one eval that reads the protocol from the checkpoint and dispatches
internally, with the legacy scripts deleted rather than fenced off.

**Fixed:** the hardcoded `--cache` default (one particular byte-tokenized
hash, unchecked) now defaults to the largest `data/cache/instruct_*.pt` and
says which it picked.

## 2. The root directory is a graveyard

**Status: fixed 2026-09-15.** `transformer.py`, `training_8k.py`,
`training_8k_big.py`, `train_sequence_moe.py`, `big_training_test.py` and
`live_training_test.py` moved to `archive/superseded/` after checking the
import graph - each had zero importers. `reasoning.py` stayed: it looked dead
by name but `training/trainer.py` and `test_modern_features.py` both import
`ReasoningTimeHelper` from it. Nineteen stray `agent_run_*.log` files remain,
gitignored, local clutter only.

Original note:

Twelve `.py` files sit at the repo root, most superseded by packages:

| file | state |
|---|---|
| `transformer.py` | superseded by `model/` - imported by **0** files |
| `reasoning.py` | imported by 0 files |
| `training_8k.py`, `training_8k_big.py`, `train_sequence_moe.py`, `big_training_test.py`, `live_training_test.py` | earlier training entry points, none referenced by any doc or script |
| `sampling.py` | **live** - imported by 5 modules |
| `results_logger.py` | **live** - imported by 1 |
| `smoke_test.py`, `test_modern_features.py`, `watch_agent.py` | live, referenced by AGENTS.md |

Nineteen `agent_run_*.log` files from August also sit there. They are
gitignored, so this is local clutter, not repo clutter.

Deleting the dead five plus `transformer.py` and `reasoning.py` is safe by
import graph, but they are the only record of how earlier runs were launched.
Move to `archive/` rather than delete.

## 3. `tok.eot` is a string, and there was no id accessor

**Status: fixed 2026-09-15.** Both tokenizers now expose `eot_id` alongside
`eot`, so callers stop improvising. `eot` is a string on *both* (`"<|im_end|>"`
and `"\x03"`) - the original note here was wrong about them disagreeing; the
real problem was that nothing offered the number.

`BPEAgentTokenizer.eot` returns `"<|im_end|>"`; code that treats it as an id
writes garbage. In `scripts/build_pretrain_shards.py` this would have poisoned
ten billion tokens - it resolves the id explicitly and asserts it encodes to a
single token. Every other caller is on its own. The tokenizers should agree on
what `eot` means, or expose `eot_id` alongside it.

## 4. `multiprocessing.Pool` deadlocks on a dead worker

**Status: fixed in the shard builder only.**

`Pool.imap_unordered` waits forever if a worker dies: the shard build hung for
1h50m with every child a zombie and the parent in `futex_do_wait`, having
written 9.5B good tokens and no manifest. `ProcessPoolExecutor` raises instead
- but `Executor.map` consumes its whole input iterable up front, which against
a streaming dataset queues the entire dataset before yielding anything. The
builder now submits a bounded window of futures by hand, which is lazy *and*
loud.

**Fixed:** `agent/dataset_builder.py` moved to `ProcessPoolExecutor` too.
`map()` is safe there because the job list is finite - it is only over a
*stream* that `Executor.map` becomes a trap by consuming the whole input up
front.

## 5. Manifests and metadata written only on the happy path

**Status: fixed in the shard builder; check others.**

The hung build had 19 valid shards and no `manifest.json`, because the write
came after the loop. Any long job whose output needs an index should write it
incrementally or in a `finally`. The teacher generators already learned this
(`save_atomic` + `.partial.json` resume, after a power cut destroyed hours of
work); the shard builder had to learn it separately.

## 6. Training run length is easy to get wrong by 7x

**Status: guarded 2026-09-15.** `Trainer` now prints
`horizon: N iters = X epochs` at construction and warns loudly above 10
epochs, naming the reason a late stop cannot rescue it.

`--iters` is optimizer steps, and epochs are
`batches_per_epoch / grad_accum`. 150,000 iters was described in this repo's
own notes as "2.5 epochs"; it is ~18. The LR schedule is cosine across
`--iters`, so a too-long run cannot be rescued by stopping early - the weights
never anneal. `scripts/train_split.py` should print epochs alongside iters and
refuse (or warn loudly) above some ratio.

## 7. Default checkpoints rot

**Status: one fixed, pattern live.**

`agent/conftest.py` defaulted to `checkpoints_v2_M`, whose vocabulary (271) is
older than the tokenizer the suite builds (273), so four tests errored on a
vocab assertion for weeks without anyone reading the message. Several scripts
carry similar hardcoded defaults (`scripts/run/04_eval.sh` →
`checkpoints_long_M`, `agent/chat.py` → `checkpoints_v2_M`). A checkpoint whose
config does not match the code should fail with that sentence, not a vocab
range error.

## 8. Gitignore rules that a typo defeats

**Status: fixed, worth remembering.**

`config.json` was created as `config..json`, which the `config.json` ignore
rule does not match - a live Discord webhook URL sat one `git add -A` from
being published. Now `config*.json` with `!config.example.json`. The earlier
`archive/` version of this: git cannot re-include a file whose parent directory
is excluded, so `archive/` + `!archive/notes/` silently did nothing and needed
`archive/*`.

## 9. Blocking and async notifications can arrive out of order

**Status: fixed 2026-09-15.** A blocking send now drains the queue first (up
to 15 s) so a finish or crash message cannot overtake the progress it reports
the end of.

`agent/notify.py` queues async posts on a daemon thread but sends
`blocking=True` inline, so a crash report can land before progress messages
queued earlier. Correct for the purpose (the blocking path exists because a
daemon thread dies at interpreter shutdown) but surprising in a channel.

## 11. `.gitignore` assumes a data layout that keeps changing

**Status: fixed each time, three instances so far.**

`archive/` needed `archive/*` plus a negation, because git cannot re-include a
file under an excluded directory. `config.json` was created as `config..json`,
which the rule did not match, leaving a live webhook URL one `git add -A` from
being published. Then `data/pretrain/` did not exist when the data rules were
written, so a `git add -A` swept **18 GB of packed shards** into a commit and
turned the push into a timeout - caught before it reached the remote.

The pattern: every new artefact directory is ignored *after* something goes
wrong. `data/` holds generated and regenerable things almost exclusively;
ignoring it wholesale and negating the few committed libraries
(`personas.json`, `chapters.json`, `knowledge*.json`) would invert the default
to the safe side.

## 10. Composed data is far less diverse than its trace count suggests

**Status: measured per slice 2026-09-15. Partly a property, partly fixable.**

"The corpus is bunk" is too broad - the damage is concentrated. Measured at
400 traces per generator:

| slice | repeat sentences | distinct-8 |
|---|---|---|
| OpenHermes (imported) | 0.1% | 0.963 |
| xlam (imported) | 0.0% | 0.827 |
| `rich_dataset` (tools) | 4.7% | 0.385 |
| `knowledge_traces` | 22.4% | 0.570 |
| `direct_traces` | 32.7% | 0.305 |
| **`project_traces`** | **81.3%** | **0.134** |

The tool traces - the part this project is actually for - are *fine*.
`project_traces` is the outlier by a wide margin, and 70% of its repetition
was `outline.md` coming back verbatim from every `read_file`. Paged reads
(`read_file` has taken `start`/`chars` since it was written; the generator
never used them) took it to 75.8% / 0.168: real, and nowhere near enough.

The ceiling is the library. 894 chapters over 400 arcs means each chapter is
written verbatim ~4.5 times before any arc repeats, and every arc writes its
chapters into files. No amount of composition fixes that - only more stories
would, and the teacher cost is the reason composition exists.

**So treat `project_traces` as a teacher of structure, not language**:
write-then-read-back continuity over a long horizon, worth keeping at a small
share. The SFT mix at 55% imported / 10% project measures 50.7% / 0.423
against the old instruct-only corpus's 78.8% / 0.288.

`direct_traces` is second worst for the same reason in miniature - its
`SMALL_TALK`, `REASONING` and `DECLINE` lists are a dozen hardcoded strings
each, so "Hello!" has exactly one correct answer in the entire corpus. That is
why the trained model answers it verbatim.
