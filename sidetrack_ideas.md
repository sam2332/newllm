# Sidetrack / future experiment ideas

These are ideas that came up while trying to make the JSON+web_search S-size agent pass the promotion gate. They are deliberately **not** the main path right now so we can finish one experiment at a time, but they should be easy to pick up later.

1. **Digit-word expansion (tried, did not help)**
   - Spell numbers as words in the dataset so the byte-level tokenizer can copy them more easily.
   - Collapse words back to digits in the `calc` tool and final answer extraction.
   - Result: the 25M model still scored ~23.5%. It produced malformed expressions like `three + one three` and sometimes picked the wrong tool (`now` for math).
   - Lesson: the bottleneck is not just digit-string copying; the model struggles with the whole multi-task mapping.

2. **Math-only pre-training stage (S-size result)**
   - Trained only on single-step math traces for 3k iters. Loss dropped to ~0.13 but eval accuracy was only 5.88%.
   - The model still got `12 + 8` wrong (`148`), `15 * 4` malformed, `7 * 6` = `84`.
   - Conclusion: the 25M model is too small (or the byte-level representation is too weak) to learn arithmetic copying even in isolation. This is now a **capacity** experiment.

3. **Pointer / constrained decoding for calc expr**
   - Force the model to copy tokens from the question into the `expr` argument using a pointer network or constrained beam search.
   - More invasive but would guarantee correct expressions.

4. **Increase model capacity (tried)**
   - M-size (~60M) math-only diagnostic completed: 5.88% accuracy, same as S-size.
   - Larger capacity did **not** fix arithmetic copying. The problem is not model size.

5. **Separate web_search query vocabulary**
   - Give the web_search tool a small set of canonical query strings from the knowledge base instead of free-form natural-language questions.
   - Might make the lookup task much easier.

6. **Reward/RL-style fine-tuning**
   - After supervised pretraining, sample candidate answers and give higher weight to traces that reach the correct numeric answer.
   - Could fix the last-mile accuracy problem.

7. **Smaller effective batch / longer stage 1**
   - The current `batch 16 x accum 8 = 128` is large. A smaller batch might help the model learn rare copy patterns faster.

8. **Diagnostic eval during training**
   - Add a small fixed eval set that is printed every 500 steps so we can see when arithmetic accuracy starts to improve.

9. **Single-digit math-only diagnostic (done)**
   - S-size single-digit math-only run completed: 5.88% overall accuracy (because eval includes unseen multi-digit/memory/web), but it produced valid single-digit expressions.
   - Crucially, for `What is 7 * 6?` it generated `five * two` → 10, not `seven * six` → 42. It is **not copying from the question**; it learned an expression prior.
   - This means the problem is not multi-token copying; it's **grounding/copying the expression from the question**.

10. **5-check promotion competition (implemented)**
    - `agent/promote.py` now compares candidate vs current best on 5 checks: overall accuracy, numeric accuracy, exact accuracy, required-question pass rate, and a robustness composite score.
    - Candidate must pass absolute thresholds AND win ≥3/5 checks to promote. This prevents regressions.

11. **Structural fixes (active)**
    - Add a compact math format where the question literally contains the expression, e.g. `calc(seven * six) = ?`. The target `expr` is exactly `seven * six` so the model only has to copy the substring inside the parentheses. This tests whether the model can copy when the expression is contiguous and clearly marked.
    - If that still fails, try pointer / constrained decoding for `calc` expr.
    - If that fails, consider a separate number encoder or BPE tokenizer.
