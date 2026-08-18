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

4. **Increase model capacity (active)**
   - M-size (~60M) math-only diagnostic is now running: `scripts/train_web_agent_m_math_only.ps1`.
   - If M-size learns arithmetic, the problem is capacity; we can then re-add tasks on top of an M-size foundation.
   - If M-size also fails, the problem is the byte-level representation or the copying task itself and we need a structural fix (pointer / constrained decoding / separate number encoder).

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

9. **5-check promotion competition (implemented)**
   - `agent/promote.py` now compares candidate vs current best on 5 checks: overall accuracy, numeric accuracy, exact accuracy, required-question pass rate, and a robustness composite score.
   - Candidate must pass absolute thresholds AND win ≥3/5 checks to promote. This prevents regressions.
