# Dreams & Future Plans

This is the long-term wish list for newllm — things we want to build, explore, or prove. Nothing here is committed; it is a living record of directions worth chasing.

## Near-term goals (next few sessions)

- **Evaluate the L-size curriculum model.** Let the 151M-parameter agent finish training, then run the multi-hop test battery and compare to the 25M S-size baseline.
- **Stabilise multi-step tool use.** Make the agent reliably chain 3+ tool calls, handle malformed arguments, and recover from tool errors gracefully.
- **Add a real validation metric.** Track token-level accuracy on held-out traces, not just loss, so we can stop guessing whether the model actually learned the format.
- **Better answer extraction.** Return the last tool result as the answer only when appropriate; otherwise extract the explicit `Answer:` line after `END_THINK`.
- **Curriculum that ramps difficulty.** Start with single-step traces, then introduce two-step, then three-step, rather than mixing all difficulties from the start.

## Medium-term goals (weeks)

- **Scale to 1B parameters.** Train an XL-size model with MoE/MLA and/or sparse attention, tuning batch size and gradient accumulation to fit a 16GB GPU.
- **KV-cache inference.** Add a cached key/value store so generation is O(1) per new token instead of O(seq_len²).
- **Long-context competence.** Train the 8K-context story model to convergence with sparse/MLA attention and measure coherent continuation quality.
- **Real text corpus.** Load a small public corpus (e.g., Wikitext, TinyStories, or a curated domain dataset) and train a general language model, not just synthetic traces.
- **Sequence MoE that actually routes.** Make the sequence-level mixture-of-experts cluster by topic and generate coherent topic-specific continuations.
- **Tool expansion.** Add more useful tools: web fetch, file read/write, Python code execution (sandboxed), vector memory search, summarisation.
- **Self-correction loop.** Let the agent inspect its own trace, detect bad tool calls or wrong answers, and retry.

## Big dreams (months)

- **Autonomous research agent.** A model that can read code, run tests, propose fixes, and validate them — a tiny self-improving coding assistant.
- **Multi-modal input.** Extend the byte-level tokenizer to handle image patches or audio frames inside the same autoregressive framework.
- **Continuous learning.** Train on a stream of new tasks without forgetting the old ones, using replay buffers or modular experts.
- **Distributed training.** Scale across multiple GPUs or machines with data-parallel and pipeline-parallel training.
- **Quantisation and edge deployment.** Run a compressed 4-bit version of the best model on consumer hardware or export to ONNX/safetensors.
- **Reasoning-time compute.** Let the model spend more computation at inference time — beam search over tool chains, internal simulation, or verification steps.
- **Alignment and safety.** Add a simple reward model or rejection-sampling loop so the agent learns to refuse harmful or uncertain requests.

## Wild ideas (no guarantee)

- Train a model purely on its own generated reasoning traces, bootstrapping from a small seed of human-style examples.
- Build a "council of experts" where multiple small models debate an answer before producing a final result.
- Use the transformer as a world model: predict future observations, then plan actions with tree search.
- Create a tiny LLM operating system where the model reads files, writes code, and runs commands in a sandbox.

## Principles that guide the dreams

- Keep everything reproducible: every run writes a timestamped result JSON and a snapshot checkpoint.
- Measure before scaling: a 25M model that works is worth more than a 1B model that does not.
- Stay interpretable: explicit `BEGIN_THINK` blocks and tool traces make it possible to debug what the model is doing.
- All experiments must pass `smoke_test.py` and `test_modern_features.py` before being promoted to "best".

---

If you pick something from this list, move it into [goals.md](goals.md) as an in-progress item and update [progress.md](progress.md) as you go.
