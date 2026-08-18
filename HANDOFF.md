# JSON Agent Migration Handoff

## Objective

Replace the legacy flat ReAct grammar with a consistent message protocol: send a user JSON-message context to the LLM; receive assistant JSON that contains explicit reasoning and either a tool call or final response.

## Contract

The byte-level model is trained on tagged text representations of messages:

```text
<user>What is 12 + 8?</user>
<assistant>{"thought":"I need the calculator.","tool_call":{"name":"calc","arguments":{"expr":"12 + 8"}}}</assistant>
<tool name=calc>20</tool>
<assistant>{"thought":"The calculation is complete.","response":"20"}</assistant>
```

Assistant schema:

```json
{
  "thought": "required string",
  "tool_call": {
    "name": "calc | now | search_memory | web_search | finish",
    "arguments": {}
  }
}
```

or:

```json
{
  "thought": "required string",
  "response": "final answer"
}
```

`thought` is required. `tool_call` and `response` are mutually exclusive.

## Changed Files

- `agent/agent_dataset.py`: synthetic tagged JSON-message traces; raw digits are retained.
- `agent/agent_loop.py`: parser/executor for assistant JSON messages and tool-result injection.
- `agent/train_agent.py`: uses the new dataset interface; old compact-math option removed.
- `agent/chat.py`: displays JSON-loop results.
- `agent/test_agent_loop.py`: parser and toolbox unit coverage for the new protocol.
- `agent/ollama_cot.py`: optional Kimi/Ollama CoT-data generator.

## Verified

- `python smoke_test.py`: passed.
- `python -m pytest agent/test_agent_loop.py -v`: passed, 9 tests.
- Synthetic single-step math traces are about 268 bytes; full multi-hop traces can reach 702 bytes. The JSON agent therefore defaults to a 768-token byte context.

## Current State and Blockers

1. The JSON protocol is not backward compatible with legacy ReAct checkpoints.
2. A quick JSON training diagnostic had low character-level loss but 0% task accuracy and no parsed final answers.
3. That diagnostic overwrote `checkpoints/agent_best.pt`; do not treat it as the best model or promote it.
4. The old regression/promotion expectations are for the previous agent family. Re-baseline them only after a viable JSON checkpoint exists.
5. Training serialization and inference serialization must be identical. Confirm that the system prompt, role tags, newlines, and tool tags match before spending GPU time.

## Recommended Continuation

1. Preserve or recover the prior legacy checkpoint as an archived artifact; do not run it through the JSON loop.
2. Make one shared message serializer used by both `agent/agent_dataset.py` and `agent/agent_loop.py` so training and inference prompts cannot drift.
3. Add unit tests that compare a generated training prefix with the inference context for the same question and tool result.
4. Train a fresh JSON checkpoint in `checkpoints_json/`, never the default best path.
5. Begin with a narrow math-only JSON dataset. Evaluate parsed JSON validity, tool-call rate, expression-copy accuracy, and numerical correctness separately.
6. Only then mix in memory, date, web, and multi-hop traces.
7. Update `agent/test_agent_regression.py` and `agent/promote.py` to use the JSON checkpoint path and a fresh baseline. Keep the 3-of-5 competition gate.

## Ollama CoT Data

Start Ollama with access to the requested Kimi cloud model, then run:

```powershell
python -m agent.ollama_cot --samples 100 --model kimi2.7-cloud --output agent/cot_data.json
```

This produces a JSON file with question, expected answer, task kind, and generated CoT. It is not yet mixed into the training dataset; validate its quality and decide how to serialize it before adding it to training.