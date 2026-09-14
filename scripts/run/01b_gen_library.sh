#!/usr/bin/env bash
# Generate the persona pool and the chapter library with the Ollama teacher.
# Both endpoints (11434 + 11435) are used, so both GPUs are busy and their
# memory is full - do not run training or evals at the same time. Inference
# draw is modest (~250 W total measured).
set -euo pipefail
cd "$(dirname "$0")/../.."
STORIES=${STORIES:-150}
CHAPTERS=${CHAPTERS:-6}
PERSONA_ROUNDS=${PERSONA_ROUNDS:-2}
mkdir -p logs
.venv/bin/python -u scripts/gen_personas_ollama.py --per-seed 4 --rounds "$PERSONA_ROUNDS" --workers 6 \
  2>&1 | tee logs/gen_personas.log
.venv/bin/python -u scripts/gen_chapters_ollama.py --stories "$STORIES" --chapters "$CHAPTERS" --workers 5 \
  2>&1 | tee logs/gen_chapters.log
echo "personas: $(.venv/bin/python -c "import json;print(len(json.load(open('data/personas.json'))))")"
echo "stories : $(.venv/bin/python -c "import json;print(len(json.load(open('data/chapters.json'))))")"
