#!/usr/bin/env bash
# Build and cache the tokenized dataset. CPU only - safe to run while a training
# job holds a GPU.
#
# Defaults reproduce the chain-depth mix of the dataset the current results were
# measured on: ~23% 0-1 calls, ~30% 2-3, ~47% four or more, tail past 40 hops.
# Half the held-out distribution needing 4+ calls is why that bucket dominates
# the score.
set -euo pipefail
cd "$(dirname "$0")/../.."

SAMPLES=${SAMPLES:-250000}
MAX_LEN=${MAX_LEN:-16384}
POOL=${POOL:-data/ollama_pool.json}
DEEP_FRACTION=${DEEP_FRACTION:-0.5}
DEEP_MIN=${DEEP_MIN:-4}
DEEP_MAX=${DEEP_MAX:-48}
SCENARIO_FRACTION=${SCENARIO_FRACTION:-0.2}
WORKERS=${WORKERS:-48}
SEED=${SEED:-42}

# ~6 GB on disk and ~20 minutes at these settings. The key is a hash of every
# parameter below, so changing any of them builds a new cache rather than
# reusing this one.
.venv/bin/python -u scripts/build_dataset.py \
  --mode instruct --pool "$POOL" \
  --samples "$SAMPLES" --max-len "$MAX_LEN" --seed "$SEED" \
  --deep-fraction "$DEEP_FRACTION" --deep-min "$DEEP_MIN" --deep-max "$DEEP_MAX" \
  --scenario-fraction "$SCENARIO_FRACTION" \
  --workers "$WORKERS"
