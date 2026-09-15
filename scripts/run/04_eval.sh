#!/usr/bin/env bash
# Score a checkpoint. Defaults to the measure that actually reflects the data.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=${1:-checkpoints_long_M/agent_best.pt}
N=${N:-100}
GPU=${GPU:-1}
CACHE=${CACHE:-}
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
if [ -z "$CACHE" ]; then
  CACHE=$(ls -S data/cache/instruct_*.pt 2>/dev/null | head -1 || true)
fi

# Held-out traces from the same split and seed the trainer used. This is the
# number to quote. The 17-case battery below cannot resolve finer than 5.9
# points and 13 of its cases are single-hop toy questions, so it is a smoke
# test, not an estimate of accuracy.
# Which harness can read this checkpoint is a property of the checkpoint, not
# a flag to remember: a byte-tokenizer eval pointed at a ChatML model reports
# 0/0 without ever querying it, which reads as a broken model.
PROTO=$(.venv/bin/python - "$CKPT" <<'PY'
import sys, torch
# Read the config only. Building the model to answer one question about the
# tokenizer would load 2.2 GB of weights for nothing.
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ckpt.get("config", ckpt) if isinstance(ckpt, dict) else {}
tok = (cfg or {}).get("tokenizer") or {}
print("chatml" if tok.get("kind") == "bpe" else "json")
PY
) || PROTO=json

echo "=== held-out traces (n=$N, protocol $PROTO) ==="
if [ "$PROTO" = "chatml" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/eval_chatml_random.py \
    "$CKPT" -n "$N" ${CACHE:+--cache "$CACHE"}
  BATTERY=0
else
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/eval_random.py \
    "$CKPT" -n "$N" ${CACHE:+--cache "$CACHE"}
fi

if [ "${BATTERY:-1}" = "1" ]; then
  echo; echo "=== 17-case battery (smoke test) ==="
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/eval_agent.py "$CKPT" || true
fi

# A large gap here means the model went back to memorizing tool names and cannot
# serve a user's own tools. Treat it as a blocking defect.
if [ "${RENAMED:-0}" = "1" ]; then
  echo; echo "=== renamed-tool gap ==="
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/eval_renamed.py "$CKPT" || true
fi
