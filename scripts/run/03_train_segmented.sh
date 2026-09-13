#!/usr/bin/env bash
# Train in short timed segments, for when a long run is not safe to leave up.
#
# --iters MUST stay identical across segments: it is the denominator of the
# cosine LR schedule, so raising it between segments rewrites the schedule
# mid-run. Each segment stops on the wall clock, writes resume state and exits
# 0; the next one continues from that step.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${GPU:-1}
ITERS=${ITERS:-40000}
SEGMENT=${SEGMENT:-20}            # minutes per segment
OUT=${OUT:-checkpoints_long_M}
CACHE=${CACHE:-}
if [ -z "$CACHE" ]; then
  CACHE=$(ls -S data/cache/instruct_*.pt 2>/dev/null | head -1 || true)
  [ -n "$CACHE" ] || { echo "no dataset cache; run 02_build_dataset.sh" >&2; exit 1; }
fi
mkdir -p logs "$OUT"
LOG=logs/train_segmented_$(date +%Y%m%d_%H%M).log
echo "segments of ${SEGMENT} min until ${ITERS} iters; log $LOG"

seg=0
while true; do
  seg=$((seg+1))
  resume=()
  [ -f "$OUT/agent_best.pt.resume" ] && resume=(--resume)
  echo "=== segment $seg $(date '+%H:%M:%S') ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/train_split.py \
    --mode instruct --size "${SIZE:-M}" --dataset-cache "$CACHE" \
    --max-len "${MAX_LEN:-16384}" --batch-size "${BATCH:-2}" --grad-accum "${ACCUM:-8}" \
    --grad-checkpoint --iters "$ITERS" --lr "${LR:-5e-4}" \
    --val-batches "${VAL_BATCHES:-50}" --eval-every "${EVAL_EVERY:-0}" \
    --save-every "${SAVE_EVERY:-500}" --max-minutes "$SEGMENT" \
    "${resume[@]}" --out "$OUT" >> "$LOG" 2>&1 || {
      echo "segment $seg failed; see $LOG" >&2; exit 1; }
  if grep -qa "saved ->" "$LOG"; then
    echo "run complete after $seg segments"; break
  fi
  tail -2 "$LOG" | tr '\r' '\n' | tail -1
done
