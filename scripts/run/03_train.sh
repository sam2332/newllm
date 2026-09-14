#!/usr/bin/env bash
# Train on ONE GPU. Detaches, survives logout, and prints how to follow it.
#
# Single GPU is deliberate. Two GPUs at stock limits is ~1025 W of GPU before
# the CPU, which was hard-rebooting the machine mid-run. One capped GPU measures
# ~350 W and has completed 89-minute runs without incident.
#
# Set MAX_MINUTES to run in timed segments instead (see 03_train_segmented.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${GPU:-1}                      # 1 = RTX 5090, 32 GB
ITERS=${ITERS:-40000}
LR=${LR:-5e-4}
SIZE=${SIZE:-M}
OUT=${OUT:-checkpoints_long_M}
MAX_LEN=${MAX_LEN:-16384}
# batch 2 is not a memory compromise, it is the fastest setting. collate_pad
# pads to the longest sequence in the batch and the length spread is wide
# (p50 1624, max 14584), so one outlier drags a big batch to its length:
#   bs=2  accum=8  3,541 opt-steps/hr   2.7 GiB  341 W
#   bs=8  accum=2  2,748 opt-steps/hr  12.1 GiB  450 W
#   bs=16 accum=1  2,315 opt-steps/hr  17.7 GiB  450 W
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-8}
# Full validation is 12,499 samples and costs ~4 min a pass. Capped, it is
# seconds, and val_loader does not shuffle so the subset is a fair comparison.
VAL_BATCHES=${VAL_BATCHES:-50}
# The 17-case battery is OFF by default as a training signal. One case is worth
# 5.9 points, and it did real damage twice: it early-stopped a run at step 3000
# of 12000 (val 0.239, where the full run reached 0.0709), and it then selected
# step-7000 weights over better step-11499 ones. Validation loss picks the
# weights instead. Score the finished checkpoint with 04_eval.sh.
EVAL_EVERY=${EVAL_EVERY:-0}
EVAL_PATIENCE=${EVAL_PATIENCE:-10}
SAVE_EVERY=${SAVE_EVERY:-500}
CACHE=${CACHE:-}
MAX_MINUTES=${MAX_MINUTES:-0}
RESUME=${RESUME:-0}
# Retrain pipeline. The tokenizer must be the one the cache was built with
# (the cache's .json sidecar records it); arch 3 exports to GGUF.
TOKENIZER=${TOKENIZER:-byte}
ARCH_VERSION=${ARCH_VERSION:-3}
ROPE_BASE=${ROPE_BASE:-}

if [ -z "$CACHE" ]; then
  CACHE=$(ls -S data/cache/instruct_*.pt 2>/dev/null | head -1 || true)
  [ -n "$CACHE" ] || { echo "no dataset cache; run 02_build_dataset.sh" >&2; exit 1; }
  echo "using largest cache found: $CACHE"
fi

mkdir -p logs "$OUT"
STAMP=$(date +%Y%m%d_%H%M)
LOG=logs/train_${OUT##*/}_${STAMP}.log
args=(--mode instruct --size "$SIZE" --dataset-cache "$CACHE"
      --max-len "$MAX_LEN" --batch-size "$BATCH" --grad-accum "$ACCUM"
      --grad-checkpoint --iters "$ITERS" --lr "$LR"
      --val-batches "$VAL_BATCHES" --eval-every "$EVAL_EVERY"
      --eval-patience "$EVAL_PATIENCE" --save-every "$SAVE_EVERY" --out "$OUT"
      --tokenizer "$TOKENIZER" --arch-version "$ARCH_VERSION")
[ -n "$ROPE_BASE" ] && args+=(--rope-base "$ROPE_BASE")
[ "$MAX_MINUTES" != "0" ] && args+=(--max-minutes "$MAX_MINUTES")
[ "$RESUME" = "1" ] && args+=(--resume)

# setsid so the run outlives this shell. Note: $! is the wrapper, NOT the
# trainer - always find the real pid by pattern.
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u \
  scripts/train_split.py "${args[@]}" > "$LOG" 2>&1 < /dev/null &
sleep 6
PID=$(pgrep -f "train_split.py --mode instruct" | head -1 || true)
echo "log : $LOG"
echo "pid : ${PID:-<starting>}"
echo "eta : ~$(awk "BEGIN{printf \"%.0f\", $ITERS/9.4/60}") min at 9.4 it/s, plus the first cache load (~96 s)"
echo
echo "follow:  tr '\\r' '\\n' < $LOG | grep -a training: | tail -1"
echo "stop  :  kill \$(pgrep -f 'train_split.py --mode instruct' | head -1)"
