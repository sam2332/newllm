#!/usr/bin/env bash
# Stage 1: pretrain on the packed FineWeb-Edu shards. One GPU (the 5090).
# Sized from measured throughput: 110k tokens/s at 86.9M parameters, so one
# pass over 9.5B tokens is ~24 h. 109 tokens per parameter is well past
# Chinchilla's 20:1, which is the right side to err on for a small model that
# has to be coherent rather than merely well-fitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
SIZE=${SIZE:-M}
SEQ=${SEQ:-2048}
BS=${BS:-16}
ACCUM=${ACCUM:-4}
ITERS=${ITERS:-72000}
LR=${LR:-6e-4}
OUT=${OUT:-checkpoints_pretrain}
GPU=${GPU:-1}
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/pretrain.py \
  --size "$SIZE" --seq-len "$SEQ" --batch-size "$BS" --grad-accum "$ACCUM" \
  --iters "$ITERS" --lr "$LR" --warmup 2000 \
  --val-windows 2000 --val-batches 40 --save-every 1000 \
  --out "$OUT" >> logs/pretrain.log 2>&1 < /dev/null &
sleep 2
echo "pretrain launched on GPU $GPU -> $OUT (tail logs/pretrain.log)"
