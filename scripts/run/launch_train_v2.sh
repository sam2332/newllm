#!/bin/bash
# 5090 only (CUDA_VISIBLE_DEVICES=1); the 4090 is deliberately left alone.
cd /home/lmeadows/llm
setsid nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/train_split.py \
  --mode instruct --size deep-moe-20 \
  --dataset-cache data/cache/instruct_f6a8c14be1ad0ea0.pt \
  --tokenizer data/tokenizer/bpe8192_v2.json \
  --max-len 32768 --max-tokens 16384 --max-batch 64 --grad-accum 8 \
  --grad-checkpoint --iters 30000 --lr 4e-4 \
  --eval-every 0 --val-batches 40 --save-every 500 \
  --out checkpoints_moe_v2 >> logs/train_moe_v2.log 2>&1 < /dev/null &
sleep 2
echo "training launched on the 5090"
