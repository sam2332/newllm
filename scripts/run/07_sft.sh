#!/usr/bin/env bash
# Stage 2: SFT on top of a pretrained base. One GPU (the 5090).
#
# The mix is the point. The instruct-only run memorised a corpus that was
# 78.8% repeated sentences, so the synthetic slices are deliberately a
# minority here and the imported data - 300k OpenHermes conversations and
# 60k xlam tool calls across 3,605 tool names - carries the weight.
#
#   INIT=checkpoints_pretrain/pretrain_best.pt scripts/run/07_sft.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
INIT=${INIT:-checkpoints_pretrain/pretrain_best.pt}
SIZE=${SIZE:-M}
TOK=${TOK:-data/tokenizer/bpe8192_v2.json}
SAMPLES=${SAMPLES:-300000}
MAXLEN=${MAXLEN:-4096}
ITERS=${ITERS:-12000}
LR=${LR:-2e-4}
OUT=${OUT:-checkpoints_sft}
GPU=${GPU:-1}

CACHE=$(.venv/bin/python - <<PY
import sys
sys.path.insert(0, ".")
from agent.dataset_builder import build
info = {}
build(samples=$SAMPLES, mode="instruct", max_len=$MAXLEN, workers=16,
      tokenizer_spec={"kind": "bpe", "path": "$TOK", "vocab_size": 8192},
      protocol="chatml", verbose=True, info=info,
      extras=dict(hf_fraction=0.55, hf_traces="data/hf_openhermes.json,data/hf_xlam.json",
                  knowledge_fraction=0.15, knowledge="data/knowledge.json,data/knowledge_b.json,data/knowledge_c.json",
                  project_fraction=0.10, stories="data/chapters.json", max_turns=52,
                  direct_fraction=0.05, persona_fraction=0.10,
                  personas="data/personas.json", format_fraction=0.05))
print(info["path"])
PY
)
echo "dataset: $CACHE"
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/train_split.py \
  --mode instruct --size "$SIZE" --dataset-cache "$CACHE" --tokenizer "$TOK" \
  --init-checkpoint "$INIT" \
  --max-len "$MAXLEN" --max-tokens 16384 --max-batch 64 --grad-accum 4 \
  --iters "$ITERS" --lr "$LR" --eval-every 0 --val-batches 40 --save-every 500 \
  --out "$OUT" >> logs/sft.log 2>&1 < /dev/null &
sleep 2
echo "SFT launched on GPU $GPU -> $OUT (tail logs/sft.log)"
