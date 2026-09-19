#!/usr/bin/env bash
# Stage 2, conversational: SFT the L24 base on a chat-first mix. One GPU.
#
# The target is multi-turn conversation in ChatML, so imported conversations
# carry 75% and the synthetic slices are small: knowledge Q&A for grounded
# facts, personas for roleplay, a little direct and project data so tool
# habits are not lost. SmolTalk replaces OpenHermes - it was built for
# sub-2B models. hf_disjoint keeps any imported row from repeating.
#
#   screen -dmS sft-chat scripts/run/09_sft_chat.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
INIT=${INIT:-checkpoints_pretrain_L24/pretrain_best.pt}
SIZE=${SIZE:-L24}
TOK=${TOK:-data/tokenizer/bpe8192_v2.json}
SAMPLES=${SAMPLES:-400000}
MAXLEN=${MAXLEN:-4096}
ITERS=${ITERS:-40000}
LR=${LR:-1e-4}
OUT=${OUT:-checkpoints_sft_chat}
GPU=${GPU:-1}
HF=${HF:-data/hf_smol_magpie.json,data/hf_smol_everyday.json,data/hf_smol_systemchats.json,data/hf_smol_math.json,data/hf_smol_summarize.json,data/hf_xlam.json}

# spawn-pool workers re-import __main__ by path: build from a file, not stdin.
BUILD_PY=$(mktemp --suffix=.py)
PATH_OUT=$(mktemp)
trap 'rm -f "$BUILD_PY" "$PATH_OUT"' EXIT
cat > "$BUILD_PY" <<PY
import sys
sys.path.insert(0, "$PWD")
from agent.dataset_builder import build
if __name__ == "__main__":
    info = {}
    build(samples=$SAMPLES, mode="instruct", max_len=$MAXLEN, workers=16,
          tokenizer_spec={"kind": "bpe", "path": "$TOK", "vocab_size": 8192},
          protocol="chatml", verbose=True, info=info,
          extras=dict(hf_fraction=0.75, hf_traces="$HF", hf_disjoint=True,
                      knowledge_fraction=0.08, knowledge="data/knowledge.json,data/knowledge_b.json,data/knowledge_c.json",
                      persona_fraction=0.10, personas="data/personas.json",
                      direct_fraction=0.04,
                      project_fraction=0.02, stories="data/chapters.json,data/hf_stories.json", max_turns=52,
                      format_fraction=0.01))
    open("$PATH_OUT", "w").write(info["path"])
PY
.venv/bin/python "$BUILD_PY"
CACHE=$(cat "$PATH_OUT")
echo "dataset: $CACHE"

# Runs in the foreground: launch this inside a named screen.
#
# --grad-checkpoint is not optional at this size. L24 is 283M dense against the
# M preset's 86.9M, and without it a 16,384-token micro-batch OOMs in the
# backward of iter 1 (2026-09-18: 31.09 of the 5090's 31.45 GiB in use, short
# by 512 MiB). With it the same batch peaks at 10.7 GiB and runs at 2.2 it/s,
# so the whole 40,000 iters is ~5 h. Recomputation is cheap here; the headroom
# is what keeps a long trace late in the run from ending it.
env CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/train_split.py \
  --mode instruct --size "$SIZE" --dataset-cache "$CACHE" --tokenizer "$TOK" \
  --init-checkpoint "$INIT" \
  --max-len "$MAXLEN" --max-tokens 16384 --max-batch 64 --grad-accum 4 \
  --grad-checkpoint \
  --iters "$ITERS" --lr "$LR" --eval-every 0 --val-batches 40 --save-every 1000 \
  --out "$OUT" 2>&1 | tee -a logs/sft_chat.log
exit "${PIPESTATUS[0]}"
