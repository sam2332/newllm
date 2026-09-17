#!/usr/bin/env bash
# Drive the L24 chat model to the end without supervision:
#   1. wait for the running pretrain (or resume it until it completes)
#   2. chat-first SFT (09_sft_chat.sh)
#   3. coherence probe, with the sky answer posted to the webhook
# One GPU job at a time, always. Run inside a screen:
#   screen -dmS chat-pipeline scripts/run/10_chat_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=checkpoints_pretrain_L24
LOG=logs/pretrain_L24.log
PRE_ARGS=(--size L24 --seq-len 2048 --batch-size 4 --grad-accum 16 --iters 1150000
          --lr 4e-4 --warmup 16000 --val-windows 2000 --val-batches 40
          --save-every 10000 --out "$OUT")
say() { .venv/bin/python -c "import sys; sys.path.insert(0,'.'); from agent.notify import notify; notify(sys.argv[1], tag='pipeline', blocking=True)" "$1"; echo "$1"; }
running() { pgrep -f "^.venv/bin/python -u scripts/pretrain.py --size L24" >/dev/null; }
done_pretrain() { [ -f "$OUT/run.json" ]; }

while ! done_pretrain; do
  if running; then sleep 60; continue; fi
  if [ -f "$OUT/pretrain_best.pt.resume" ]; then
    say "pretrain L24 not running and not finished - resuming"
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u scripts/pretrain.py "${PRE_ARGS[@]}" --resume 2>&1 | tee -a "$LOG"
  else
    say "pretrain L24 stopped with no resume state; pipeline halted"; exit 1
  fi
done
say "pretrain L24 finished; starting chat SFT"

if ! scripts/run/09_sft_chat.sh; then say "chat SFT failed; see logs/sft_chat.log"; exit 1; fi

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/coherence_probe.py checkpoints_sft_chat/agent_best.pt \
  2>&1 | tee logs/probe_sft_chat.log
SKY=$(grep -A2 -- "--- sky ---" logs/probe_sft_chat.log | grep "A:" | cut -c1-1500)
say ":white_check_mark: **chat model done.** Why is the sky blue? ${SKY:-<no answer parsed>}"
