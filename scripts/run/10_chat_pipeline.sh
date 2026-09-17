#!/usr/bin/env bash
# Drive the L24 chat model to the end without supervision:
#   1. pretrain, resuming from the last saved state when there is one
#   2. chat-first SFT (09_sft_chat.sh)
#   3. coherence probe, with the sky answer posted to the webhook
# One GPU job at a time, always. Normally run by systemd
# (scripts/systemd/llm-training.service), which restarts it after a crash or a
# reboot; by hand, inside a screen:
#   screen -dmS chat-pipeline scripts/run/10_chat_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=checkpoints_pretrain_L24
LOG=logs/pretrain_L24.log
# Saves every 5,000 iters (~15 min): a reboot at 23 min into the first run
# lost everything because the first save was due at 10,000.
PRE_ARGS=(--size L24 --seq-len 2048 --batch-size 4 --grad-accum 8 --iters 200000
          --lr 4e-4 --warmup 8000 --val-windows 2000 --val-batches 40
          --save-every 5000 --compile --out "$OUT")
say() { .venv/bin/python -c "import sys; sys.path.insert(0,'.'); from agent.notify import notify; notify(sys.argv[1], tag='pipeline', blocking=True)" "$1"; echo "$1"; }
running() { pgrep -f "scripts/pretrain.py --size L24" >/dev/null; }
done_pretrain() { [ -f "$OUT/run.json" ]; }

# Everything already done: exit, so a boot-time start does not redo the SFT.
if [ -f logs/probe_sft_chat.log ] && [ -f checkpoints_sft_chat/agent_best.pt ]; then
  echo "chat pipeline already complete"; exit 0
fi

while ! done_pretrain; do
  if running; then sleep 60; continue; fi
  RESUME=()
  if [ -f "$OUT/pretrain_best.pt.resume" ]; then
    RESUME=(--resume)
    say "pretrain L24 starting from its last saved state"
  else
    say "pretrain L24 starting from scratch (no saved state yet)"
  fi
  # Both GPUs (DDP): the power was fixed with the PSU swap and a 20-minute
  # dual load held. iters/grad-accum are per rank, halved from the 1-GPU run
  # so tokens per optimizer step and the total stay the same.
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/torchrun --standalone --nproc_per_node=2 scripts/pretrain.py "${PRE_ARGS[@]}" "${RESUME[@]}" 2>&1 | tee -a "$LOG"
  # A crash exits non-zero: leave, and let systemd restart after its back-off
  # rather than spinning here.
  [ "${PIPESTATUS[0]}" = 0 ] || { say "pretrain L24 exited with an error"; exit 1; }
done
say "pretrain L24 finished; starting chat SFT"

if ! scripts/run/09_sft_chat.sh; then say "chat SFT failed; see logs/sft_chat.log"; exit 1; fi

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/coherence_probe.py checkpoints_sft_chat/agent_best.pt \
  2>&1 | tee logs/probe_sft_chat.log
SKY=$(grep -A2 -- "--- sky ---" logs/probe_sft_chat.log | grep "A:" | cut -c1-1500)
say ":white_check_mark: **chat model done.** Why is the sky blue? ${SKY:-<no answer parsed>}"
