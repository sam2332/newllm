#!/usr/bin/env bash
# Export a checkpoint to GGUF, load it into the Ollama docker instance, and
# prove it is the same model (next-token logprobs vs PyTorch).
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT=${1:-checkpoints_moe/agent_best.pt}
NAME=${NAME:-newllm}
CONTAINER=${CONTAINER:-ollama-4090}
DTYPE=${DTYPE:-f16}
OUT="$(dirname "$CKPT")/$NAME.gguf"
.venv/bin/python -u scripts/export_gguf.py "$CKPT" --out "$OUT" --dtype "$DTYPE" --name "$NAME"
CUDA_VISIBLE_DEVICES=${GPU:-1} .venv/bin/python -u scripts/test_export_gguf.py "$CKPT" --gguf "$OUT" \
  --container "$CONTAINER" --name "$NAME"
echo
echo "now: docker exec $CONTAINER ollama run $NAME"
echo "or : .venv/bin/python scripts/test_ollama_client.py --host http://localhost:11434 --model $NAME"
