#!/usr/bin/env bash
# Conversational data for the chat-mode SFT: SmolTalk's subsets are built for
# models under 2B, which fits here far better than OpenHermes (7B-era). Math
# is in because the SFT'd M model answered "2 + 2" with "2 + 2".
#   screen -dmS import-chat scripts/run/08_import_chat.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
imp() { .venv/bin/python -u scripts/import_hf_dataset.py --dataset HuggingFaceTB/smoltalk "$@"; }
imp --config smol-magpie-ultra       --limit 250000 --out data/hf_smol_magpie.json
imp --config everyday-conversations  --limit 10000  --out data/hf_smol_everyday.json
imp --config systemchats-30k         --limit 30000  --out data/hf_smol_systemchats.json
imp --config metamathqa-50k          --limit 50000  --out data/hf_smol_math.json
imp --config smol-summarize          --limit 50000  --out data/hf_smol_summarize.json
echo "all imports done"
