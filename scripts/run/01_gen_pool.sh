#!/usr/bin/env bash
# Widen the language pool with the Ollama teacher, then MERGE into the existing
# pool. gen_data_ollama.py builds its output from scratch and overwrites --out,
# so pointing it at data/ollama_pool.json would discard what is already there.
#
# The teacher never produces a tool result. It supplies phrasings, reasoning
# sentences and factual key/value pairs only; every observation in every emitted
# trace comes from the real Toolbox. Do not relax that.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3:30b-a3b-q8_0}
THOUGHTS=${THOUGHTS:-250}       # per situation (16 situations)
PARAPHRASES=${PARAPHRASES:-120} # per intent (4 intents)
FACTS=${FACTS:-150}             # per category
WORKERS=${WORKERS:-8}
POOL=${POOL:-data/ollama_pool.json}
STAMP=$(date +%Y%m%d_%H%M)
RAW=data/ollama_pool_gen_${STAMP}.json

# Uses BOTH endpoints (11434 + 11435), so both GPUs are active. Inference draws
# far less than training - measured at 113 W + 170 W - but do not run this
# alongside a training job without thinking about the breaker.
echo "generating -> $RAW  (model $MODEL, $WORKERS workers, both endpoints)"
.venv/bin/python -u scripts/gen_data_ollama.py \
  --model "$MODEL" --workers "$WORKERS" \
  --thoughts-per-situation "$THOUGHTS" \
  --paraphrases "$PARAPHRASES" \
  --facts-per-category "$FACTS" \
  --out "$RAW"

cp -n "$POOL" "${POOL%.json}.backup_${STAMP}.json"
.venv/bin/python - "$POOL" "$RAW" <<'PY'
import json, sys, shutil
pool_path, raw_path = sys.argv[1], sys.argv[2]
old = json.load(open(pool_path)); new = json.load(open(raw_path))
merged = {}
for key in ("thoughts", "paraphrases"):
    merged[key] = {}
    for n in set(old.get(key, {})) | set(new.get(key, {})):
        merged[key][n] = sorted(set(old.get(key, {}).get(n, [])) |
                                set(new.get(key, {}).get(n, [])))
# Keep the OLD value on a key collision, so traces generated earlier stay
# reproducible against the same facts.
merged["facts"] = dict(new.get("facts", {})); merged["facts"].update(old.get("facts", {}))
for k, v in old.items():
    merged.setdefault(k, v)          # e.g. an attached sandbox pool
def n(p, k):
    return len(p.get(k, {})) if k == "facts" else sum(len(v) for v in p.get(k, {}).values())
print(f"{'axis':12s} {'old':>8s} {'new':>8s} {'merged':>8s}")
for k in ("thoughts", "paraphrases", "facts"):
    print(f"{k:12s} {n(old,k):8,} {n(new,k):8,} {n(merged,k):8,}")
json.dump(merged, open(pool_path, "w"), indent=1)
print(f"\nwrote {pool_path}")
PY
echo "raw generation kept at $RAW"
echo "NOTE: the pool changed, so every dataset cache built from it is stale."
echo "      Run 02_build_dataset.sh next."
