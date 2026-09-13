#!/usr/bin/env bash
# Check everything that has silently killed a run before. Read-only.
set -uo pipefail
cd "$(dirname "$0")/../.."
fail=0
ok(){ printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }
warn(){ printf '  \033[33mwarn\033[0m %s\n' "$1"; }

echo "== interpreter =="
if [ -x .venv/bin/python ]; then
  ok "$(.venv/bin/python -V 2>&1) at .venv/bin/python"
  .venv/bin/python -c "import torch" 2>/dev/null \
    && ok "torch $(.venv/bin/python -c 'import torch;print(torch.__version__)')" \
    || bad "torch not importable - see scripts/run/README.md"
else
  bad ".venv missing - see scripts/run/README.md"
fi

echo "== background jobs survive logout =="
# Linger=no means systemd kills the whole user slice at logout, taking any
# training run with it. nohup and setsid do not help.
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then
  ok "Linger=yes"
else
  bad "Linger=no - run: loginctl enable-linger $USER"
fi

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total \
  --format=csv,noheader | while IFS=, read -r i name lim def mem; do
  printf '  GPU%s%s  cap%s (stock%s )%s\n' "$i" "$name" "$lim" "$def" "$mem"
done
# Sustained draw on both GPUs at stock limits (~1025 W of GPU alone) browned the
# machine out. Caps make a single-GPU run sit near 350 W.
capped=$(nvidia-smi --query-gpu=power.limit,power.default_limit \
         --format=csv,noheader,nounits | awk -F, '$1+0 < $2+0 {n++} END{print n+0}')
[ "$capped" -ge 1 ] && ok "$capped GPU(s) power-capped below stock" \
                    || warn "no power cap set; see README 'Power'"

echo "== data =="
[ -f data/ollama_pool.json ] && ok "pool $(du -h data/ollama_pool.json | cut -f1)" \
                             || bad "data/ollama_pool.json missing"
if [ -d data/cache ] && [ -n "$(ls -A data/cache 2>/dev/null)" ]; then
  for f in data/cache/*.pt; do
    meta="${f%.pt}.json"
    [ -f "$meta" ] && ok "cache $(basename "$f") ($(du -h "$f"|cut -f1)) + params" \
                   || warn "cache $(basename "$f") ($(du -h "$f"|cut -f1)) has no params sidecar; reuse with --dataset-cache"
  done
else
  warn "no dataset cache; 02_build_dataset.sh will build one (~20 min)"
fi

echo "== disk =="
avail=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
[ "$avail" -ge 50 ] && ok "${avail}G free" || bad "${avail}G free - a run writes ~1.2 GB per resume state"

echo "== ollama (only needed for 01_gen_pool.sh) =="
for p in 11434 11435; do
  curl -s -m 5 "http://localhost:$p/api/tags" >/dev/null 2>&1 \
    && ok "responding on :$p" || warn "not responding on :$p"
done

echo
[ "$fail" -eq 0 ] && echo "preflight passed" || { echo "preflight FAILED"; exit 1; }
