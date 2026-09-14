#!/bin/bash
# Instance A, on the 4090. qwen3:30b-a3b-q8_0 is 34 GB and spills ~10 GB onto
# the CPU on a 24 GB card, which pegged 48 cores and made this the slow half.
# qwen2.5-coder:32b is 22.1 GB - fully resident, no spill - so A takes the
# code domains and leaves science to the q8 teacher on the 5090.
cd /home/lmeadows/llm
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --model qwen2.5-coder:32b --items 8000 --per-request 6 --workers 6 \
  --artifact-fraction 0.4 \
  --domain python --domain bash --domain algorithms \
  --domain debugging --domain coding_principles \
  --endpoint http://localhost:11434 --out data/knowledge.json \
  >> logs/gen_knowledge.log 2>&1 < /dev/null &
sleep 2
echo "A launched (coder:32b, code domains)"
