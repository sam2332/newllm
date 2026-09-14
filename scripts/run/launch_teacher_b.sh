#!/bin/bash
# Instance B, on the 5090: qwen3:30b-a3b-q8_0 (34 GB, 31.9 of it resident),
# all six domains including science, which the coder model on A does not cover.
cd /home/lmeadows/llm
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --items 12000 --per-request 6 --workers 6 --artifact-fraction 0.4 --seed 101 \
  --endpoint http://localhost:11435 --out data/knowledge_b.json \
  >> logs/gen_knowledge_b.log 2>&1 < /dev/null &
sleep 2
echo "B launched (q8, all domains)"
