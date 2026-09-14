#!/bin/bash
# Instance C runs on Ollama's cloud - no local GPU at all - so it stacks on
# top of both local teachers instead of competing with them. gpt-oss:120b is
# on the free tier (kimi and minimax return 402 without credits).
# think=low matters: at "false" the model still spends ~173 tokens reasoning
# before emitting content, which returns empty replies on short budgets.
cd /home/lmeadows/llm
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --model gpt-oss:120b-cloud --think low \
  --items 12000 --per-request 6 --workers 5 --artifact-fraction 0.4 --seed 202 \
  --endpoint http://localhost:11435 --out data/knowledge_c.json \
  >> logs/gen_knowledge_c.log 2>&1 < /dev/null &
sleep 2
echo "C launched (gpt-oss:120b-cloud, all domains)"
