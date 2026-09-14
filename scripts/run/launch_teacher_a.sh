#!/bin/bash
# Instance A used to run qwen2.5-coder:32b on the 4090. It was the slowest of
# the three (7.5 items/min against the cloud teacher's 44) and covered code
# domains the cloud model covers anyway, so A moved to cloud and the 4090 is
# free for training. It keeps the same output file, so it resumes the 4,365
# items the local run had already written.
# Four workers rather than five: instance C is already using the same free
# tier, and concurrent bursts earn a 429 on every request.
cd /home/lmeadows/llm
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --model gpt-oss:120b-cloud --think low \
  --items 8000 --per-request 6 --workers 4 --artifact-fraction 0.4 \
  --domain python --domain bash --domain algorithms \
  --domain debugging --domain coding_principles \
  --endpoint http://localhost:11435 --out data/knowledge.json \
  >> logs/gen_knowledge.log 2>&1 < /dev/null &
sleep 2
echo "A launched (gpt-oss:120b-cloud, code domains)"
