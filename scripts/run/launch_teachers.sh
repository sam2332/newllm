#!/bin/bash
# Start both knowledge teachers, one per GPU. Kept as a file on purpose:
# launching (or pkill-ing) these from an interactive command line puts the
# pattern "gen_knowledge_ollama.py" into the shell's own cmdline, and a
# later `pkill -f` then matches that shell and kills the whole session.
cd /home/lmeadows/llm
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --items 12000 --per-request 6 --workers 6 --artifact-fraction 0.4 \
  --endpoint http://localhost:11434 --out data/knowledge.json \
  > logs/gen_knowledge.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python -u scripts/gen_knowledge_ollama.py \
  --items 12000 --per-request 6 --workers 6 --artifact-fraction 0.4 --seed 101 \
  --endpoint http://localhost:11435 --out data/knowledge_b.json \
  > logs/gen_knowledge_b.log 2>&1 < /dev/null &
sleep 2
echo "launched; tail logs/gen_knowledge.log logs/gen_knowledge_b.log"
