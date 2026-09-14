#!/bin/bash
# Restart either teacher if it dies. Both have now died silently mid-run -
# once to a `pkill -f` that matched the launching shell, once with no
# traceback at all - and an unattended 8-hour generation that quietly stops
# after 20 minutes is the expensive failure. Every generator resumes from
# its .partial.json, so a restart costs at most --save-every requests.
cd /home/lmeadows/llm
while true; do
  if ! pgrep -f "out data/knowledge.json" > /dev/null; then
    echo "$(date +%H:%M:%S) teacher A down, restarting" >> logs/supervise.log
    bash scripts/run/launch_teacher_a.sh >> logs/supervise.log 2>&1
  fi
  if ! pgrep -f "out data/knowledge_b.json" > /dev/null; then
    echo "$(date +%H:%M:%S) teacher B down, restarting" >> logs/supervise.log
    bash scripts/run/launch_teacher_b.sh >> logs/supervise.log 2>&1
  fi
  sleep 60
done
