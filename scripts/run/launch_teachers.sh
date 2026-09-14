#!/bin/bash
# Both knowledge teachers, one per GPU. Kept as files on purpose: launching
# or pkill-ing these from an interactive command line puts the pattern
# "gen_knowledge_ollama.py" into the shell's own cmdline, and a later
# `pkill -f` then matches that shell and kills the whole session.
cd /home/lmeadows/llm
bash scripts/run/launch_teacher_a.sh
bash scripts/run/launch_teacher_b.sh
bash scripts/run/launch_teacher_c.sh
