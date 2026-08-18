# Train the current best small agentic model (25M params, curriculum, multi-step)
python -u -m agent.train_agent --samples 20000 --iters 4000 --size S --curriculum --no-resume > agent_run_best_s.log 2>&1
