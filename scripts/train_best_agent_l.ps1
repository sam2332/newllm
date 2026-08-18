# Train the larger 151M agentic model (takes ~50 min on RTX 5060 Ti)
python -u -m agent.train_agent --samples 100000 --iters 10000 --size L --curriculum --no-resume > agent_run_best_l.log 2>&1
