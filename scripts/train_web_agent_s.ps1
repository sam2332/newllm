# Train a fresh S-size agent on JSON + web_search traces.
# Writes its own checkpoint dir and log so the previous agent_best.pt stays intact.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 4000 `
    --size S `
    --curriculum `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    > "agent_run_web_s_$timestamp.log" 2>&1
