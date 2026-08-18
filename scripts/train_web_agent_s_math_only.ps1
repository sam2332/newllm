# Diagnostic: train S-size agent only on single-step math traces.
# This tells us whether the 25M model can learn arithmetic copying at all.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size S `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --math-only `
    --lr 3e-4 `
    > "agent_run_web_s_${timestamp}_math_only.log" 2>&1

exit $LASTEXITCODE
