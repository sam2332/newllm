# Diagnostic: train M-size agent only on single-step math traces.
# This tests whether the arithmetic failure is a capacity problem.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size M `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --math-only `
    --lr 3e-4 `
    > "agent_run_web_m_${timestamp}_math_only.log" 2>&1

exit $LASTEXITCODE
