# Diagnostic: train S-size agent on compact calc(expr) math traces.
# The expression is a contiguous marked substring of the question.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size S `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --math-only `
    --compact-math `
    --lr 3e-4 `
    > "agent_run_web_s_${timestamp}_math_compact.log" 2>&1

exit $LASTEXITCODE
