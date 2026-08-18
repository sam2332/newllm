# Diagnostic: train S-size agent only on single-digit single-step math traces.
# This tests whether the model can copy short (single-token) numbers.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size S `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --math-only `
    --single-digit-math `
    --lr 3e-4 `
    > "agent_run_web_s_${timestamp}_math_digits.log" 2>&1

exit $LASTEXITCODE
