# Diagnostic: train S-size agent on math traces that reprint the expression
# right before the Action: slot. This tests whether short-distance copying
# fixes the arithmetic grounding problem.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size S `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --math-only `
    --lr 3e-4 `
    > "agent_run_web_s_${timestamp}_math_reprint.log" 2>&1

exit $LASTEXITCODE
