# Two-stage curriculum for a JSON + web_search S-size agent.
# Stage 1 trains on single-step traces only (math, memory, date, web facts).
# Stage 2 resumes the same checkpoint and mixes in multi-hop + web-math traces.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# --- Stage 1: single-step foundation -----------------------------------------
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 3000 `
    --size S `
    --no-resume `
    --checkpoint-dir checkpoints_web `
    --lr 3e-4 `
    > "agent_run_web_s_${timestamp}_stage1.log" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage 1 failed with exit code $LASTEXITCODE"
    exit 1
}

# --- Stage 2: multi-step + web-math fine-tuning ------------------------------
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 2000 `
    --size S `
    --curriculum `
    --checkpoint-dir checkpoints_web `
    --lr 1e-4 `
    > "agent_run_web_s_${timestamp}_stage2.log" 2>&1

exit $LASTEXITCODE
