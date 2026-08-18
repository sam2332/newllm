# Train a JSON + web_search S-size agent with digit-word expansion.
# The byte tokenizer copies word numbers more easily than raw digit strings.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Stage 1: single-step foundation (math as word numbers, memory, date, web facts).
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

# Stage 2: multi-step + web-math fine-tuning at lower LR.
python -u -m agent.train_agent `
    --samples 20000 `
    --iters 2000 `
    --size S `
    --curriculum `
    --checkpoint-dir checkpoints_web `
    --lr 1e-4 `
    > "agent_run_web_s_${timestamp}_stage2.log" 2>&1

exit $LASTEXITCODE
