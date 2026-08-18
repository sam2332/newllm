Set-Location -Path $PSScriptRoot
$Env:PYTHONUNBUFFERED = "1"
python -u -c "
import torch, random
torch.manual_seed(42)
random.seed(42)
from agent.train_agent import train_agent
train_agent(num_samples=100000, iters=5000)
" *>&1 | Tee-Object -FilePath agent_run_v3b.log
