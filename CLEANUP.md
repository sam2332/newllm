# Project Cleanup Notes

## Active artifacts (kept in root / working dirs)

- `checkpoints/agent_best.pt` — current best agent checkpoint (25M S-size, curriculum, multi-step).
- `checkpoints/agent_s_multi_4k_v2.pt` — source snapshot for `agent_best.pt`.
- `results/agent_20260817_230527.json` — v3d L single-step result.
- `results/agent_20260817_235516.json` — v3c S single-step result.
- `agent_run_v3f.log` — active L-size curriculum training log.

## Archived artifacts (`archive/`)

- `archive/checkpoints/` — older model checkpoints.
- `archive/results/` — older experiment JSON files.
- `archive/logs/` — older training logs.

## Helper scripts (`scripts/`)

- `scripts/train_best_agent_s.ps1` — train best small agent (25M, ~13 min).
- `scripts/train_best_agent_l.ps1` — train best large agent (151M, ~50 min).
- `scripts/watch_agent.ps1` — tail latest log.
- `scripts/chat_agent.ps1` — interactive chat.
- `scripts/test_agent.ps1` — run tool-use test battery.

## `.gitignore` change

`results/` was removed from `.gitignore` so experimental records are preserved in git. Logs, pyc files, and the `.venv/` folder remain ignored.
