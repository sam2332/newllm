# Run the promotion gate on the latest web-search candidate.
# If it passes, this copies checkpoints_web/agent_best.pt over the main best.
python -m agent.promote checkpoints_web/agent_best.pt --best checkpoints/agent_best.pt --min-accuracy 0.75 --min-numeric 0.60 --min-exact 0.80
