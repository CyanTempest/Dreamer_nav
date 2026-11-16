#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
while tmux has-session -t nav3_diag_validation 2>/dev/null; do sleep 5; done
for variant in base rocks; do
  for method in dreamer_s0 ppo_s0 planner; do
    "$NAV3_PYTHON" -m nav3.diag_rollout --method "$method" --tag "synthetic_local_${variant}_${method}" --root "maps/diag_synthetic_${variant}" --split heldout --episodes 15 --around-log --log-start-offset -6 > "logs/diag_synthetic_local_${variant}_${method}.log" 2>&1
  done
done
"$NAV3_PYTHON" -m nav3.diag_candidate > logs/diag_candidate_probe.log 2>&1
