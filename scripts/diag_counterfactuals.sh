#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for seed in 9137 9141; do
  "$NAV3_PYTHON" -m nav3.diag_rollout --tag "start150_${seed}" --start 150 --seed "$seed" --record > "logs/diag_start150_${seed}.log" 2>&1
  for variant in remove_log remove_opposing_rocks shift_log_left_030 rotate_log_30; do
    "$NAV3_PYTHON" -m nav3.diag_rollout --tag "${variant}_${seed}" --root "maps/diag_${variant}" --start 150 --seed "$seed" > "logs/diag_${variant}_${seed}.log" 2>&1
  done
done
for method in ppo_s0 planner; do
  "$NAV3_PYTHON" -m nav3.diag_rollout --method "$method" --tag "start150_${method}" --start 150 --record > "logs/diag_start150_${method}.log" 2>&1
done
