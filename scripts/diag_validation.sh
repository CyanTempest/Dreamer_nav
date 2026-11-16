#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for variant in base rocks; do
  for method in dreamer_s0 ppo_s0 planner; do
    "$NAV3_PYTHON" -m nav3.diag_rollout --method "$method" --tag "synthetic_${variant}_${method}" --root "maps/diag_synthetic_${variant}" --split heldout --episodes 15 --around-log > "logs/diag_synthetic_${variant}_${method}.log" 2>&1
  done
done
for seed in 9137 9141; do
  "$NAV3_PYTHON" -m nav3.diag_rollout --tag "oracle_obstacles_${seed}" --start 150 --seed "$seed" --oracle-obstacles > "logs/diag_oracle_obstacles_${seed}.log" 2>&1
done
"$NAV3_PYTHON" -m nav3.diag_rollout --method ppo_s0 --tag full_ppo_protocol --record > logs/diag_full_ppo_protocol.log 2>&1
"$NAV3_PYTHON" -m nav3.diag_rollout --tag log_center_occupancy_full_v2 --root maps/diag_log_center_occupancy_v2 --record > logs/diag_log_center_occupancy_full_v2.log 2>&1
