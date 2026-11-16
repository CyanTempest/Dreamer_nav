#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
export PYTHONPATH="$PWD/r9_official_runner:$PYTHONPATH"
runner=r9_official_runner
for variant in planner dreamer hybrid; do
  options=()
  if [[ "$variant" == planner ]]; then options+=(--method planner)
  else options+=(--checkpoint checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl); fi
  if [[ "$variant" == hybrid ]]; then options+=(--safety-filter --filter-hold-steps 5); fi
  "$NAV3_PYTHON" "$runner/diagnose_policy.py" --maps "$runner/maps_scan_holdout" --body-bands --repeats 2 --steps 1502 --seed 914901 --output "$runner/metadata/holdout_$variant.json" "${options[@]}"
done
printf 'R9_SPATIALLY_SEPARATED_HOLDOUT_COMPLETE\n'
