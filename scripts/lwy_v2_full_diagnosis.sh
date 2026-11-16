#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for mode in baseline no_drift yaw4; do
  opts=()
  if [ "$mode" = no_drift ]; then opts=(--no-drift); fi
  if [ "$mode" = yaw4 ]; then opts=(--tracker-v2 --yaw-gain 4); fi
  timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 --tag "v2_full_${mode}" --start 2 --max-steps 18002 --record-start 970 --record "${opts[@]}" > "logs/lwy_v2_full_${mode}.log" 2>&1
  timeout 300 .venv_lwy/bin/python -m nav3.lwy_v2_trace_analysis --tag "v2_full_${mode}" > "logs/lwy_v2_full_${mode}_analysis.log" 2>&1
done
