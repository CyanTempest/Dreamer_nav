#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for ((i=0;i<120;i++)); do
  if [[ -f maps_lwy_v2/manifest.json ]] && ! tmux has-session -t nav3_lwy_v2_obstacles2 2>/dev/null; then break; fi
  sleep 15
done
[[ -f maps_lwy_v2/manifest.json ]]
! tmux has-session -t nav3_lwy_v2_obstacles2 2>/dev/null
timeout 10500 .venv_lwy/bin/python -m nav3.lwy_obstacles_v2 --workers 16 > logs/lwy_v2_obstacles_final.log 2>&1
timeout 10500 "$NAV3_PYTHON" -m nav3.lwy_queue_v2 --stage profiles > logs/lwy_v2_profiles.log 2>&1
