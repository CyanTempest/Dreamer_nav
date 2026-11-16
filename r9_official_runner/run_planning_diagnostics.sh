#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
runner=r9_official_runner
cloud=/home/hpf/wang-sm/ATEC_NAV_WSM/research/official_atec/e5a3b987/extracted/01_LiDAR_point_cloud/ds_map/plannar_map.ply
while tmux has-session -t r9_planning_fast 2>/dev/null; do sleep 5; done
test -f "$cloud"
.venv_lwy/bin/python "$runner/build_scan_patches.py" "$cloud" --trajectory "$runner/metadata/trajectory_audit.npz" --output "$runner/maps_planning" --source-sha256 ae8e5cb2cc065529eaa3567e274a9109fab6d38caad3de253ab0b522285a3305
.venv_lwy/bin/python "$runner/qualify_routes.py" "$runner/maps_planning" --output "$runner/maps_planning_qualified"
source scripts/env.sh
export PYTHONPATH="$PWD/$runner:$PYTHONPATH"
checkpoint=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
for variant in baseline body; do
  options=()
  if [[ "$variant" == body ]]; then options+=(--body-bands); fi
  "$NAV3_PYTHON" "$runner/diagnose_policy.py" --maps "$runner/maps_planning" --checkpoint "$checkpoint" --output "$runner/metadata/policy_planning_$variant.json" "${options[@]}"
done
if .venv_lwy/bin/python -c 'import json,sys;sys.exit(not json.load(open("r9_official_runner/maps_planning_qualified/manifest.json"))["maps"])'; then
  for variant in baseline body; do
    options=()
    if [[ "$variant" == body ]]; then options+=(--body-bands); fi
    "$NAV3_PYTHON" "$runner/diagnose_policy.py" --maps "$runner/maps_planning_qualified" --checkpoint "$checkpoint" --output "$runner/metadata/policy_planning_qualified_$variant.json" "${options[@]}"
  done
fi
printf 'R9_PLANNING_DIAGNOSTICS_COMPLETE\n'
