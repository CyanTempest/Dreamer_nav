#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
export PYTHONPATH="$PWD/r9_official_runner:$PYTHONPATH"
runner=r9_official_runner
checkpoint=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
for scope in raw qualified; do
  if [[ "$scope" == raw ]]; then
    maps="$runner/maps_planning_crop"; repeats=2
  else
    maps="$runner/maps_planning_crop_qualified"; repeats=8
  fi
  for variant in baseline body; do
    options=()
    if [[ "$variant" == body ]]; then options+=(--body-bands); fi
    "$NAV3_PYTHON" "$runner/diagnose_policy.py" --maps "$maps" --checkpoint "$checkpoint" --repeats "$repeats" --output "$runner/metadata/policy_dense_${scope}_$variant.json" "${options[@]}"
  done
done
printf 'R9_DENSE_CROP_POLICY_MATRIX_COMPLETE\n'
