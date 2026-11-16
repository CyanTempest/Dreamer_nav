#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 --root maps_lwy_v2_qualification --tag v2_round0_capture --start 2 --stop 998.5 --max-steps 18002 --tracker-v2 --yaw-gain 4 --capture-file configs/lwy_v2_capture_round0.json --match-full metrics/lwy_v2_round0_blind.json > logs/lwy_v2_round0_capture.log 2>&1
for mode in baseline gain2 horizon30 border1000; do
  opts=();gain=4
  case "$mode" in
    gain2) gain=2;;
    horizon30) opts=(--horizon 30);;
    border1000) opts=(--border-weight 1000);;
  esac
  timeout 600 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 --root maps_lwy_v2_qualification --tag "v2_round0_replay_${mode}" --tracker-v2 --yaw-gain "$gain" --resume-capture metrics/lwy_v2_round0_capture_capture.npz --record "${opts[@]}" > "logs/lwy_v2_round0_replay_${mode}.log" 2>&1
done
