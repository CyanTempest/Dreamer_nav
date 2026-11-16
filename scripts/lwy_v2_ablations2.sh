#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for mode in margin tracker_parity yaw4 yaw6; do
  case "$mode" in
    margin) opts=(--prediction-margin .10);;
    tracker_parity) opts=(--tracker-v2 --yaw-gain 2);;
    yaw4) opts=(--tracker-v2 --yaw-gain 4);;
    yaw6) opts=(--tracker-v2 --yaw-gain 6);;
  esac
  timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 --tag "v2_probe_${mode}" --record "${opts[@]}" > "logs/lwy_v2_probe_${mode}.log" 2>&1
done
