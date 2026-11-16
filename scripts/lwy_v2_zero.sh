#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for method in dreamer_s0 ppo_s0; do
  for dataset in original blind; do
    episodes=100
    if [[ "$dataset" == blind ]]; then episodes=5; fi
    tag="v2_final_${method}_${dataset}_ob0_sf0"
    timeout 3000 "$NAV3_PYTHON" -m nav3.lwy_compare_v2 --root maps_lwy_v2_qualification --method "$method" --dataset "$dataset" --episodes "$episodes" --tag "$tag" > "logs/lwy_${tag}.log" 2>&1
  done
done
timeout 1500 "$NAV3_PYTHON" -m nav3.lwy_video_v2 --probe > logs/lwy_v2_renderer_probe.log 2>&1
