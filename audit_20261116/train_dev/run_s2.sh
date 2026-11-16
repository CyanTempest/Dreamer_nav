#!/usr/bin/env bash
# S2: legacy R8-P2 checkpoint through the modified nav4 (default env config), compared with nav4_dev/c3.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
start=$(date +%s)
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method dreamer --action mode \
  --checkpoint checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl --repeats 8 --seed 914901 \
  --output audit_20261116/train_dev/s2.json
echo "S2_EXIT=$? WALL_S=$(( $(date +%s)-start ))"
