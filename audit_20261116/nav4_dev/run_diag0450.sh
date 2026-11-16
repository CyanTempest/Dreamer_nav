#!/usr/bin/env bash
# Diagnostic: planner on segment_0450 only, 16 traverses, default vs noise-free sensing (legacy fitted dropout).
set -uo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
D=audit_20261116/nav4_dev
M=r9_official_runner/maps_scan_holdout
NOISE=(sensor.height_std=[0,0] sensor.dropout_base=[0.03,0.03] sensor.dropout_distance=[0.00025,0.00025])
for s in 914901 20261116; do
  $NAV3_PYTHON -m nav4.evaluate --manifest $M --maps segment_0450 --method planner --repeats 16 --seed $s --output $D/diag_c2_0450_default_s$s.json > $D/logs/diag_c2_0450_default_s$s.log 2>&1
  $NAV3_PYTHON -m nav4.evaluate --manifest $M --maps segment_0450 --method planner --repeats 16 --seed $s --env "${NOISE[@]}" --output $D/diag_c2_0450_nonoise_s$s.json > $D/logs/diag_c2_0450_nonoise_s$s.log 2>&1
done
echo "DIAG0450_DONE $(date -Is)" >> $D/logs/matrix.log
