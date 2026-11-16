#!/usr/bin/env bash
set -uo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
D=audit_20261116/nav4_dev
for s in 914901 20261116; do
  $NAV3_PYTHON $D/diag_legacy_planner.py --manifest r9_official_runner/maps_scan_holdout --maps segment_0450 --method planner --repeats 16 --seed $s --output $D/diag_c2_0450_legacyplanner_s$s.json > $D/logs/diag_c2_0450_legacyplanner_s$s.log 2>&1
done
echo "DIAGLEGACY_DONE $(date -Is)" >> $D/logs/matrix.log
