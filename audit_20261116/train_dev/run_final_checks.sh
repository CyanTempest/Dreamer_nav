#!/usr/bin/env bash
# Final-code reruns: S2 (legacy dreamer, holdout) and P1 (default planner, holdout) vs nav4_dev, then the S5 breakdown.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
T=audit_20261116/train_dev
while tmux list-windows -t nav4_train 2>/dev/null | grep -q "s1f"; do
  grep -q S1_discover_all $T/logs/s1_final.log 2>/dev/null && break; sleep 20; done
sha256sum nav4/env.py nav4/policy.py nav4/evaluate.py nav4/config/env.json > $T/final_checks_sha256.txt
s=$(date +%s)
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method dreamer --action mode \
  --checkpoint checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl --repeats 8 --seed 914901 \
  --output $T/s2_final.json > $T/logs/s2_final.log 2>&1
echo "S2_FINAL EXIT=$? WALL_S=$(( $(date +%s)-s ))"
$NAV3_PYTHON $T/compare_rows.py audit_20261116/nav4_dev/c3_dreamer_mode_s914901.json $T/s2_final.json | tee $T/s2_final_compare.txt
s=$(date +%s)
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method planner --repeats 8 --seed 914901 \
  --output $T/p1_final.json > $T/logs/p1_final.log 2>&1
echo "P1_FINAL EXIT=$? WALL_S=$(( $(date +%s)-s ))"
$NAV3_PYTHON $T/compare_rows.py audit_20261116/nav4_dev/c2_planner_s914901.json $T/p1_final.json | tee $T/p1_final_compare.txt
while pgrep -f "p2_planner_forward_assist" > /dev/null; do sleep 20; done
uptime > $T/logs/s5_breakdown_load.txt
$NAV3_PYTHON $T/s5_breakdown.py 64 30 > $T/s5_breakdown.json 2> $T/logs/s5_breakdown.log
echo "S5_BREAKDOWN EXIT=$?"
uptime >> $T/logs/s5_breakdown_load.txt
