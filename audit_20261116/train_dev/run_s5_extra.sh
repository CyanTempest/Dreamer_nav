#!/usr/bin/env bash
# S5 throughput/memory, then planner checks through the refactored evaluate.drive:
#  P1 default planner on the holdout windows (compare with nav4_dev/c2_planner_s914901.json, must be identical),
#  P2 planner on trail_a forward with assist, localization.mode random_walk vs odometry (dead reckoning), same map and seed.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
T=audit_20261116/train_dev
uptime > $T/logs/s5_load_before.txt
s=$(date +%s)
$NAV3_PYTHON $T/s5_throughput.py 64 40 256 > $T/s5.json 2> $T/logs/s5.log
echo "S5 EXIT=$? WALL_S=$(( $(date +%s)-s ))"
uptime >> $T/logs/s5_load_before.txt
s=$(date +%s)
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method planner --repeats 8 --seed 914901 \
  --output $T/p1_planner_holdout_s914901.json > $T/logs/p1.log 2>&1
echo "P1 EXIT=$? WALL_S=$(( $(date +%s)-s ))"
$NAV3_PYTHON $T/compare_rows.py audit_20261116/nav4_dev/c2_planner_s914901.json $T/p1_planner_holdout_s914901.json | tee $T/p1_compare.txt
sha256sum data/trail_a/manifest.json data/trail_a/forward/map.npz > $T/p2_map_sha256_before.txt
s=$(date +%s)
for mode in random_walk odometry; do
  $NAV3_PYTHON -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --assist \
    --max-steps 25000 --env "localization.mode=\"$mode\"" --output $T/p2_planner_forward_assist_$mode.json --plot > $T/logs/p2_$mode.log 2>&1 &
done
wait
echo "P2 WALL_S=$(( $(date +%s)-s ))"
sha256sum data/trail_a/manifest.json data/trail_a/forward/map.npz > $T/p2_map_sha256_after.txt
