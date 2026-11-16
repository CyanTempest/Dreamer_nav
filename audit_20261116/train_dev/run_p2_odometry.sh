#!/usr/bin/env bash
# P2 (dead-reckoning variant) rerun with the final env.py: planner, trail_a forward, assist, localization.mode=odometry.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
T=audit_20261116/train_dev
sha256sum data/trail_a/manifest.json data/trail_a/forward/map.npz nav4/env.py > $T/p2_odometry_sha256_before.txt
s=$(date +%s)
$NAV3_PYTHON -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --assist \
  --max-steps 25000 --env "localization.mode=\"odometry\"" --output $T/p2_planner_forward_assist_odometry.json --plot > $T/logs/p2_odometry.log 2>&1
echo "P2_ODOMETRY EXIT=$? WALL_S=$(( $(date +%s)-s ))"
sha256sum data/trail_a/manifest.json data/trail_a/forward/map.npz nav4/env.py > $T/p2_odometry_sha256_after.txt
