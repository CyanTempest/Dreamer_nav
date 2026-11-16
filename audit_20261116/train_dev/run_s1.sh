#!/usr/bin/env bash
# S1: every nav4 unit test on CPU (core + new + the mapping group), one log per group.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
L=audit_20261116/train_dev/logs
core='nav4.tests.test_bank_loader nav4.tests.test_env_localization nav4.tests.test_env_noise nav4.tests.test_env_obstacles nav4.tests.test_env_route_inputs nav4.tests.test_env_support nav4.tests.test_env_task nav4.tests.test_planner_hybrid nav4.tests.test_tracker_consistency'
new='nav4.tests.test_env_odometry nav4.tests.test_env_priv nav4.tests.test_train_dreamer'
for group in core new; do
  s=$(date +%s)
  $NAV3_PYTHON -m unittest -v ${!group} > $L/s1_$group.log 2>&1
  echo "S1_$group EXIT=$? WALL_S=$(( $(date +%s)-s ))" | tee -a $L/s1_$group.log
done
s=$(date +%s)
$NAV3_PYTHON -m unittest discover -v -s nav4/tests -p 'test_mapping_*.py' > $L/s1_mapping.log 2>&1
echo "S1_mapping EXIT=$? WALL_S=$(( $(date +%s)-s ))" | tee -a $L/s1_mapping.log
s=$(date +%s)
$NAV3_PYTHON -m unittest discover -s nav4/tests -p 'test_*.py' > $L/s1_discover_all.log 2>&1
echo "S1_discover_all EXIT=$? WALL_S=$(( $(date +%s)-s ))" | tee -a $L/s1_discover_all.log
