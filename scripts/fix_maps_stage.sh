#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_fix.py > logs/fix_pre_map_checks.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/fix_diag_checks.json > logs/fix_diag_checks.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.maps_v2 --workers 6 > logs/fix_maps_generation.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.fix_prepare census > logs/fix_census.log 2>&1
"$NAV3_PYTHON" -m nav3.fix_eval --name planner --root maps_v2 --split blind --episodes 5 > logs/fix_g1_eval.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.fix_prepare gate > logs/fix_g1.log 2>&1
"$NAV3_PYTHON" tests/check_fix.py --reward > logs/fix_reward_checks.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.fix_prepare prepare > logs/fix_prepare.log 2>&1
