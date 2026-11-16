#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python3 scripts/lwy_integrity.py freeze --output metrics/lwy_r7_protected_remote.json
python3 scripts/lwy_integrity.py package --package ../../ATEC_Env
JAX_PLATFORMS=cpu .venv_lwy/bin/python -m nav3.lwy_obstacles_r7 --splits test blind
export XLA_FLAGS='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
export TF_CUDNN_DETERMINISTIC=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
"$NAV3_PYTHON" -m nav3.lwy_compare_r7 --method planner --dataset original --episodes 10 --random-obstacles --tag r7_gate_original > logs/lwy_r7_gate_original.log 2>&1
"$NAV3_PYTHON" -m nav3.lwy_compare_r7 --method planner --dataset blind --episodes 5 --random-obstacles --tag r7_gate_blind > logs/lwy_r7_gate_blind.log 2>&1
