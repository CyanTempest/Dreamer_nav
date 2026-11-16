#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export XLA_FLAGS='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
export TF_CUDNN_DETERMINISTIC=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
while tmux has-session -t nav3_r7_gate 2>/dev/null; do sleep 20; done
nvidia-smi > logs/lwy_r7_gpu_before_diagnosis.log
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage prep
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage diagnosis
JAX_PLATFORMS=cpu .venv_lwy/bin/python -m nav3.lwy_r7_report
