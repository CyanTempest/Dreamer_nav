#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export XLA_FLAGS='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
export TF_CUDNN_DETERMINISTIC=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
nvidia-smi > logs/lwy_r7_gpu_before_main.log
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage train
JAX_PLATFORMS=cpu .venv_lwy/bin/python -m nav3.lwy_r7_report
tmux new-session -d -s nav3_r7_video "cd '$PWD'; source scripts/env.sh; JAX_PLATFORMS=cpu \"\$NAV3_PYTHON\" scripts/lwy_r7_video.py > logs/lwy_r7_video_queue.log 2>&1"
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage ppo
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage final
JAX_PLATFORMS=cpu .venv_lwy/bin/python -m nav3.lwy_r7_report
