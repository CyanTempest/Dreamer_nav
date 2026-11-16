#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export XLA_FLAGS='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
export TF_CUDNN_DETERMINISTIC=1
"$NAV3_PYTHON" -m nav3.lwy_queue_v2 --released --stage all
python3 scripts/lwy_runtime2_acceptance.py
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/lwy_runtime2_check_diag_final.json
JAX_PLATFORMS=cpu .venv_lwy/bin/python tests/check_lwy.py --v2 --protected metrics/lwy_runtime2_protected_remote.json --output metrics/lwy_runtime2_checks_final.json
JAX_PLATFORMS=cpu .venv_lwy/bin/python -m nav3.lwy_report_runtime2
