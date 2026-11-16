#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export JAX_PLATFORMS=cpu
.venv_lwy/bin/python tests/check_diag.py --output metrics/lwy_r7_check_diag.json
.venv_lwy/bin/python tests/check_lwy.py --v2 --protected metrics/lwy_r7_protected_remote.json --output metrics/lwy_r7_check_lwy.json
"$NAV3_PYTHON" tests/check_lwy_r7.py
.venv_lwy/bin/python -m nav3.lwy_r7_obstacle_analysis
