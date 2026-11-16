#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
nvidia-smi
"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split test --episodes 10 --tag g1_original_lwy --assert-min-progress 999
"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split test --episodes 10 --tag g1_original_lwy_repeat --assert-min-progress 999
"$NAV3_PYTHON" -m nav3.lwy_planner_check
