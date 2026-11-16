#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
nvidia-smi
if [[ ! -f metrics/lwy_g1_original_dynamic.json ]]; then
  "$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --episodes 10 --tag g1_original_dynamic --assert-min-progress 999
fi
if [[ ! -f metrics/lwy_g1_original_dynamic_repeat.json ]]; then
  "$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --episodes 10 --tag g1_original_dynamic_repeat --assert-min-progress 999
fi
if [[ ! -f metrics/lwy_planner_fix_dynamic.json ]]; then "$NAV3_PYTHON" -m nav3.lwy_planner_check --dynamic; fi
if [[ ! -f metrics/lwy_g1_blind_dynamic.json ]]; then
  "$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split blind --episodes 1 --tag g1_blind_dynamic
fi
if [[ ! -f metrics/lwy_g1.json ]]; then "$NAV3_PYTHON" -m nav3.lwy_finish; fi
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/lwy_check_diag.json
if [[ ! -f metrics/lwy_checks.json ]]; then JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_lwy.py; fi
export NAV3_PYTHON="$PWD/.venv_lwy/bin/python"
if [[ ! -f metrics/lwy_endpoint_geometry.json ]]; then "$NAV3_PYTHON" -m nav3.lwy_endpoints; fi
"$NAV3_PYTHON" -m nav3.lwy_plots
"$NAV3_PYTHON" -m nav3.lwy_report
echo LWY_FINAL_QUEUE_COMPLETE
