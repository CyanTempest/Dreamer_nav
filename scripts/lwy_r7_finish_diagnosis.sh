#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
while tmux has-session -t '=nav3_r7_diagnosis' 2>/dev/null; do sleep 20; done
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage diagnosis
.venv_lwy/bin/python -m nav3.lwy_r7_diagnosis_analysis
"$NAV3_PYTHON" -m nav3.lwy_r7_queue --stage failures
.venv_lwy/bin/python -m nav3.lwy_r7_failure_analysis
.venv_lwy/bin/python -m nav3.lwy_r7_report
