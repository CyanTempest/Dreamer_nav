#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export NAV3_PYTHON="$PWD/.venv_lwy/bin/python"
timeout 10500 "$NAV3_PYTHON" -m nav3.lwy_audit
