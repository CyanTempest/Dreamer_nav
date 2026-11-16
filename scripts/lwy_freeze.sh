#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
timeout 10500 "$NAV3_PYTHON" scripts/lwy_integrity.py freeze
"$NAV3_PYTHON" ../../ATEC_Env/tools/verify_package.py
