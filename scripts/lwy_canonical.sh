#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
while [[ ! -f metrics/lwy_split_manifest.json ]]; do sleep 5; done
export NAV3_PYTHON="$PWD/.venv_lwy/bin/python"
"$NAV3_PYTHON" -m nav3.lwy_canonicalize
