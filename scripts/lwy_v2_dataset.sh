#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export JAX_PLATFORMS=cpu
timeout 10500 .venv_lwy/bin/python -m nav3.lwy_dataset_v2 --workers 24
