#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
nvidia-smi
timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 "$@"
