#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
nvidia-smi
timeout 10500 "$NAV3_PYTHON" -m nav3.lwy_profile --mode legacy --envs 1024
timeout 10500 "$NAV3_PYTHON" -m nav3.lwy_profile --mode lwy --envs 1024
timeout 10500 "$NAV3_PYTHON" -m nav3.lwy_observations
