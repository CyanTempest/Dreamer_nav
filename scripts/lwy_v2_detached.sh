#!/usr/bin/env bash
# Launch with a new session and redirected stdio; no client or tmux is required.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export PYTHONUNBUFFERED=1
exec "$NAV3_PYTHON" -m nav3.lwy_queue_v2 --stage all
