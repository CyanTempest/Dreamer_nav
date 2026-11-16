#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export PYTHONUNBUFFERED=1
for ((i=0;i<120;i++)); do
  if ! tmux has-session -t nav3_lwy_v2_preflight_final 2>/dev/null; then break; fi
  sleep 5
done
! tmux has-session -t nav3_lwy_v2_preflight_final 2>/dev/null
test -f metrics/lwy_v2_memory_selection.json
timeout 600 "$NAV3_PYTHON" -m nav3.lwy_queue_v2 --stage parallel-check
timeout 2200 "$NAV3_PYTHON" -m nav3.lwy_queue_v2 --stage smoke
