#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
deadline=$((SECONDS+9000))
while [ ! -f maps_lwy_v2/manifest.json ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo 'Dataset deadline exceeded; no G1 or training started.'
    exit 1
  fi
  sleep 10
done
exec bash scripts/lwy_v2_g1.sh v2_round0
