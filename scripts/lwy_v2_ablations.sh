#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
# Single variable relative to the recorded baseline; none is auto-adopted.
for mode in dense lookahead slow no_drift; do
  case "$mode" in
    dense) opts=(--candidates 57);;
    lookahead) opts=(--lookahead .6);;
    slow) opts=(--prediction-speed .28);;
    no_drift) opts=(--no-drift);;
  esac
  timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_probe_v2 --tag "v2_probe_${mode}" --record "${opts[@]}" > "logs/lwy_v2_probe_${mode}.log" 2>&1
done
