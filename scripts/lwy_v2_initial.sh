#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python3 scripts/lwy_integrity.py freeze --output metrics/lwy_v2_protected_remote.json
python3 - <<'PY'
import json
from pathlib import Path
from scripts.lwy_integrity import digest
p = Path('metrics/lwy_v2_old_maps.json')
with p.open('x') as f:
    json.dump({str(x): digest(x) for x in Path('maps_lwy').rglob('*') if x.is_file()}, f)
PY
python3 scripts/lwy_integrity.py package --package ../../ATEC_Env
nvidia-smi
timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split blind --episodes 1 --start 970 --tag v2_repro_old --assert-min-progress 998
