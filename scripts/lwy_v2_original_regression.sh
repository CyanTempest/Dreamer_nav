#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python3 - <<'PY'
import json
from pathlib import Path
root=Path('maps_lwy_v2_test_view');root.mkdir(exist_ok=True)
link=root/'test_862428'
if not link.exists():link.symlink_to('../maps_lwy_v2/test_862428',target_is_directory=True)
record=json.loads((link/'metadata.json').read_text())
with (root/'manifest.json').open('x') as f:json.dump(dict(maps=[record]),f)
PY
for gain in 2 4; do
  timeout 1200 "$NAV3_PYTHON" -m nav3.lwy_eval_v2 --root maps_lwy_v2_test_view --split test --episodes 10 --tracker-gain "$gain" --tag "v2_original_gain${gain}" > "logs/lwy_v2_original_gain${gain}.log" 2>&1
done
