#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
while tmux has-session -t nav3_fix_baselines 2>/dev/null; do sleep 10; done
"$NAV3_PYTHON" - <<'PY'
from pathlib import Path
import json
paths=list(Path('metrics').glob('fix_eval_v1_*.json'))
assert len(paths)==24,len(paths)
for p in paths:
    d=json.loads(p.read_text())
    assert d['episodes']==100
    assert d['runtime']['xla_flags']=='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
print('BASELINES_COMPLETE',flush=True)
PY
for tag in default warm level1; do
  flags=()
  if [ "$tag" = warm ]; then flags+=(--warm); fi
  if [ "$tag" = level1 ]; then flags+=(--level 1); fi
  "$NAV3_PYTHON" scripts/fix_numeric_probe.py --tag "$tag" "${flags[@]}" > "logs/fix_numeric_${tag}.log" 2>&1
done
timeout 7200 bash scripts/fix_maps_stage.sh
