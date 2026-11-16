#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
for method in ppo_s0 dreamer_s0; do
  for n in 1 2; do
    tag="v2_fixed_protocol_${method}_n${n}"
    timeout 1500 "$NAV3_PYTHON" -m nav3.lwy_compare_v2 --root maps_lwy_v2_qualification --method "$method" --dataset original --episodes "$n" --tag "$tag" > "logs/lwy_${tag}.log" 2>&1
  done
done
"$NAV3_PYTHON" - <<'PY'
import json
from pathlib import Path
rows=[]
for method in ['ppo_s0','dreamer_s0']:
    a=json.loads(Path(f'metrics/lwy_v2_fixed_protocol_{method}_n1.json').read_text())
    b=json.loads(Path(f'metrics/lwy_v2_fixed_protocol_{method}_n2.json').read_text())
    assert a['rows'][0]['action_pose_sha256']==b['rows'][0]['action_pose_sha256'],method
    rows.append(dict(method=method,batch_invariant=True,policy_batch_width=100,hash=a['rows'][0]['action_pose_sha256']))
Path('metrics/lwy_v2_policy_protocol.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
PY
