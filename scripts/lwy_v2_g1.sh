#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
tag="${1:?unique round tag required}"
root="${2:-maps_lwy_v2}"
nvidia-smi
timeout 3000 "$NAV3_PYTHON" -m nav3.lwy_eval_v2 --root "$root" --split test --episodes 10 --tag "${tag}_original"
timeout 3000 "$NAV3_PYTHON" -m nav3.lwy_eval_v2 --root "$root" --split blind --episodes 1 --tag "${tag}_blind"
"$NAV3_PYTHON" - "$tag" "$root" <<'PY'
import json, sys
from pathlib import Path
from nav3.lwy_eval_v2 import protocol_hash
tag=sys.argv[1]
a=json.loads(Path(f'metrics/lwy_{tag}_original.json').read_text())
b=json.loads(Path(f'metrics/lwy_{tag}_blind.json').read_text())
r=dict(passed=a['success_rate']==1 and a['episodes']==10 and b['success_rate']>=.95 and b['episodes']==20,
       original_successes=sum(x['success'] for x in a['rows']),original_episodes=10,
       blind_success_rate=b['success_rate'],blind_episodes=20,
       environment_sha256=protocol_hash(sys.argv[2]),sources=[f'lwy_{tag}_original.json',f'lwy_{tag}_blind.json'],
       planner_parameters=a['planner_parameters'],tracker_parameters=a['tracker_parameters'])
with Path(f'metrics/lwy_{tag}_g1.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True)
if r['passed']:
    certificate=Path('metrics/lwy_v2_g1.json')
    if certificate.exists():
        old=json.loads(certificate.read_text())
        assert old['passed'] and old['environment_sha256']==r['environment_sha256']
    else:
        with certificate.open('x') as f:json.dump(r,f,indent=2)
PY
