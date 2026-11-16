#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python3 scripts/lwy_integrity.py freeze --output metrics/lwy_runtime2_protected_remote.json
python3 - <<'PY'
import json,subprocess,time,os
from pathlib import Path
rows=[]
for n in [128,640]:
    tag=f'lwy_runtime2_memory_ppo_{n}'
    args=[os.environ['NAV3_PYTHON'],'-m','nav3.lwy_memory_v2','--method','ppo','--envs',str(n),'--output',f'metrics/{tag}.json','--live-output','metrics/lwy_runtime2_memory_live.json']
    start=time.monotonic()
    with open(f'logs/{tag}.log','x') as f:
        result=subprocess.run(['timeout','-k','30','600',*args],stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(envs=n,returncode=result.returncode,wall_seconds=time.monotonic()-start,log=f'logs/{tag}.log'))
    Path('metrics/lwy_runtime2_preflight.json').write_text(json.dumps(dict(jobs=rows,wall_seconds=sum(r['wall_seconds'] for r in rows)),indent=2))
    if n==128 and result.returncode:raise SystemExit(result.returncode)
PY
