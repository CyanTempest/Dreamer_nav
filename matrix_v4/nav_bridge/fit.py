"""On the server CPU, select the first validated route placement at/after a requested station."""
import json
import sys
from pathlib import Path
import numpy as np
from nav4.matrix.collision_spawn import prepare, stable
from nav4.matrix.full_scan import route

request = float(sys.argv[1])
base = Path('data/trail_a_mujoco/full_v2')
spec = json.loads((base/'assembly.json').read_text())
_, path = route()
for advance in np.arange(0, 3.001, .1):
    station = min(request+advance,722.)
    row = next(r for r in spec['statics'] if r['station_m'][0]<=station<=r['station_m'][1])
    for tread in [False,True]:
        try:
            pose = prepare(Path(row['window']).stem,station,path,np.array(spec['shared_shift_m']),False,tread)
        except (ValueError,AssertionError,FloatingPointError):
            continue
        if stable(pose['stand']):
            print(json.dumps(dict(pose,station_m=station),default=lambda v:v.tolist()))
            sys.exit()
raise RuntimeError(f'No stable placement within 3 m of {request}')
