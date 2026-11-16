"""Extract actual normalization state, rather than treating normalized returns as scales."""
import json
import pickle
from pathlib import Path
import numpy as np
rows=[]
for h in [3.,3.5,4.]:
    p=Path(f'checkpoints/snapshots/dreamer_lwy_runtime2_{h:.2f}h/latest.pkl')
    params=pickle.load(p.open('rb'))['agent']['params']
    values={k:float(np.asarray(v)) for k,v in params.items() if k.startswith('retnorm/')}
    rows.append(dict(hours=h,values=values,scale=max(1.,values['retnorm/hi/value']-values['retnorm/lo/value'])))
with Path('metrics/lwy_r7_return_scales.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows))
