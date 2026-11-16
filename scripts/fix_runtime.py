"""Record software, driver and relevant source hashes for reproducibility."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

packages={}
for name in ['jax','jaxlib','numpy','scipy','optax','elements','ninjax','mujoco','nvidia-cudnn-cu12','nvidia-cublas-cu12']:
    try:packages[name]=metadata.version(name)
    except metadata.PackageNotFoundError:packages[name]=None
files=['nav3/env.py','nav3/bank.py','nav3/control.py','nav3/ppo.py','nav3/dreamer_adapter.py','nav3/evaluate.py',
    'nav3/fix_eval.py','nav3/fix_training.py','nav3/train_dreamer.py','nav3/train_ppo.py','nav3/maps.py','nav3/maps_v2.py',
    'configs/env.json','configs/runtime.json']
result=dict(python=sys.version,platform=platform.platform(),packages=packages,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,uuid','--format=csv,noheader'],text=True).strip(),
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
with Path('metrics/fix_runtime.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
