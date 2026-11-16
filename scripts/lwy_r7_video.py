"""Independent CPU video task, started only after validation model selection."""
import json
import os
from pathlib import Path
import subprocess
state=json.loads(Path('metrics/lwy_r7_queue.json').read_text())
env={**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'8'}
command=[os.environ['NAV3_PYTHON'],'-m','nav3.lwy_video_record_r7','--checkpoint',state['best']['checkpoint'],
    '--out','videos/lwy_r7_dreamer']
with Path('logs/lwy_r7_video_cpu.log').open('x') as f:
    result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=10200)
assert result.returncode==0,'CPU video failed; see log'
