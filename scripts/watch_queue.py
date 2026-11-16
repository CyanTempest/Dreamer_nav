"""Read-only one-minute telemetry for this experiment's queue processes."""
import json
from pathlib import Path
import subprocess
import time

root=Path.cwd().resolve()
while True:
    processes=[];runner=False
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if (path/'cwd').resolve()!=root:continue
            argv=(path/'cmdline').read_bytes().split(b'\0')
            if b'-m' not in argv:continue
            module=argv[argv.index(b'-m')+1].decode()
            if module=='nav3.runner':runner=True
            if module not in ('nav3.runner','nav3.train_dreamer','nav3.train_ppo','nav3.eval_model'):continue
            command=b' '.join(argv).decode(errors='replace')
            status={line.split(':')[0]:line.split(':')[1].strip() for line in (path/'status').read_text().splitlines() if ':' in line}
            processes.append(dict(pid=int(path.name),command=command,rss_kb=int(status.get('VmRSS','0 kB').split()[0]),swap_kb=int(status.get('VmSwap','0 kB').split()[0])))
        except (OSError,PermissionError):continue
    memory={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    row=dict(time=time.time(),processes=processes,gpu=gpu,host_available_kb=memory['MemAvailable'])
    with Path('metrics/queue_telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if not runner:break
    time.sleep(60)
