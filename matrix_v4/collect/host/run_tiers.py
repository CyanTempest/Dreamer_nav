"""One card per invocation. Preserve full wall-time denominators, including resets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
base=Path(os.environ['MATRIX_COLLECTION_DIR']).resolve()
session=os.environ['MV4_SESSION']
name=session.removeprefix('mv4col-')
out=base/'results'/name
out.mkdir(parents=True,exist_ok=True)
mux=['bash',str(Path(__file__).with_name('mux.sh'))]


def vm(cmd):
    return subprocess.check_output([*mux,cmd],text=True,timeout=90)


def main():
    first=int(os.environ.get('FIRST_TIER','1'))
    last=None
    if first>1:
        previous,=out.glob(f'prod{first-1}-{first-1}-*/done.json')
        row=json.loads(previous.read_text())
        last=row['qualified']/(row['end']-row['start'])
    for n in range(first,15):
        if (base/'logs/budget_stop').exists():
            return
        usage=(base/'logs/guard22.log').read_text().splitlines()
        snapshots=[line.split() for line in usage if len(line.split())==4]
        if snapshots:
            balance,rate,spent=map(float,snapshots[-1][1:])
            if spent+rate*(15/60)>=22:
                print('Ending ladder with transfer and shutdown reserve',spent,rate,flush=True)
                break
        label=os.environ.get('TIER_PREFIX','prod')+str(n)
        cmd=f'cd /content/col; nohup python3 -u collect/benchmark.py {n} --tag {label} </dev/null >{label}.log 2>&1 & echo $!'
        pid=vm(cmd).strip().splitlines()[-1]
        if not pid.isdecimal():
            raise RuntimeError(pid)
        print(name,n,'started',time.time(),flush=True)
        while True:
            time.sleep(20)
            if (base/'logs/budget_stop').exists():
                return
            try:
                result=vm(f'cat /content/col/runs/{label}-{n}-*/done.json 2>/dev/null || (kill -0 {pid} 2>/dev/null && echo running || echo failed)')
            except (subprocess.SubprocessError,OSError) as e:
                print(e,flush=True)
                continue
            if result.strip()=='running':
                continue
            if result.strip()=='failed':
                raise RuntimeError(vm(f'tail -30 /content/col/{label}.log'))
            row=json.loads(result)
            break
        # Pull diagnostic text separately, exclude all assets and shard bytes.
        tmp=out/(label+'.tar.gz.tmp')
        with tmp.open('wb') as f:
            subprocess.run([*mux,f'cd /content/col/runs && tar --exclude=ue.log --exclude=ns.pid --exclude=interrupted.partial -czf - {label}-{n}-*'],stdout=f,check=True)
        subprocess.run(['tar','-xzf',str(tmp),'-C',str(out)],check=True)
        tmp.unlink()
        productivity=row['qualified']/(row['end']-row['start'])
        print(name,n,'done',row,flush=True)
        if last is not None and productivity<=last:
            break
        last=productivity
    (out/'tiers_finished').write_text(str(time.time()))


if __name__=='__main__':
    main()
