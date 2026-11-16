"""Read-only final audit; never stop another user's process or session."""
import datetime
import json
import os
from pathlib import Path
import subprocess


root=Path.cwd().resolve()
ancestors=set();pid=os.getpid()
while pid and pid not in ancestors:
    ancestors.add(pid)
    try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
    except (OSError,ValueError,IndexError):break
remaining=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:continue
    try:
        if entry.stat().st_uid!=os.getuid():continue
        cwd=(entry/'cwd').resolve(strict=True)
        if cwd!=root and root not in cwd.parents:continue
        command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        if command:remaining.append(dict(pid=int(entry.name),cwd=str(cwd),command=command))
    except (OSError,RuntimeError):continue
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
sessions=[s for s in sessions if s.startswith('nav3_')]
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nav3_sessions=sessions,
    remaining_project_processes=remaining,
    gpu=subprocess.check_output(['nvidia-smi'],text=True))
assert not sessions and not remaining,result
with Path('metrics/fix_remote_final.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
