"""Record cleanup only after all nav3 tmux jobs and our workers have exited."""
import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    tmux = subprocess.run(['tmux', 'list-sessions', '-F', '#S'], capture_output=True, text=True)
    sessions = [x for x in tmux.stdout.splitlines() if x.startswith('nav3_')]
    processes = []
    for line in subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3: continue
        pid, command, args = parts
        if int(pid) == os.getpid(): continue
        tokens = args.split()
        worker = command.startswith('python') and ('-m nav3.lwy_' in args or 'tests/check_lwy.py' in args or 'tests/check_diag.py' in args)
        script = command in ['bash', 'sh'] and len(tokens) > 1 and tokens[1].startswith('scripts/lwy_')
        if worker or script: processes.append(dict(pid=int(pid), command=command, args=args))
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), nav3_tmux_sessions=sessions,
        our_processes=processes, gpu_compute_processes=gpu.splitlines() if gpu else [],
        tmux_raw=tmux.stdout+tmux.stderr, training_started=False)
    print(json.dumps(result), flush=True)
    assert not sessions and not processes, 'Wait for owned jobs to exit before closing the task'
    with Path('metrics/lwy_remote_final.json').open('x') as f: json.dump(result, f, indent=2)


if __name__ == '__main__': main()
