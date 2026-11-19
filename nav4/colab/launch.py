#!/usr/bin/env python3
"""Server entry point for one budget-limited temporary Colab A100 training job."""
import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parent / 'nav4_colab'
CLI = Path.home() / '.local/bin/colab'
SESSION = 'nav4a100-dev'
SSH = [HERE / 'vm-ssh.sh', SESSION]
RATE = 8.88
TARGET = 13


def usage():
    out = subprocess.check_output([CLI, 'usage'], text=True, timeout=45)
    print(out, flush=True)
    return (float(re.search(r'Current balance:\s*([\d.]+)', out)[1]),
            float(re.search(r'Usage rate:\s*([\d.]+)', out)[1]),
            int(re.search(r'Active assignments:\s*(\d+)', out)[1]))


def train_and_sync(a, run, remaining, map_dir='/content/nav4-map', weights=None):
    """Start the VM entry point once, poll it, and keep the server copies current."""
    logdir = f'/content/runs/{a.name}'
    output = WORK / 'runs' / a.name
    args = ['bash', '/content/nav/nav4/colab/train.sh', '--map', map_dir, '--logdir', logdir,
            '--seconds', str(a.seconds), '--envs', str(a.envs), '--model', a.model,
            '--batch', str(a.batch), '--seed', str(a.seed), '--eval-minutes', str(a.eval_minutes)]
    if a.save_minutes is not None:
        args += ['--save-minutes', str(a.save_minutes)]
    if a.fast_a100:
        args += ['--fast-a100']
    if weights:
        args += ['--weights', weights]
    body = shlex.join(args) + f'; echo $? > {logdir}/exitcode'
    run(SSH + [f'mkdir -p {logdir} && test ! -e {logdir}/driver.log || exit 1; '
               f'nohup bash -c {shlex.quote(body)} > {logdir}/driver.log 2>&1 < /dev/null &'])
    next_sync = time.monotonic() + 600
    closing_sync = False
    while True:
        try:
            status = run(SSH + [f'cat {logdir}/exitcode 2>/dev/null || echo running'],
                         max_seconds=60, capture_output=True, text=True).stdout.strip()
        except subprocess.CalledProcessError as error:
            if remaining() < 30:
                raise
            print(f'status unavailable ({error.returncode}), retrying', flush=True)
            time.sleep(10)
            continue
        final = status != 'running'
        budget_sync = not closing_sync and remaining() <= 240
        if final or budget_sync or time.monotonic() >= next_sync:
            for retry in range(3):
                try:
                    run([HERE / 'sync_to_server.sh', a.name, '--once', SESSION])
                    break
                except subprocess.CalledProcessError:
                    if retry == 2:
                        raise
            print(f'{time.strftime("%FT%T")} synced {a.name}', flush=True)
            next_sync += 600
            closing_sync = closing_sync or budget_sync
        if final:
            if status != '0' or not (output / 'latest.pkl').is_file():
                raise RuntimeError(f'trainer exit {status}, inspect {output}')
            hashes = run(SSH + [f'cd {logdir} && sha256sum latest.pkl && '
                                'if test -e best.json; then sha256sum best.pkl best.json; fi'],
                         capture_output=True, text=True).stdout
            run(['sha256sum', '-c', '-'], text=True, input=hashes, cwd=output)
            (output / 'weights.sha256').write_text(hashes)
            return
        if remaining() <= 90:
            raise TimeoutError('budget cutoff: latest completed checkpoint synchronized')
        time.sleep(25)


def main():
    defaults = json.loads((REPO/'nav4/config/train.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--envs', type=int, default=defaults['envs'])
    p.add_argument('--batch', type=int, default=defaults['batch_size'])
    p.add_argument('--model', default=defaults['model'])
    p.add_argument('--seed', type=int, default=defaults['seed'])
    p.add_argument('--save-minutes', type=float, help='override the interval between full checkpoints')
    p.add_argument('--fast-a100', action='store_true', help='use the dedicated A100 training settings')
    p.add_argument('--eval-minutes', type=float)
    p.add_argument('--allocated-at', type=datetime.fromisoformat,
                   help='reuse the single named A100 and count its budget from this timezone-aware ISO timestamp')
    p.add_argument('--weights', type=Path, help='matching best.pkl or latest.pkl for a fresh replay')
    a = p.parse_args()
    if a.eval_minutes is None:
        a.eval_minutes = 60 if a.fast_a100 else defaults['eval']['minutes']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', a.name):
        p.error('invalid run name')
    if not re.fullmatch(r'size\d+m', a.model) or min(a.seconds, a.envs, a.batch) <= 0:
        p.error('invalid training dimensions')
    if a.save_minutes is not None and (not math.isfinite(a.save_minutes) or a.save_minutes < 0):
        p.error('invalid checkpoint interval')
    if not math.isfinite(a.eval_minutes) or a.eval_minutes < 0:
        p.error('invalid evaluation interval')
    out = WORK / 'runs' / a.name
    if out.exists():
        p.error(f'run already exists: {out}')
    if a.weights and not a.weights.is_file():
        p.error('missing --weights file')
    manifest = a.map / 'manifest.json'
    record, = [r for r in json.loads(manifest.read_text())['maps'] if r['name'] == 'forward']
    rel = Path(record.get('path', 'forward/map.npz'))
    if rel.is_absolute() or '..' in rel.parts or not (a.map / rel).is_file():
        p.error(f'invalid forward map path {rel}')
    WORK.mkdir(exist_ok=True)
    with (WORK/'launch.lock').open('a') as lock, tempfile.TemporaryDirectory(dir=WORK, prefix='stage-') as scratch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = Path(scratch)
        with tarfile.open(stage / 'nav4-map.tar', 'w', dereference=True) as tar:
            tar.add(manifest, 'manifest.json')
            tar.add(a.map / rel, str(rel))
        with tarfile.open(stage / 'nav4-code.tar', 'w') as tar:
            for name in ['nav4/train_dreamer.py', 'nav4/policy.py', 'nav4/config/train.json',
                         'nav4/colab/train.sh', 'nav4/colab/requirements.txt']:
                tar.add(REPO / name, name)
        sessions = subprocess.check_output([CLI, 'sessions'], text=True, stderr=subprocess.STDOUT, timeout=45)
        balance, rate, active = usage()
        age = 0
        if a.allocated_at:
            if a.allocated_at.tzinfo is None:
                p.error('--allocated-at requires a timezone')
            age = time.time() - a.allocated_at.timestamp()
            if (active != 1 or f'[{SESSION}]' not in sessions or 'Hardware: A100' not in sessions
                    or 'Shape: High-RAM' not in sessions or rate <= 0 or age < 0 or age*rate/3600 >= TARGET):
                p.error('expected the active high-memory A100 inside its budget')
        elif active or 'No active sessions' not in sessions:
            p.error('an existing Colab assignment is active')
        if balance < TARGET:
            p.error('insufficient balance')
        out.mkdir(parents=True)
        (out / 'usage-before.txt').write_text(f'{balance}\n')
        begin = time.monotonic() - age
        deadline = begin + TARGET * 3600 / RATE - 30

        def run(args, max_seconds=None, **kwargs):
            remaining = int(deadline - time.monotonic())
            if max_seconds is not None:
                remaining = min(remaining, max_seconds)
            if remaining <= 0:
                raise TimeoutError('budget deadline reached')
            return subprocess.run(['timeout', '-k', '5', str(remaining), *args], check=True, **kwargs)

        def upload(src, dst):
            for attempt in range(3):
                try:
                    run(['rsync', '-rt', '--partial', '--timeout=180', '-e', HERE / 'vm-ssh.sh',
                         src, f'{SESSION}:/content/{dst}'])
                    return
                except subprocess.CalledProcessError:
                    if attempt == 2:
                        raise

        try:
            if not a.allocated_at:
                run([CLI, 'new', '-s', SESSION, '--gpu', 'A100', '--high-mem'])
                _, rate, assignments = usage()
                if rate <= 0 or assignments != 1:
                    raise RuntimeError('unexpected Colab usage after allocation')
            deadline = begin + TARGET * 3600 / rate - 30
            upload(stage / 'nav4-code.tar', 'nav4-code.tar')
            upload(stage / 'nav4-map.tar', 'nav4-map.tar')
            upload(HERE / 'setup.sh', 'nav4-setup.sh')
            if a.weights:
                upload(a.weights, 'nav4-init.pkl')
            run(SSH + ['bash /content/nav4-setup.sh > /content/setup.log 2>&1 || { tail -60 /content/setup.log; exit 1; }'])
            run([CLI, 'download', '-s', SESSION, 'content/setup.log', out / 'setup.log'])
            train_and_sync(a, run, lambda: deadline - time.monotonic(),
                           weights='/content/nav4-init.pkl' if a.weights else None)
        finally:
            for attempt in range(3):
                try:
                    subprocess.run([CLI, 'stop', '-s', SESSION], timeout=60)
                except subprocess.TimeoutExpired:
                    pass
                status = subprocess.check_output([CLI, 'sessions'], text=True, stderr=subprocess.STDOUT, timeout=45)
                if SESSION not in status and 'No active sessions' in status:
                    break
                if attempt == 2:
                    raise RuntimeError(f'check session release: {status}')
            print(status)
            final, rate, _ = usage()
            (out / 'usage-after.txt').write_text(f'balance={final} difference={balance-final:.2f} rate={rate}\n')


if __name__ == '__main__':
    main()

