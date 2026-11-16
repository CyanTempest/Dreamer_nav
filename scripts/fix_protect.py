"""Snapshot/check every pre-existing protected file without changing it."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    out = Path('metrics/fix_protected_hashes.json')
    if a.check:
        records = json.loads(out.read_text())
        changed = [name for name, h in records.items() if not Path(name).exists() or digest(Path(name)) != h]
        added = [str(f) for directory in ['maps', 'checkpoints/dreamer_s0', 'checkpoints/ppo_s0']
                 for f in Path(directory).rglob('*') if f.is_file() and str(f) not in records]
        assert not changed, changed
        assert not added, added
        print(json.dumps(dict(protected_files=len(records), changed=changed, added=added)), flush=True)
    else:
        assert not out.exists(), out
        files = list(Path('metrics').glob('*.json'))
        for directory in ['maps', 'checkpoints/dreamer_s0', 'checkpoints/ppo_s0']:
            files += [f for f in Path(directory).rglob('*') if f.is_file()]
        files += [Path(x) for x in ['nav3/env.py', 'nav3/bank.py', 'nav3/control.py', 'configs/env.json', 'configs/calibration.yaml']]
        records = {str(f): digest(f) for f in sorted(set(files))}
        with out.open('x') as f:
            json.dump(records, f, indent=2)
        print(json.dumps(dict(protected_files=len(records))), flush=True)
