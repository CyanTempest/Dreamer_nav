"""Freeze/check round-six protected inputs; stdlib only, including Python 3.9."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def protected():
    files = list(Path('metrics').glob('*.json'))
    for name in ['checkpoints', 'maps', 'maps_v2', 'maps_mix_v2']:
        files += [p for p in Path(name).rglob('*') if p.is_file()]
    files += [Path(p) for p in ['nav3/env.py', 'nav3/bank.py', 'nav3/control.py',
                               'configs/env.json', 'configs/calibration.yaml']]
    return sorted(set(files))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['freeze', 'check', 'package'])
    p.add_argument('--output', default='metrics/lwy_protected_remote.json')
    p.add_argument('--package', type=Path)
    a = p.parse_args()
    if a.mode == 'package':
        # Compatibility implementation of hashlib.file_digest, no package edits.
        if not hasattr(hashlib, 'file_digest'):
            def file_digest(f, name):
                h = hashlib.new(name)
                for block in iter(lambda: f.read(8 << 20), b''):
                    h.update(block)
                return h
            hashlib.file_digest = file_digest
        sys.argv = [str(a.package / 'tools/verify_package.py'), '--root', str(a.package)]
        runpy.run_path(sys.argv[0], run_name='__main__')
        return
    path = Path(a.output)
    if a.mode == 'freeze':
        assert not path.exists(), path
        records = {str(f): digest(f) for f in protected()}
        with path.open('x') as f:
            json.dump(records, f, indent=2)
        print(json.dumps(dict(protected_files=len(records), manifest=str(path))), flush=True)
    else:
        records = json.loads(path.read_text())
        changed = [f for f, h in records.items() if not Path(f).is_file() or digest(f) != h]
        # Existing checkpoint/map trees are read-only, including their caches.
        added = [str(f) for directory in ['maps', 'maps_v2', 'maps_mix_v2']
                 for f in Path(directory).rglob('*') if f.is_file() and str(f) not in records]
        print(json.dumps(dict(protected_files=len(records), changed=changed, added=added)), flush=True)
        assert not changed and not added


if __name__ == '__main__':
    main()
