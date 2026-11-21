"""Check local assets against the frozen v015 checkpoint and scene."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'third_party/runtimes/MATRiX_Robot_MC-v0.6.6'
DREAMER = ROOT / 'third_party/dreamerv3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='store_true', help='Also check the 3DGS splats used for recording')
    options = parser.parse_args()

    lock = json.loads((ROOT / 'evidence/input_sha256.json').read_text())
    for relative, expected in (lock['checkpoint'] | lock['external_files']).items():
        if relative.endswith('/splats.npz') and not options.video:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f'Missing {relative}. See README.md for the local asset layout.')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise SystemExit(f'SHA-256 mismatch for {relative}')

    for path in (
        RUNTIME / 'build/export/mc/bin/mc_ctrl',
        RUNTIME / 'local_deps/usr/include/boost/version.hpp',
        ROOT / 'build/matrix/libatec_matrix_shm.so',
    ):
        if not path.is_file():
            raise SystemExit(f'Missing {path.relative_to(ROOT)}. See README.md for MATRiX setup.')
    if not (DREAMER / '.git').exists():
        raise SystemExit('Missing DreamerV3 checkout at third_party/dreamerv3.')
    revision = subprocess.check_output(['git', '-C', str(DREAMER), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['dreamerv3_commit']:
        raise SystemExit(f'Expected DreamerV3 {lock["dreamerv3_commit"]}, got {revision}')
    print('v015 checkpoint, local scene, M1 model and runtime prerequisites verified')


if __name__ == '__main__':
    main()
