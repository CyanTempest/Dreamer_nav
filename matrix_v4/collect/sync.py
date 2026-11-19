"""Poll sealed episode manifests through an SSH-compatible command. Resume partial pulls.
Usage: python sync.py DEST --transport 'bash /path/to/ops/mux.sh' [--watch]
"""
import argparse
import fcntl
from pathlib import Path
import re
import shlex
import subprocess
import time
from shard import digest


def pull(dest, transport):
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.pull.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=subprocess.check_output([*transport,
            'cd /content/col/data && cat -- *.sha256 2>/dev/null || true'],text=True)
        for line in manifest.splitlines():
            hit=re.fullmatch(r'([a-f0-9]{64})  ([a-zA-Z0-9_-]+\.npz)',line)
            if not hit:
                raise ValueError('Invalid VM manifest')
            expected,name=hit.groups()
            path=dest/name
            marker=path.with_suffix('.sha256')
            if path.exists() and marker.exists() and marker.read_text().strip()==line:
                continue
            if path.exists() and digest(path)==expected:
                marker.write_text(line+'\n')
                continue
            partial=path.with_suffix('.partial')
            offset=partial.stat().st_size if partial.exists() else 0
            with partial.open('ab') as out:
                subprocess.run([*transport,f'tail -c +{offset+1} /content/col/data/{name}'],stdout=out,check=True)
            if digest(partial)!=expected:
                partial.unlink()
                raise IOError('SHA mismatch: '+name)
            partial.replace(path)
            marker.write_text(line+'\n')
            print('verified',name,path.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('dest',type=Path)
    p.add_argument('--transport',required=True)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        try:
            pull(a.dest,shlex.split(a.transport))
        except (OSError,subprocess.CalledProcessError) as e:
            if not a.watch:
                raise
            print(e,flush=True)
        if not a.watch:
            break
        time.sleep(15)
