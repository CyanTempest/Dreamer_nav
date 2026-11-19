"""Parallel resumable HTTP Range pulls from the authenticated Jupyter files endpoint."""
import concurrent.futures
import fcntl
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import requests
from colab_cli.common import state

base=Path(os.environ['MATRIX_COLLECTION_DIR']).resolve()
dest=base/'data'
session=os.environ['MV4_SESSION']
block=8<<20


def pull_one(s,name,expected):
    partial=dest/(name+'.parts')
    partial.mkdir(exist_ok=True)
    url=s.url.rstrip('/')+'/files/content/col/data/'+name
    params={'authuser':'0','colab-runtime-proxy-token':s.token}
    def get(a,b,size=None):
        for attempt in range(4):
            try:
                with requests.get(url,params=params,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'},
                                  stream=True,timeout=(12,45)) as r:
                    if r.status_code==200:
                        length=r.headers.get('Content-Length')
                        if length and size is None:
                            return int(length)
                        if not (a==0 and b==size-1):
                            raise IOError('server ignored partial range')
                        total=size
                    cr=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                    if r.status_code!=200 and (r.status_code!=206 or not cr or tuple(map(int,cr.groups()[:2]))!=(a,b)):
                        raise IOError(f'range status {r.status_code}')
                    if r.status_code==206:
                        total=int(cr[3])
                    if size is None:
                        return total
                    if size!=total:
                        raise IOError('size changed')
                    path=partial/str(a)
                    tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as f:
                        for data in r.iter_content(256<<10):
                            f.write(data)
                    if tmp.stat().st_size!=b-a+1:
                        raise IOError('short range')
                    tmp.replace(path)
                    return
            except (requests.RequestException,OSError) as e:
                # Requests exceptions can include the authenticated URL.
                if attempt==3:
                    raise RuntimeError('range fetch failed: '+type(e).__name__) from None
                time.sleep(2**attempt)
    size=get(0,0)
    offsets=range(0,size,block)
    with concurrent.futures.ThreadPoolExecutor(8) as workers:
        futures=[]
        for a in offsets:
            b=min(size,a+block)-1
            path=partial/str(a)
            if not path.exists() or path.stat().st_size!=b-a+1:
                futures.append(workers.submit(get,a,b,size))
        for future in futures:
            future.result()
    path=dest/name
    tmp=dest/(name+'.joining')
    h=hashlib.sha256()
    with tmp.open('wb') as f:
        for a in offsets:
            chunk=(partial/str(a)).read_bytes()
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest()!=expected:
        for a in offsets:
            (partial/str(a)).unlink()
        tmp.unlink()
        raise IOError('SHA mismatch')
    tmp.replace(path)
    marker=path.with_suffix('.sha256')
    marker.write_text(f'{expected}  {name}\n')
    for a in offsets:
        (partial/str(a)).unlink()
    partial.rmdir()
    print('verified',name,size,flush=True)


def cycle():
    with (dest/'.pull.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifests=subprocess.check_output(['bash',str(Path(__file__).with_name('mux.sh')),
                    'cd /content/col/data && cat -- *.sha256 2>/dev/null || true'],text=True,timeout=45)
        s=state.get_session(session)
        for line in manifests.splitlines():
            m=re.fullmatch(r'([a-f0-9]{64})  ([a-zA-Z0-9_-]+\.npz)',line)
            if not m:
                raise ValueError('invalid manifest')
            expected,name=m.groups()
            path=dest/name
            mark=path.with_suffix('.sha256')
            if mark.exists() and path.exists() and mark.read_text().strip()==line:
                continue
            try:
                pull_one(s,name,expected)
            except (RuntimeError, IOError) as e:
                print('retry via SSH',name,type(e).__name__,flush=True)
                tmp=path.with_suffix('.partial')
                offset=tmp.stat().st_size if tmp.exists() else 0
                with tmp.open('ab') as f:
                    subprocess.run(['bash',str(Path(__file__).with_name('mux.sh')),
                                   f'tail -c +{offset+1} /content/col/data/{name}'],stdout=f,check=True)
                with tmp.open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=expected:
                        tmp.unlink()
                        raise IOError('SSH checksum mismatch')
                tmp.replace(path)
                mark.write_text(line+'\n')
                print('verified',name,path.stat().st_size,flush=True)


while True:
    try:
        cycle()
    except Exception as e:
        print('pull failed',type(e).__name__,flush=True)
        if '--watch' not in sys.argv:
            raise SystemExit(1)
    if '--watch' not in sys.argv:
        break
    time.sleep(12)
