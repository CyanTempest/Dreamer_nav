"""A ZIP64 NPZ with step arrays and indexed, deduplicated native-resolution scans."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from codec import pack_wire
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4<<20), b''):
            h.update(block)
    return h.hexdigest()


class Episode:
    def __init__(self,out,name,meta):
        self.meta,self.rows = meta,[]
        self.path = Path(out)/(name+'.npz')
        self.tmp = self.path.with_suffix('.partial')
        self.zip = zipfile.ZipFile(self.tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=1)
        self.writer=ThreadPoolExecutor(1)
        self.pending=deque()
        self.last = {}
        self.count = dict(front=0,rear=0)

    def put(self,key,value,stored=False):
        buf=io.BytesIO()
        np.lib.format.write_array(buf,np.asarray(value),allow_pickle=False)
        self.zip.writestr(key+'.npy',buf.getbuffer(),compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,compresslevel=1)

    def scan(self,side,scan):
        if scan is None:
            return -1
        if self.last.get(side) != scan.stamp:
            i=self.count[side]
            prefix=f'{side}_{i:05d}_'
            if len(self.pending)>=8:
                self.pending.popleft().result()
            self.pending.append(self.writer.submit(self.write_scan,prefix,scan))
            self.last[side]=scan.stamp
            self.count[side]+=1
        return self.count[side]-1

    def write_scan(self,prefix,scan):
        self.put(prefix+'mm',scan.distance)
        self.put(prefix+'intensity',scan.intensity)
        self.put(prefix+'cdr_zstd',np.frombuffer(pack_wire(scan.raw),np.uint8),stored=True)
        self.put(prefix+'shape_count',[*scan.shape,int(np.prod(scan.shape)),scan.finite])
        self.put(prefix+'times',[scan.stamp,scan.end,scan.received])

    def finish(self,reason):
        self.writer.shutdown(wait=True)
        for future in self.pending:
            future.result()
        for key in (self.rows[0] if self.rows else ()):
            self.put(key,[r[key] for r in self.rows])
        self.put('meta',json.dumps(dict(self.meta,termination=reason,steps=len(self.rows)),allow_nan=False))
        self.zip.close()
        self.tmp.replace(self.path)
        checksum = self.path.with_suffix('.sha256')
        tmp=checksum.with_suffix('.sha256.tmp')
        tmp.write_text(f'{digest(self.path)}  {self.path.name}\n')
        tmp.replace(checksum)
