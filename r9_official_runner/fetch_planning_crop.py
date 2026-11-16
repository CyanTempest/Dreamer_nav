"""Verify full official planning PLY while retaining only diagnostic map bounds.

Bounded parallel range fetches avoid storing a 22 GB ZIP or 4.2 GB full PLY.
All original compressed bytes are consumed and the full uncompressed SHA checked.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import time
import zipfile
import zlib

import numpy as np

from remote_zip import RangeReader
from build_scan_patches import resample_route
from inspect_packages import file_sha256

DTYPE=np.dtype([(k,'<f4') for k in ['x','y','z','intensity','normal_x','normal_y','normal_z']]+[('time','<f8'),('classification','u1')])


class CropWriter:
    def __init__(self,stream,bounds):
        self.stream=stream;self.bounds=bounds;self.pending=b'';self.header=None
        self.total=0;self.kept=0;self.expected=None
    def feed(self,data):
        self.pending+=data
        if self.header is None:
            marker=self.pending.find(b'end_header\n')
            if marker<0:
                if len(self.pending)>65536:raise ValueError('Invalid PLY header')
                return
            end=marker+len(b'end_header\n');header=self.pending[:end];self.pending=self.pending[end:]
            if not header.startswith(b'ply\nformat binary_little_endian 1.0\n'):raise ValueError('Unsupported PLY format')
            properties=[line for line in header.decode().splitlines() if line.startswith('property ')]
            expected=['property float '+k for k in DTYPE.names[:7]]+['property double time','property uchar classification']
            if properties!=expected:raise ValueError('Unexpected planning PLY record schema')
            match=re.search(rb'element vertex (\d+)',header)
            if match is None:raise ValueError('Missing vertex count')
            self.expected=int(match.group(1))
            self.header=header[:match.start(1)]+b'0'*20+header[match.end(1):]
            self.count_offset=match.start(1);self.stream.write(self.header)
        length=len(self.pending)//DTYPE.itemsize*DTYPE.itemsize
        if not length:return
        block=np.frombuffer(self.pending[:length],dtype=DTYPE)
        keep=np.zeros(len(block),bool)
        for lo,hi in self.bounds:
            keep|=(block['x']>=lo[0])&(block['x']<hi[0])&(block['y']>=lo[1])&(block['y']<hi[1])
        self.stream.write(block[keep].tobytes())
        self.total+=len(block);self.kept+=int(keep.sum());self.pending=self.pending[length:]
    def finish(self):
        if self.pending or self.total!=self.expected:raise ValueError('Incomplete PLY records')
        self.stream.seek(self.count_offset);self.stream.write(f'{self.kept:020d}'.encode())


def main():
    p=argparse.ArgumentParser();p.add_argument('--capabilities',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--stations',nargs='+',type=float,default=[20,145,270,395,520,645,770,895])
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite completed crop')
    _,route=resample_route(np.load(a.trajectory)['developer']);bounds=[]
    for station in a.stations:
        segment=route[int(round(station/.2)):int(round(station/.2))+201]
        if len(segment)!=201:raise ValueError('Incomplete segment')
        bounds.append((np.floor((segment[:,:2].min(0)-8)/.1)*.1,np.ceil((segment[:,:2].max(0)+8)/.1)*.1))
    caps=json.loads(a.capabilities.read_text());entry=next(e for e in caps['files'] if e['name'].startswith('01_'))
    reader=RangeReader(entry['url'],entry['bytes'])
    member='01_LiDAR_point_cloud/ds_map/plannar_map.ply'
    with zipfile.ZipFile(reader) as archive:
        info=archive.getinfo(member);manifest=json.loads(archive.read('01_LiDAR_point_cloud/_package_info/MANIFEST.json'))
    expected=next(e for e in manifest['files'] if e['path']==member)
    if info.compress_type!=zipfile.ZIP_DEFLATED or info.flag_bits&1:raise ValueError('Unsupported compression')
    reader.seek(info.header_offset);header=reader.read(30)
    if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP entry')
    offset=info.header_offset+30+sum(struct.unpack_from('<HH',header,26))
    size=16*1024*1024;count=(info.compress_size+size-1)//size
    def fetch(i):
        for attempt in range(4):
            try:
                r=RangeReader(entry['url'],entry['bytes']);r.seek(offset+i*size)
                return r.read(min(size,info.compress_size-i*size))
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.ply.part')
    digest=hashlib.sha256();decoder=zlib.decompressobj(-15);crc=0;written=0
    with temporary.open('w+b') as stream,ThreadPoolExecutor(max_workers=a.workers) as pool:
        crop=CropWriter(stream,bounds)
        futures=deque(pool.submit(fetch,i) for i in range(min(a.workers,count)))
        for i in range(count):
            compressed=futures.popleft().result()
            if i+a.workers<count:futures.append(pool.submit(fetch,i+a.workers))
            raw=decoder.decompress(compressed);digest.update(raw);crc=zlib.crc32(raw,crc);written+=len(raw);crop.feed(raw)
            if i%10==0:
                print(json.dumps({'chunks':i+1,'total':count,'points_seen':crop.total,'points_kept':crop.kept}),flush=True)
                if shutil.disk_usage(a.output.parent).free<3*1024**3:raise RuntimeError('Disk reserve reached')
        raw=decoder.flush();digest.update(raw);crc=zlib.crc32(raw,crc);written+=len(raw);crop.feed(raw);crop.finish()
    if not decoder.eof or written!=expected['size_bytes'] or digest.hexdigest()!=expected['sha256'] or crc!=info.CRC:
        raise ValueError('Full source integrity verification failed')
    temporary.replace(a.output)
    crop_sha=file_sha256(a.output)
    report=dict(revision=caps['revision'],source_member=member,source_sha256=digest.hexdigest(),
                source_bytes_verified=written,source_points=crop.total,crop_points=crop.kept,
                crop_sha256=crop_sha,crop_bytes=a.output.stat().st_size,
                bounds=[(lo.tolist(),hi.tolist()) for lo,hi in bounds],
                note='Retained all point records inside union of diagnostic patch bounds; no semantic or height filtering')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    a.capabilities.unlink(missing_ok=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
