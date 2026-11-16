"""Fetch one ZIP member in parallel byte ranges; verify raw file SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

from remote_zip import RangeReader
from inspect_packages import safe_name, file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--capabilities',type=Path,required=True)
    p.add_argument('--member',required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    caps=json.loads(a.capabilities.read_text());entry=next(e for e in caps['files'] if e['name'].startswith('01_'))
    reader=RangeReader(entry['url'],entry['bytes'])
    with zipfile.ZipFile(reader) as archive:
        info=archive.getinfo(a.member)
        manifest=json.loads(archive.read('01_LiDAR_point_cloud/_package_info/MANIFEST.json'))
    expected=next(e for e in manifest['files'] if e['path']==a.member)
    if not safe_name(a.member) or info.compress_type!=zipfile.ZIP_DEFLATED or info.flag_bits&1:
        raise ValueError('Unsupported archive entry')
    reader.seek(info.header_offset);header=reader.read(30)
    if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid local header')
    namelen,extralen=struct.unpack_from('<HH',header,26)
    offset=info.header_offset+30+namelen+extralen
    destination=a.root/a.member;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        digest=file_sha256(destination)
        if digest!=expected['sha256']:raise ValueError('Existing file checksum mismatch')
        print('Already verified');return
    cache=destination.with_suffix('.compressed_parts');cache.mkdir(exist_ok=True)
    size=16*1024*1024;count=(info.compress_size+size-1)//size
    def fetch(i):
        n=min(size,info.compress_size-i*size);dest=cache/f'{i:05}.part'
        if dest.exists() and dest.stat().st_size==n:return i
        for attempt in range(4):
            try:
                r=RangeReader(entry['url'],entry['bytes']);r.seek(offset+i*size)
                data=r.read(n);temp=dest.with_suffix('.tmp');temp.write_bytes(data);temp.replace(dest)
                return i
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for done,_ in enumerate(pool.map(fetch,range(count)),1):
            if done%10==0:print(json.dumps({'compressed_chunks':done,'total':count}),flush=True)
    digest=hashlib.sha256();decoder=zlib.decompressobj(-15);crc=0;written=0
    temporary=destination.with_suffix(destination.suffix+'.parallel.part')
    with temporary.open('wb') as stream:
        for i in range(count):
            raw=decoder.decompress((cache/f'{i:05}.part').read_bytes())
            stream.write(raw);digest.update(raw);crc=zlib.crc32(raw,crc);written+=len(raw)
        raw=decoder.flush();stream.write(raw);digest.update(raw);crc=zlib.crc32(raw,crc);written+=len(raw)
    if not decoder.eof or written!=expected['size_bytes'] or digest.hexdigest()!=expected['sha256'] or crc!=info.CRC:
        raise ValueError('Member integrity verification failed')
    temporary.replace(destination)
    # Delete only this downloader's duplicate compressed chunks after verification.
    for i in range(count):(cache/f'{i:05}.part').unlink()
    cache.rmdir()
    print(json.dumps({'verified':a.member,'bytes':written,'sha256':digest.hexdigest()}),flush=True)


if __name__=='__main__':main()
