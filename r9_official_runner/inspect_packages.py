"""Read-only official archive inspection. Never extract or execute package files."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import zipfile

INFO_LIMIT=4*1024*1024

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def safe_name(name):
    path=PurePosixPath(name.replace('\\','/'))
    return not path.is_absolute() and '..' not in path.parts and not (path.parts and ':' in path.parts[0])

def las_header(data):
    if len(data)<227 or data[:4]!=b'LASF':raise ValueError('Invalid LAS header')
    major,minor=data[24:26]
    count=struct.unpack_from('<I',data,107)[0]
    if (major,minor)>=(1,4) and len(data)>=255:
        extended=struct.unpack_from('<Q',data,247)[0]
        if extended:count=extended
    scales=struct.unpack_from('<3d',data,131);offsets=struct.unpack_from('<3d',data,155)
    raw=struct.unpack_from('<6d',data,179)
    return dict(version=f'{major}.{minor}',point_count=count,point_format=data[104]&63,
        point_record_bytes=struct.unpack_from('<H',data,105)[0],scale=scales,offset=offsets,
        min_xyz=[raw[1],raw[3],raw[5]],max_xyz=[raw[0],raw[2],raw[4]],
        warning='Header bounds only; alignment with trajectory is not established.')

def trajectory_preview(data):
    rows=[]
    for line in data.decode('utf-8-sig',errors='replace').splitlines():
        parts=line.split()
        if len(parts)!=8:continue
        try:values=list(map(float,parts))
        except ValueError:continue
        if all(math.isfinite(x) for x in values):rows.append(values)
        if len(rows)>=200:break
    if not rows:return dict(numeric_rows=0)
    qnorm=[math.sqrt(sum(x*x for x in r[4:])) for r in rows]
    return dict(numeric_rows=len(rows),first=rows[0],last_preview=rows[-1],
        quaternion_order='qw qx qy qz',quaternion_norm_min=min(qnorm),quaternion_norm_max=max(qnorm),
        timestamps_increasing=all(b[0]>a[0] for a,b in zip(rows,rows[1:])),
        xyz_min=[min(r[k] for r in rows) for k in [1,2,3]],xyz_max=[max(r[k] for r in rows) for k in [1,2,3]],
        warning='First 200 numeric rows only; not whole-path extent or race-route validation.')

def inspect(path,expected_size=None):
    path=Path(path)
    if expected_size is not None and path.stat().st_size!=expected_size:raise ValueError(f'{path.name}: archive size mismatch')
    result=dict(archive=path.name,bytes=path.stat().st_size,members=[],package_info={})
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        for info in infos:
            if not safe_name(info.filename):raise ValueError('Unsafe archive member path')
            item=dict(path=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,crc32=f'{info.CRC:08x}')
            if info.flag_bits&1:raise ValueError('Encrypted archive member')
            name=info.filename.lower()
            if '/_package_info/' in '/'+name and name.endswith(('.json','.txt')) and info.file_size<=INFO_LIMIT:
                with archive.open(info) as f:text=f.read(INFO_LIMIT+1).decode('utf-8-sig',errors='replace')
                result['package_info'][info.filename]=json.loads(text) if name.endswith('.json') else text
            elif name.endswith('.las'):
                with archive.open(info) as f:item['las_header']=las_header(f.read(375))
            elif name.endswith('.ply'):
                lines=[]
                with archive.open(info) as f:
                    for _ in range(256):
                        line=f.readline(4096).decode('ascii',errors='replace').strip();lines.append(line)
                        if line=='end_header':break
                item['ply_header']=lines
            elif name.endswith('.csv'):
                with archive.open(info) as f:item['trajectory_preview']=trajectory_preview(f.read(65536))
            result['members'].append(item)
        result['uncompressed_bytes']=sum(i.file_size for i in infos)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lock',type=Path,default=Path(__file__).with_name('dataset_lock.json'))
    args=parser.parse_args();lock=json.loads(args.lock.read_text())
    results=[];missing=[]
    for name,size in lock['files'].items():
        path=args.directory/name
        if path.exists():results.append(inspect(path,size))
        else:missing.append(name)
    payload=dict(repo_id=lock['repo_id'],revision=lock['revision'],packages=results,missing=missing,
        ready_for_frame_audit=not missing,ready_for_rl_evaluation=False,
        note='Archive inspection does not validate frame registration, route coverage, or collision surfaces.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps(dict(packages=len(results),missing=missing,output=str(args.output)),ensure_ascii=False))
if __name__=='__main__':main()
