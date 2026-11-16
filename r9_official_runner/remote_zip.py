"""Read selected ZIP entries by HTTPS byte ranges without fetching the whole ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from inspect_packages import safe_name, las_header, file_sha256


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.transferred = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if pos < 0: raise ValueError('Negative seek')
        self.pos = pos
        return pos

    def read(self, size=-1):
        if size < 0: size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0: return b''
        start, end = self.pos, self.pos + size - 1
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                actual = response.headers.get('Content-Range', '')
                expected = f'bytes {start}-{end}/{self.size}'
                if response.status != 206 or actual != expected:
                    raise RuntimeError('Server did not honor exact byte range')
                data = response.read(size + 1)
        except Exception as exc:
            # Signed URL must not appear in error logs.
            raise RuntimeError('Range request failed: ' + type(exc).__name__) from None
        if len(data) != size: raise RuntimeError('Short range response')
        self.pos += size
        self.transferred += size
        return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capabilities', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extract-root', type=Path)
    p.add_argument('--extract', action='append', default=[])
    args = p.parse_args()
    caps = json.loads(args.capabilities.read_text())
    entry = next(e for e in caps['files'] if e['name'].startswith('01_'))
    reader = RangeReader(entry['url'], entry['bytes'])
    output = {'revision': caps['revision'], 'members': [], 'package_info': {}}
    with zipfile.ZipFile(reader) as archive:
        for info in archive.infolist():
            if not safe_name(info.filename): raise ValueError('Unsafe member')
            output['members'].append({'path': info.filename, 'bytes': info.file_size,
                                      'compressed_bytes': info.compress_size})
            if info.filename.endswith('.las'):
                with archive.open(info) as stream:
                    output['members'][-1]['las_header'] = las_header(stream.read(375))
            if info.filename.endswith('.ply'):
                with archive.open(info) as stream:
                    header = stream.read(4096).split(b'end_header')[0]
                    output['members'][-1]['ply_header'] = header.decode('ascii', errors='replace')
            if '/_package_info/' in info.filename and info.file_size < 4 * 1024 * 1024:
                data = archive.read(info).decode('utf-8-sig')
                output['package_info'][info.filename] = json.loads(data) if info.filename.endswith('.json') else data
        # Save metadata early, before optional larger member downloads.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        manifests = [v for k, v in output['package_info'].items() if k.endswith('MANIFEST.json')]
        entries = {v['path']: v for v in manifests[0]['files']}
        for name in args.extract:
            if args.extract_root is None: raise ValueError('Missing extraction destination')
            if name not in entries or not safe_name(name): raise ValueError('Unmanifested extraction request')
            dest = args.extract_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                digest = file_sha256(dest)
                if digest == entries[name]['sha256']: continue
                raise ValueError('Existing member hash mismatch')
            digest = hashlib.sha256()
            temporary = dest.with_suffix(dest.suffix + '.part')
            with archive.open(name) as src, temporary.open('wb') as dst:
                while chunk := src.read(16 * 1024 * 1024):
                    digest.update(chunk)
                    dst.write(chunk)
            if digest.hexdigest() != entries[name]['sha256']:
                raise ValueError('Extracted member hash mismatch')
            temporary.replace(dest)
            print(json.dumps({'extracted_verified': name, 'bytes': dest.stat().st_size}), flush=True)
    output['transferred_bytes'] = reader.transferred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
