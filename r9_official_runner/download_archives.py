"""Resume official archives from short-lived capabilities; never log URLs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capabilities', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    caps = json.loads(args.capabilities.read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    status_path = args.directory / 'download_status.json'
    status = {'revision': caps['revision'], 'state': 'running', 'files': {}}

    def save():
        status['updated_unix'] = time.time()
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)

    try:
        for entry in caps['files']:
            name, expected = entry['name'], entry['bytes']
            if Path(name).name != name:
                raise ValueError('Invalid archive basename')
            dest = args.directory / name
            part = dest.with_suffix(dest.suffix + '.part')
            if dest.exists():
                if dest.stat().st_size != expected:
                    raise ValueError('Existing archive size mismatch')
                status['files'][name] = 'size_verified_existing'
                save()
                continue
            status['files'][name] = 'downloading'
            save()
            config = args.capabilities.with_suffix('.curl')
            # JSON quoting is compatible with curl config quoted strings.
            config.write_text('url = ' + json.dumps(entry['url']) + '\n')
            result = subprocess.run([
                'curl', '--silent', '--show-error', '--fail', '--location',
                '--proto', '=https', '--proto-redir', '=https',
                '--connect-timeout', '30', '--retry', '5', '--retry-delay', '5',
                '--speed-time', '90', '--speed-limit', '1024',
                '--continue-at', '-', '--output', str(part), '--config', str(config)
            ], capture_output=True)
            config.unlink(missing_ok=True)
            if result.returncode:
                status['files'][name] = {'curl_exit_code': result.returncode,
                                         'downloaded_bytes': part.stat().st_size if part.exists() else 0}
                raise RuntimeError('Archive transfer failed; refresh capability to resume')
            if part.stat().st_size != expected:
                raise ValueError('Downloaded archive size mismatch')
            part.replace(dest)
            status['files'][name] = 'size_verified_downloaded'
            save()
            print(json.dumps({'downloaded': name, 'bytes': expected}), flush=True)
        status['state'] = 'downloaded_pending_integrity_audit'
    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = str(exc)
        raise
    finally:
        save()
        args.capabilities.unlink(missing_ok=True)
        args.capabilities.with_suffix('.curl').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
