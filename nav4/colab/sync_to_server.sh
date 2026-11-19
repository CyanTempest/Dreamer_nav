#!/usr/bin/env bash
# Server entry point: sync_to_server.sh RUN [--once] [SESSION]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=${1:?run name required}
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]
shift
once=0
if [[ ${1:-} == --once ]]; then once=1; shift; fi
session=${1:-nav4a100-dev}
dest="$here/../../../nav4_colab/runs/$name"
mkdir -p "$dest"
while :; do
  rc=0
  rsync -rt --delay-updates --partial-dir=.rsync-partial --timeout=180 -e "$here/vm-ssh.sh" \
    --include='/metrics.jsonl' --include='/*.log' --include='/best.pkl' --include='/latest.pkl' \
    --include='/best.json' --include='/config.json' --include='/gpu.csv' --include='/exitcode' --exclude='*' \
    "$session:/content/runs/$name/" "$dest/" || rc=$?
  if (( once )); then exit "$rc"; fi
  if (( rc )); then echo "$(date -Is) rsync exit $rc" >&2; fi
  sleep 600
done

