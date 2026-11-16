#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for dir in metrics figures logs; do
  rsync -a -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3' "hpf@100.108.114.64:wang-sm/ATEC_NAV_WSM/research/exp_nav3/$dir/" "$dir/"
done
rsync -a -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3' "hpf@100.108.114.64:wang-sm/ATEC_NAV_WSM/research/exp_nav3/REPORT.md" ./REPORT.md
