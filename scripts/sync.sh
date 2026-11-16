#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
remote='hpf@100.108.114.64:wang-sm/ATEC_NAV_WSM/research/exp_nav3/'
rsync -a -e 'ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3' --exclude='.git' --exclude='__pycache__' --exclude='codex.log' --exclude='codex_recover.log' --exclude='metrics/' --exclude='figures/' --exclude='logs/' --exclude='REPORT.md' --exclude='maps/' --exclude='ray_scene/' --exclude='checkpoints/' --exclude='replay/' --exclude='cache/' --exclude='tmp/' --exclude='.venv/' ./ "$remote"
