#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
out=data/trail_a_mujoco/pilot_0_30
while [[ ! -f "$out/removed_registration.json" ]]; do
    tmux has-session -t nav4_mjmap_scan 2>/dev/null || [[ -f "$out/removed_registration.json" ]] || exit 1
    sleep 4
done
/usr/bin/time -v bash nav4/matrix/cpu.sh -m nav4.matrix.build --source removed > "$out/build.log" 2>&1
