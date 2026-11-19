#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
export PYTHONPATH=../envs/mjmap/mujoco330
out=data/trail_a_mujoco/pilot_0_30
bash nav4/matrix/rebuild.sh
bash nav4/matrix/cpu.sh -m nav4.matrix.clearance > "$out/clearance.log" 2>&1
bash nav4/matrix/cpu.sh -m nav4.matrix.package > "$out/package.log" 2>&1
