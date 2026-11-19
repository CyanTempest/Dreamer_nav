#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
base=../official_atec/e5a3b987/extracted/01_LiDAR_point_cloud
out=data/trail_a_mujoco/pilot_0_30
for pair in 'removed developer_data/map_remov.las' 'map0 map-result/0.las'; do
    read -r name path <<< "$pair"
    while [[ ! -f "$base/$path" ]]; do
        tmux has-session -t nav4_mjmap_official 2>/dev/null || [[ -f "$base/$path" ]] || exit 1
        sleep 5
    done
    bash nav4/matrix/cpu.sh -m nav4.matrix.survey "$name" > "$out/$name.log" 2>&1
    bash nav4/matrix/cpu.sh -m nav4.matrix.register "$name" >> "$out/$name.log" 2>&1
    bash nav4/matrix/cpu.sh -m nav4.matrix.detail "$name" >> "$out/$name.log" 2>&1
done
