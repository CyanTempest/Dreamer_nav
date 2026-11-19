#!/usr/bin/env bash
set -eu
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
base=data/trail_a_mujoco/full_v1
work=$base/work
export PYTHONPATH=../envs/mjmap/mujoco330
while tmux has-session -t nav4_mjmap_controlled 2>/dev/null; do
    sleep 8
done
run() {
    local label=$1
    shift
    local result=0
    /usr/bin/time -v bash nav4/matrix/cpu.sh "$@" > "$work/final_$label.log" 2>&1 || result=$?
    printf '%s %s %s\n' "$(date -Iseconds)" "$label" "$result" >> "$work/final_status.txt"
    return "$result"
}
run hard -m nav4.matrix.full_hard_audit
if [[ ! -d $work/matrix_pre_audit && -d $base/matrix ]]; then
    mv "$base/matrix" "$work/matrix_pre_audit"
fi
if [[ ! -f $work/load_assemblies_pre_audit.json && -f $base/load_assemblies.json ]]; then
    mv "$base/load_assemblies.json" "$work/load_assemblies_pre_audit.json"
fi
run package -m nav4.matrix.full_package
run compare -m nav4.matrix.full_checks compare
run physics -m nav4.matrix.full_checks physics
run align -m nav4.matrix.full_frames align
run render -m nav4.matrix.full_frames render
run window -m nav4.matrix.full_checks assemblies --entry window_00.xml || true
run robot_window -m nav4.matrix.full_checks assemblies --entry zgws_atec_window_00.xml
run full_scene -m nav4.matrix.full_checks assemblies --entry scene_full.xml || true
run full_robot -m nav4.matrix.full_checks assemblies --entry zgws_atec_full.xml || true
