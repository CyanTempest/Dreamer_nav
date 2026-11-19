#!/usr/bin/env bash
set -u
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
base=data/trail_a_mujoco/full_v1
work=$base/work
export PYTHONPATH=../envs/mjmap/mujoco330
while [[ ! -s $base/segments.json ]]; do
    tmux has-session -t nav4_mjmap_full 2>/dev/null || exit 1
    sleep 12
done
run() {
    local label=$1
    shift
    /usr/bin/time -v bash nav4/matrix/cpu.sh "$@" > "$work/$label.log" 2>&1
    local result=$?
    printf '%s %s %s\n' "$(date -Iseconds)" "$label" "$result" >> "$work/accept_status.txt"
    return "$result"
}
run wood -m nav4.matrix.full_wood
run seam -m nav4.matrix.full_audit seams
run decks -m nav4.matrix.full_decks
run package -m nav4.matrix.full_package
run compare -m nav4.matrix.full_checks compare
if [[ -s $base/route_vs_v8.csv ]]; then
    run physics -m nav4.matrix.full_checks physics
fi
if [[ -s $base/assembly.json ]]; then
    run field_render -m nav4.matrix.full_frames render
    run assemblies -m nav4.matrix.full_checks all-assemblies
fi
