#!/usr/bin/env bash
# Usage: bash audit_20261116/nav4_dev/run_checks.sh <group>; JAX device from $JAX_PLATFORMS (cpu when the shared GPU is full).
set -uo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
D=audit_20261116/nav4_dev
M=r9_official_runner/maps_scan_holdout
CK=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
run() {  # name, args...
  local name=$1; shift
  echo "START $name $(date -Is) JAX_PLATFORMS=${JAX_PLATFORMS:-default}" >> $D/logs/matrix.log
  ( time $NAV3_PYTHON "$@" --output $D/$name.json ) > $D/logs/$name.log 2>&1
  echo "END $name rc=$? $(date -Is)" >> $D/logs/matrix.log
}
case "$1" in
  c2) for s in 914901 20261116; do run c2_planner_s$s -m nav4.evaluate --manifest $M --method planner --repeats 8 --seed $s --plot; done ;;
  c3) for a in mode sample; do for s in 914901 20261116; do run c3_dreamer_${a}_s$s -m nav4.evaluate --manifest $M --method dreamer --action $a --checkpoint $CK --repeats 8 --seed $s --plot; done; done ;;
  c4) for s in 914901 20261116; do run c4_hybrid_mode_s$s -m nav4.evaluate --manifest $M --method hybrid --action mode --checkpoint $CK --repeats 8 --seed $s --plot; done ;;
  diag) S=914901
    NOISE=(sensor.height_std=[0,0] sensor.dropout_base=[0.03,0.03] sensor.dropout_distance=[0.00025,0.00025])
    run diag_c3_mode_legacysupport_s$S $D/diag_obs.py legacy_support --manifest $M --method dreamer --action mode --checkpoint $CK --repeats 8 --seed $S
    run diag_c3_mode_nonoise_s$S -m nav4.evaluate --manifest $M --method dreamer --action mode --checkpoint $CK --repeats 8 --seed $S --env "${NOISE[@]}"
    run diag_c3_mode_legacysupport_nonoise_s$S $D/diag_obs.py legacy_support --manifest $M --method dreamer --action mode --checkpoint $CK --repeats 8 --seed $S --env "${NOISE[@]}" ;;
  c7) run c7_planner_forward_strict -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --max-steps 25000 --plot
      run c7_planner_forward_assist -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --max-steps 25000 --assist --plot ;;
esac
echo "GROUP_DONE $1 $(date -Is)" >> $D/logs/matrix.log
