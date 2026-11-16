#!/usr/bin/env bash
# Final pass with the final nav4 code: C1 unit tests, C2-C4 (canonical outputs), C7 smoke/strict/assist, C5 GPU if free.
set -uo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
D=audit_20261116/nav4_dev
M=r9_official_runner/maps_scan_holdout
CK=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
log() { echo "$1 $(date -Is)" >> $D/logs/final.log; }
run() {  # name, args...
  local name=$1; shift
  log "START $name"
  ( time JAX_PLATFORMS=cpu $NAV3_PYTHON "$@" --output $D/$name.json ) > $D/logs/$name.log 2>&1
  log "END $name rc=$?"
}
log "START c1"
( time JAX_PLATFORMS=cpu $NAV3_PYTHON -m unittest discover -s nav4/tests -p 'test_*.py' ) > $D/logs/c1_unittest.log 2>&1
log "END c1 rc=$?"
( run c7_planner_forward_assist -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --max-steps 25000 --assist --plot ) &
( log "START c7_smoke"; JAX_PLATFORMS=cpu $NAV3_PYTHON $D/c7_smoke.py > $D/logs/c7_smoke_cpu.log 2>&1; log "END c7_smoke rc=$?"
  run c7_planner_forward_strict -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 --seed 20261116 --max-steps 25000 --plot
  for s in 914901 20261116; do run c2_planner_s$s -m nav4.evaluate --manifest $M --method planner --repeats 8 --seed $s --plot; done ) &
( for s in 914901 20261116; do run c3_dreamer_mode_s$s -m nav4.evaluate --manifest $M --method dreamer --action mode --checkpoint $CK --repeats 8 --seed $s --plot; done ) &
( for s in 914901 20261116; do run c3_dreamer_sample_s$s -m nav4.evaluate --manifest $M --method dreamer --action sample --checkpoint $CK --repeats 8 --seed $s --plot; done ) &
( for s in 914901 20261116; do run c4_hybrid_mode_s$s -m nav4.evaluate --manifest $M --method hybrid --action mode --checkpoint $CK --repeats 8 --seed $s --plot; done ) &
wait
log "FINAL_DONE"
