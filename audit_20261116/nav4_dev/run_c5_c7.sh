#!/usr/bin/env bash
# C5 throughput (CPU; GPU too if >= 10 GB free) then C7 assist. Logs under audit_20261116/nav4_dev/logs.
set -uo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
D=audit_20261116/nav4_dev
M=r9_official_runner/maps_scan_holdout
log() { echo "$1 $(date -Is)" >> $D/logs/matrix.log; }
for n in 256 1024; do
  log "START c5_cpu_$n"; JAX_PLATFORMS=cpu $NAV3_PYTHON $D/throughput.py $M $n 10 > $D/logs/c5_cpu_$n.log 2>&1; log "END c5_cpu_$n rc=$?"
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -ge 10240 ]; then
  for n in 256 1024; do
    log "START c5_gpu_$n free=${free}MiB"; $NAV3_PYTHON $D/throughput.py $M $n 50 > $D/logs/c5_gpu_$n.log 2>&1; log "END c5_gpu_$n rc=$?"
  done
else
  log "SKIP c5_gpu free=${free}MiB"
fi
log "START c7_planner_forward_assist"
( time JAX_PLATFORMS=cpu $NAV3_PYTHON -m nav4.evaluate --manifest data/trail_a --maps forward --method planner --repeats 8 \
    --seed 20261116 --max-steps 25000 --assist --plot --output $D/c7_planner_forward_assist.json ) > $D/logs/c7_planner_forward_assist.log 2>&1
log "END c7_planner_forward_assist rc=$?"
log "GROUP_DONE c5_c7"
