#!/usr/bin/env bash
# Usage: bash audit_20261116/run_matrix.sh {sanity|A|B}
set -uo pipefail
source /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3/audit_20261116/audit_env.sh
R=r9_official_runner
CK=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
COMMON=(--maps "$R/maps_scan_holdout" --body-bands --steps 1502)

gpu_guard() {
  nvidia-smi >> "$AUD/logs/nvidia_smi_checks.log" 2>&1
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [[ -n "$apps" ]]; then
    echo "GPU_BUSY before $1: $apps" | tee -a "$AUD/logs/matrix.log"
    exit 3
  fi
}

run() {  # name, args...
  local name=$1; shift
  gpu_guard "$name"
  local t0=$(date +%s.%N)
  echo "START $name $(date -Is)" | tee -a "$AUD/logs/matrix.log"
  "$NAV3_PYTHON" "$AUD/eval_ab.py" "${COMMON[@]}" --output "$AUD/results/$name.json" "$@" > "$AUD/logs/$name.log" 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  echo "END $name rc=$rc wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')s $(date -Is)" | tee -a "$AUD/logs/matrix.log"
  [[ $rc -eq 0 ]] || exit $rc
}

case "${1:-}" in
  sanity)
    run sanity --repeats 2 --seed 914901 --yaw-gain 2 --sanity --sanity-steps 20 --checkpoint "$CK"
    ;;
  A)
    run A_planner_g2_s914901_r2 --repeats 2 --seed 914901 --yaw-gain 2 --method planner
    run A_dreamer_sample_g2_s914901_r2 --repeats 2 --seed 914901 --yaw-gain 2 --method dreamer --action sample --checkpoint "$CK"
    ;;
  B)
    for seed in 914901 20261116; do
      for g in 2 4; do
        run "B_planner_g${g}_s${seed}_r8" --repeats 8 --seed "$seed" --yaw-gain "$g" --method planner
      done
      for act in sample mode; do
        for g in 2 4; do
          run "B_dreamer_${act}_g${g}_s${seed}_r8" --repeats 8 --seed "$seed" --yaw-gain "$g" --method dreamer --action "$act" --checkpoint "$CK"
        done
      done
    done
    ;;
  *) echo "usage: $0 {sanity|A|B}"; exit 2;;
esac
echo "PHASE_DONE ${1}" | tee -a "$AUD/logs/matrix.log"
