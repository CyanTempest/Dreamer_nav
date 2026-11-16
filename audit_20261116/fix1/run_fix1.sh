#!/usr/bin/env bash
# Fix1 acceptance runs T2 (default path, as run_holdout.sh) and T3 (--action mode, r8).
set -uo pipefail
source /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3/audit_20261116/audit_env.sh
R=r9_official_runner
F="$AUD/fix1"
CK=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
gpu_guard() {
  nvidia-smi >> "$F/nvidia_smi_checks.log" 2>&1
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [[ -n "$apps" ]]; then echo "GPU_BUSY before $1: $apps" | tee -a "$F/run.log"; exit 3; fi
}
run() {  # name, args...
  local name=$1; shift
  gpu_guard "$name"
  echo "START $name $(date -Is)" | tee -a "$F/run.log"
  "$NAV3_PYTHON" "$R/diagnose_policy.py" --maps "$R/maps_scan_holdout" --body-bands --steps 1502 --output "$F/$name.json" "$@" > "$F/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -Is)" | tee -a "$F/run.log"
  [[ $rc -eq 0 ]] || exit $rc
}
run holdout_planner_default --repeats 2 --seed 914901 --method planner
run holdout_dreamer_default --repeats 2 --seed 914901 --checkpoint "$CK"
run holdout_hybrid_default --repeats 2 --seed 914901 --checkpoint "$CK" --safety-filter --filter-hold-steps 5
run dreamer_mode_g2_s914901_r8 --repeats 8 --seed 914901 --checkpoint "$CK" --action mode
echo FIX1_RUNS_DONE | tee -a "$F/run.log"
