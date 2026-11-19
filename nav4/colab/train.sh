#!/usr/bin/env bash
# VM entry point. A short preflight uses a separate logdir and fresh replay.
set -euo pipefail
cd /content/nav
map= logdir= seconds= envs= model= batch= seed= minutes= preflight=1 fast=0 save_minutes= weights=
while (($#)); do
  case "$1" in
    --map) map=$2;; --logdir) logdir=$2;; --seconds) seconds=$2;;
    --envs) envs=$2;; --model) model=$2;; --batch) batch=$2;; --seed) seed=$2;;
    --save-minutes) save_minutes=$2;; --eval-minutes) minutes=$2;; --weights) weights=$2;;
    --fast-a100) fast=1; shift; continue;;
    --skip-preflight) preflight=0; shift; continue;;
    *) echo "Unknown option $1" >&2; exit 2;;
  esac
  shift 2
done
: "${map:?--map}" "${logdir:?--logdir}" "${seconds:?--seconds}"
map=$(realpath -- "$map")
logdir=$(realpath -m -- "$logdir")
[[ -r "$map/manifest.json" && ! -e "$logdir/metrics.jsonl" ]]
mkdir -p "$logdir"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_COMPILATION_CACHE_DIR=/content/nav4-jax-cache
export XLA_FLAGS='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
args=()
if [[ -n "$envs" ]]; then args+=(--envs "$envs"); fi
if [[ -n "$weights" ]]; then args+=(--init-checkpoint "$weights"); fi
if [[ "$map" == /content/nav/* ]]; then map=${map#/content/nav/}; fi
map_json=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$map")
overrides=("manifest=$map_json" xla_memory_fraction=0.90 log_seconds=15)
if (( fast )); then
  unset XLA_FLAGS TF_CUDNN_DETERMINISTIC
  export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
  overrides+=(xla_preallocate=true xla_default_tuning=true replay.compress=false replay.batched=true
              eval.minutes=60 eval.traverses=8 eval.at_end=true)
fi
if [[ -n "$save_minutes" ]]; then overrides+=("save_minutes=$save_minutes"); fi
if [[ -n "$model" ]]; then overrides+=("model=\"$model\""); fi
if [[ -n "$batch" ]]; then overrides+=("batch_size=$batch"); fi
if [[ -n "$seed" ]]; then overrides+=("seed=$seed"); fi
run() {
  local settings=("${overrides[@]}")
  if [[ -n "$3" ]]; then settings+=("eval.minutes=$3"); fi
  if [[ "$1" == "$logdir/preflight" ]]; then settings+=(eval.at_end=false); fi
  /content/nav4-env/bin/python -m nav4.train_dreamer "${args[@]}" --logdir "$1" --seconds "$2" --set "${settings[@]}"
}
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$logdir/gpu.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
if (( preflight )); then
  run "$logdir/preflight" 10 0 2>&1 | tee "$logdir/preflight.log"
fi
run "$logdir" "$seconds" "$minutes" 2>&1 | tee "$logdir/train.log"

