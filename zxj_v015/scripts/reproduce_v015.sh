#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export ATEC_JAX_PLATFORM=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl

case "${1:-}" in
  check)
    shift
    exec .venvs/scene/bin/python scripts/check_v015_inputs.py "$@"
    ;;
  smoke|course|suite)
    if [[ $# -ne 2 ]]; then
      printf 'Expected a new output directory for %s\n' "$1" >&2
      exit 2
    fi
    if [[ $1 == suite ]]; then
      .venvs/scene/bin/python scripts/check_v015_inputs.py --video
      exec .venvs/scene/bin/python scripts/evaluate_spatial_checkpoint_suite.py weights/v015 \
        --output "$2" --seed-start 6001 --parallel-modes auto
    fi
    .venvs/scene/bin/python scripts/check_v015_inputs.py
    if [[ $1 == smoke ]]; then
      args=(--start-s 65 --length 3 --steps 301)
    else
      args=(--start-s 0 --length 100 --steps 3501 --depth-noise)
    fi
    exec bash scripts/run_matrix_isolated.sh --script scripts/evaluate_spatial_checkpoint.py \
      weights/v015 --output "$2" --seed 6001 --action-mode mean "${args[@]}"
    ;;
  video)
    if [[ $# -ne 2 ]]; then
      printf 'Expected the path of a successful course run\n' >&2
      exit 2
    fi
    .venvs/scene/bin/python scripts/check_v015_inputs.py --video
    exec .venvs/scene/bin/python scripts/record_verified_navigation.py "$2"
    ;;
  *)
    printf 'Usage: %s check [--video] | smoke OUTPUT | course OUTPUT | suite OUTPUT | video RUN\n' "$0" >&2
    exit 2
    ;;
esac
