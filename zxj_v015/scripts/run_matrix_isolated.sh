#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# A private network namespace has only loopback; a private /dev/shm prevents
# collisions with hardware drivers or other simulator/controller instances.
exec unshare -Urnm bash -c '
  set -euo pipefail
  mount -t tmpfs tmpfs /dev/shm
  ip link set lo up multicast on
  ip route add default dev lo
  export ATEC_MATRIX_ISOLATED=1 MUJOCO_GL=egl
  MATRIX_DEPS="$PWD/third_party/runtimes/MATRiX_Robot_MC-v0.6.6/local_deps"
  export LD_LIBRARY_PATH="$MATRIX_DEPS/usr/lib:$MATRIX_DEPS/usr/local/lib:$MATRIX_DEPS/usr/lib/x86_64-linux-gnu"
  unset PYTHONPATH
  if [[ "${1:-}" == "--train" ]]; then
    shift
    export JAX_PLATFORMS="${ATEC_JAX_PLATFORM:-cpu}"
    exec "${ATEC_DREAMER_PYTHON:-.venvs/dreamer/bin/python}" scripts/train.py "$@"
  fi
  if [[ "${1:-}" == "--evaluate" ]]; then
    shift
    export JAX_PLATFORMS="${ATEC_JAX_PLATFORM:-cpu}"
    exec "${ATEC_DREAMER_PYTHON:-.venvs/dreamer/bin/python}" scripts/evaluate_dreamer.py "$@"
  fi
  if [[ "${1:-}" == "--evaluate-spatial-probe" ]]; then
    shift
    export JAX_PLATFORMS="${ATEC_JAX_PLATFORM:-cpu}"
    exec "${ATEC_DREAMER_PYTHON:-.venvs/dreamer/bin/python}" scripts/evaluate_spatial_observation_probe.py "$@"
  fi
  if [[ "${1:-}" == "--collect" ]]; then
    shift
    exec .venvs/scene/bin/python scripts/collect_navigation_demos.py "$@"
  fi
  if [[ "${1:-}" == "--script" ]]; then
    shift
    exec .venvs/scene/bin/python "$@"
  fi
  exec .venvs/scene/bin/python scripts/probe_matrix_controller.py "$@"
' matrix-simulation "$@"
