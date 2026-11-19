#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/data/trail_a_mujoco/pilot_0_30/.matplotlib"
export XDG_CACHE_HOME="$PWD/../envs/mjmap/cache" MESA_SHADER_CACHE_DISABLE=true
export LD_LIBRARY_PATH="$PWD/../envs/mjmap/osmesa/usr/lib/x86_64-linux-gnu"
ulimit -v 24000000
exec taskset -c 0-7 ../envs/mjmap/bin/python "$@"
