#!/usr/bin/env bash
# Source from the project root. Uses scripts/env.sh, then redirects every
# cache/tmp location into audit_20261116/ so no existing file is touched.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
export PYTHONPATH="$PWD/r9_official_runner:$PYTHONPATH"
AUD="$PWD/audit_20261116"
export XDG_CACHE_HOME="$AUD/cache"
export MPLCONFIGDIR="$AUD/cache/matplotlib"
export NUMBA_CACHE_DIR="$AUD/cache/numba"
export __GL_SHADER_DISK_CACHE_PATH="$AUD/cache/nvidia"
export TMPDIR="$AUD/tmp"
export UV_CACHE_DIR="$AUD/cache/uv"
export JAX_COMPILATION_CACHE_DIR="$AUD/cache/jax"
export CUDA_CACHE_PATH="$AUD/cache/cuda"
export TORCH_HOME="$AUD/cache/torch"
export TRITON_CACHE_DIR="$AUD/cache/triton"
mkdir -p "$AUD/cache" "$AUD/tmp" "$AUD/logs" "$AUD/results"
