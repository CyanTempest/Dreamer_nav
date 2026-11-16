#!/usr/bin/env bash
# Source from exp_nav3. Every cache is new and isolated from P0.
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/fullgeo/r8_p1_cache"
export MPLCONFIGDIR="$PWD/fullgeo/r8_p1_cache/matplotlib"
export CUDA_CACHE_PATH="$PWD/fullgeo/r8_p1_cache/cuda"
export WP_CACHE_DIR="$PWD/fullgeo/r8_p1_cache/warp"
export JAX_COMPILATION_CACHE_DIR="$PWD/fullgeo/r8_p1_cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.24
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export PYTHONPATH="$PWD"
