#!/usr/bin/env bash
# Source from the experiment root. All caches and outputs stay in exp_nav3.
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/cache"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export NUMBA_CACHE_DIR="$PWD/cache/numba"
export __GL_SHADER_DISK_CACHE_PATH="$PWD/cache/nvidia"
export TMPDIR="$PWD/tmp"
export UV_CACHE_DIR="$PWD/cache/uv"
export JAX_COMPILATION_CACHE_DIR="$PWD/cache/jax"
export CUDA_CACHE_PATH="$PWD/cache/cuda"
export TORCH_HOME="$PWD/cache/torch"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" logs metrics figures checkpoints replay
export PYTHONPATH="$PWD:$PWD/dreamerv3${PYTHONPATH:+:$PYTHONPATH}"
export NAV3_PYTHON="$PWD/../envs/dreamer/bin/python"
