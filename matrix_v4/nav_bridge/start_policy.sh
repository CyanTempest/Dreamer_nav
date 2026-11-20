#!/bin/bash
set -euo pipefail
export PYTHONPATH=/content/collect:/content/nav:/content/matrix-v4/MATRiX_v1.1.2/Tools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=.22 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /content/navenv/bin/python -u /content/nav_bridge/drive.py
