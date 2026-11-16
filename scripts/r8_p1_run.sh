#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/r8_p1_env.sh
# Existing dependencies only; the JAX package tree is read-only.
R8_P1_GPU_PATH="$PWD:$PWD/../envs/dreamer/lib/python3.11/site-packages"
if [[ ! -f fullgeo/r8_p1_provenance_before.json ]]; then
  .venv_r8/bin/python -u -m nav3.r8_p1_common > logs/r8_p1_provenance.log 2>&1
fi
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_reference > logs/r8_p1_reference_reproduce.log 2>&1
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_bake > logs/r8_p1_bake_alpha_reproduce.log 2>&1
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_bake --opaque > logs/r8_p1_bake_opaque_reproduce.log 2>&1
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_validate > logs/r8_p1_validation_cpu_reproduce.log 2>&1
cp metrics/r8_p1_validation.json metrics/r8_p1_validation_cpu.json
cp fullgeo/r8_p1_predictions.npz fullgeo/r8_p1_predictions_cpu.npz
JAX_PLATFORMS=cuda PYTHONPATH="$R8_P1_GPU_PATH" .venv_lwy/bin/python -u -m nav3.r8_p1_validate > logs/r8_p1_validation_gpu_reproduce.log 2>&1
JAX_PLATFORMS=cuda PYTHONPATH="$R8_P1_GPU_PATH" .venv_lwy/bin/python -u -m nav3.r8_p1_checks > logs/r8_p1_checks_reproduce.log 2>&1
JAX_PLATFORMS=cuda PYTHONPATH="$R8_P1_GPU_PATH" .venv_lwy/bin/python -u -m nav3.r8_p1_audit > logs/r8_p1_audit_reproduce.log 2>&1
for method in training simple no_visibility; do
  JAX_PLATFORMS=cuda PYTHONPATH="$R8_P1_GPU_PATH" .venv_lwy/bin/python -u -m nav3.r8_p1_benchmark --method "$method" > "logs/r8_p1_throughput_${method}_reproduce.log" 2>&1
done
.venv_r8/bin/python -u -m nav3.r8_p1_common --after > logs/r8_p1_integrity_reproduce.log 2>&1
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_report > logs/r8_p1_report.log 2>&1
JAX_PLATFORMS=cpu .venv_r8/bin/python -u -m nav3.r8_p1_verify > logs/r8_p1_artifact_check.log 2>&1
