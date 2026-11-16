# Source from anywhere: nav4 development runtime (audit env + shared-GPU limits + deterministic XLA).
source /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3/audit_20261116/audit_env.sh
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export XLA_FLAGS="--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true"
export TF_CUDNN_DETERMINISTIC=1
