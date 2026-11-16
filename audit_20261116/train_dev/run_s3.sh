#!/usr/bin/env bash
# S3: CPU smoke of nav4.train_dreamer: 8 envs on data/trail_a forward; part 1 = fresh run, part 2 = resume.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export JAX_PLATFORMS=cpu
D=${S3_DIR:-audit_20261116/train_dev/s3_run}
SMOKE='eval.traverses=1 eval.max_steps=40 eval.minutes=1.5 save_minutes=3 log_seconds=30'
if [ "$1" = resume ]; then
  start=$(date +%s)
  $NAV3_PYTHON -m nav4.train_dreamer --logdir $D --seconds $2 --resume --allow-cpu
  echo "S3_RESUME_EXIT=$? WALL_S=$(( $(date +%s)-start ))"
else
  start=$(date +%s)
  $NAV3_PYTHON -m nav4.train_dreamer --logdir $D --envs 8 --seconds $1 --allow-cpu --set $SMOKE
  echo "S3_FRESH_EXIT=$? WALL_S=$(( $(date +%s)-start ))"
fi
