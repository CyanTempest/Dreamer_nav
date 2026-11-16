#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
export PYTHONPATH="$PWD/r8_p2_runner:$PYTHONPATH"
export PYTHONUNBUFFERED=1
name=dreamer_lwy_r8_finish_20261114
source_checkpoint=checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl
stage() { "$NAV3_PYTHON" -c 'import json,time,sys;from pathlib import Path;Path("metrics/r8_p3_status.json").write_text(json.dumps(dict(stage=sys.argv[1],unix=time.time(),pid=sys.argv[2]),indent=2))' "$1" "$$"; }
trap 'stage failed' ERR
"$NAV3_PYTHON" -c 'import json;assert json.load(open("metrics/r8_p3_checks.json"))["passed"]'
stage rule_only_evaluation
"$NAV3_PYTHON" -m r8_p3.evaluate --checkpoint "$source_checkpoint" --output metrics/r8_p3_rule_only.json --no-goal-clamp > logs/r8_p3_rule_only.log 2>&1
stage corrected_pretrain_evaluation
"$NAV3_PYTHON" -m r8_p3.evaluate --checkpoint "$source_checkpoint" --output metrics/r8_p3_pretrain.json > logs/r8_p3_pretrain.log 2>&1
stage training
 timeout --signal=INT --kill-after=120 2700 "$NAV3_PYTHON" -m r8_p3.train --name "$name" --envs 256 --seconds 1800 --config r8_p2_runner/r8_p3/config.json --source "$source_checkpoint" > logs/r8_p3_train.log 2>&1
stage paired_final_evaluation
"$NAV3_PYTHON" -m r8_p3.evaluate --checkpoint "checkpoints/$name/latest.pkl" --output metrics/r8_p3_final.json > logs/r8_p3_final.log 2>&1
stage fresh_seed_evaluation
for seed in 831501 831502; do
    "$NAV3_PYTHON" -m r8_p3.evaluate --checkpoint "checkpoints/$name/latest.pkl" --seed "$seed" --output "metrics/r8_p3_fresh_$seed.json" > "logs/r8_p3_fresh_$seed.log" 2>&1
done
stage completed
