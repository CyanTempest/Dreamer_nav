#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
export PYTHONPATH="$PWD/r8_p2_runner:$PYTHONPATH"
export PYTHONUNBUFFERED=1
name=dreamer_lwy_r8_pilot_20261113
source_checkpoint=checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl
stage() { "$NAV3_PYTHON" -c 'import json,time,sys;from pathlib import Path;Path("metrics/r8_p2_status.json").write_text(json.dumps(dict(stage=sys.argv[1],unix=time.time(),pid=sys.argv[2]),indent=2))' "$1" "$$"; }
trap 'stage failed' ERR
stage preflight
"$NAV3_PYTHON" -m r8_p2.check > logs/r8_p2_preflight.log 2>&1
stage baseline_evaluation
"$NAV3_PYTHON" -m r8_p2.evaluate --checkpoint "$source_checkpoint" --output metrics/r8_p2_baseline.json > logs/r8_p2_baseline.log 2>&1
stage training
"$NAV3_PYTHON" -m r8_p2.train --name "$name" --envs 256 --seconds 3600 --config r8_p2_runner/config.json --source "$source_checkpoint" > logs/r8_p2_train.log 2>&1
stage final_evaluation
"$NAV3_PYTHON" -m r8_p2.evaluate --checkpoint "checkpoints/$name/latest.pkl" --output metrics/r8_p2_final.json > logs/r8_p2_final.log 2>&1
stage completed
