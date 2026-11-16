#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
while tmux has-session -t nav3_fix_training 2>/dev/null; do sleep 10; done
"$NAV3_PYTHON" - <<'PY'
import json
from pathlib import Path
initial=json.loads(Path('metrics/fix_initial_checkpoints.json').read_text())
for name in ['dreamer_s0','ppo_s0']:
    progress=json.loads(Path(f'checkpoints/{name}_ft/progress.json').read_text())
    assert progress['wall_seconds']-initial[name]['source_progress']['wall_seconds']>=10800
print('BOTH_TRAINING_BUDGETS_COMPLETE',flush=True)
PY
"$NAV3_PYTHON" scripts/fix_final_evaluation.py > logs/fix_final_evaluation.log 2>&1
for name in dreamer_s0_ft ppo_s0_ft; do
  if [ ! -f "videos/fix/${name}_rollout.json" ]; then
    "$NAV3_PYTHON" -m nav3.inference_video --record "$name" --fixed --batch 100 --output videos/fix > "logs/fix_video_${name}.log" 2>&1
  fi
done
"$NAV3_PYTHON" - <<'PY'
import json
from pathlib import Path
for name in ['dreamer_s0_ft','ppo_s0_ft']:
    video=json.loads(Path(f'videos/fix/{name}_rollout.json').read_text())
    row=json.loads(Path(f'metrics/fix_eval_v1_{name}_test_ob0_sf0.json').read_text())['rows'][0]
    for key in ['gates','failure','duration_s','progress_m']:
        assert video[key]==row[key],(name,key,video[key],row[key])
print('VIDEO_MATCHES_FIXED_EVALUATOR_EPISODE_ZERO',flush=True)
PY
"$NAV3_PYTHON" -m nav3.inference_video --render --finetuned --output videos/fix > logs/fix_video_render.log 2>&1
"$NAV3_PYTHON" tests/check_fix.py --reward > logs/fix_checks_final.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/fix_diag_checks_final.json > logs/fix_diag_checks_final.log 2>&1
JAX_PLATFORMS=cpu "$NAV3_PYTHON" scripts/fix_protect.py --check | tee logs/fix_protect_final.log > metrics/fix_integrity_final.json
JAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.fix_report > logs/fix_report.log 2>&1
if [ ! -f metrics/fix_runtime.json ]; then
  "$NAV3_PYTHON" scripts/fix_runtime.py > logs/fix_runtime.log 2>&1
fi
echo FIX_EVALUATION_AND_CHECKS_COMPLETE
