#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
while tmux has-session -t nav3_fix_maps 2>/dev/null; do sleep 10; done
"$NAV3_PYTHON" - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('metrics/fix_g1.json').read_text())['passed']
assert Path('metrics/fix_initial_checkpoints.json').exists()
assert '"reward_checked": true' in Path('logs/fix_reward_checks.log').read_text()
print('G1_AND_REWARD_CHECKS_PASSED',flush=True)
PY
"$NAV3_PYTHON" scripts/fix_final_evaluation.py --initial > logs/fix_initial_blind.log 2>&1
"$NAV3_PYTHON" -m nav3.fix_queue > logs/fix_training_queue.log 2>&1
