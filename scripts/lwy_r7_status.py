"""Compact read-only view for the requested periodic SSH checks."""
import json
from pathlib import Path
import time


state = json.loads(Path('metrics/lwy_r7_queue.json').read_text())
print(json.dumps({k: v for k, v in state.items() if k != 'jobs'}, ensure_ascii=False))
print('gpu_hours', round(sum(r.get('seconds', 0) for r in state['jobs'] if r.get('gpu')) / 3600, 3))
for row in state['jobs'][-8:]:
    print('job', row['label'], row.get('returncode'),
          round(row.get('seconds', time.time() - row['started']), 1))
for path in sorted(Path('metrics').glob('lwy_r7_val*.json')):
    result = json.loads(path.read_text())
    print('validation', path.name, result['success_rate'], result['mean_gates'], result['failures'])
for path in sorted(Path('metrics').glob('train_*lwy_r7*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get('event') == 'train']
    if not rows:
        continue
    row = rows[-1]
    print('train', path.name, {k: row.get(k) for k in [
        'steps', 'updates', 'wall_seconds', 'replay_old_sample_fraction',
        'online_priority_sample_fraction']})
    metrics = row.get('metrics', {})
    print('health', {k: v for k, v in metrics.items() if any(
        token in k for token in ['grad_norm', 'ent/action', 'retnorm/'])})
print('final_conditions', len(list(Path('metrics').glob('lwy_r7_final_*.json'))))
for path in [Path('metrics/lwy_r7_unit_checks.json'), Path('videos/lwy_r7_dreamer.json')]:
    if path.exists():
        result = json.loads(path.read_text())
        print(path, {k: v for k, v in result.items() if not isinstance(v, list)})
