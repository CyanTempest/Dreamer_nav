"""Final artifact and protocol checks; never retrains or changes old artifacts."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    state = read('metrics/lwy_r7_queue.json')
    assert not state.get('active') and state['status'] == 'final_complete'
    assert all(row.get('returncode') == 0 for row in state['jobs'])
    assert read('metrics/lwy_r7_gate.json')['passed']
    assert read('metrics/lwy_r7_unit_checks.json')['passed']
    for test in ['check_diag', 'check_lwy']:
        assert read(f'metrics/lwy_r7_{test}_final.json')['passed']
    best = max(state['rounds'], key=lambda r: (r['success_rate'], r['mean_gates']))
    assert state['best'] == best
    assert best['checkpoint'].startswith('checkpoints/snapshots/dreamer_lwy_r7_')
    conditions = []
    for method in ['dreamer_lwy_r7', 'dreamer_lwy', 'ppo_lwy', 'planner']:
        for dataset in ['original', 'blind', 'heldout', 'mini']:
            for ob in [0, 1]:
                for sf in [0, 1]:
                    path = f'metrics/lwy_r7_final_{method}_{dataset}_ob{ob}_sf{sf}.json'
                    result = read(path)
                    assert result['episodes'] == (30 if dataset == 'mini' else 100)
                    assert len(result['rows']) == result['episodes']
                    assert result['successes'] == sum(r['success'] for r in result['rows'])
                    assert result['random_obstacles'] == bool(ob)
                    assert result['safety_filter'] == bool(sf)
                    assert result['obstacle_rule'] == 'shoulder'
                    assert result['support_margin'] == 0
                    assert result['runtime']['policy_batch_width'] == 100
                    assert '--xla_gpu_autotune_level=1' in result['runtime']['xla_flags']
                    assert '--xla_gpu_deterministic_ops=true' in result['runtime']['xla_flags']
                    assert result['runtime']['tf_cudnn_deterministic'] == '1'
                    if method == 'dreamer_lwy_r7':
                        assert result['checkpoint'] == best['checkpoint']
                    conditions.append(path)
    for path in Path('metrics').glob('lwy_r7_val*.json'):
        result = read(path)
        assert result['episodes'] == 80
        assert len({r['map'] for r in result['rows']}) == 20
        assert not result['random_obstacles'] and not result['safety_filter']
    support = sorted(Path('metrics').glob('lwy_r7_support_*.json'))
    assert len(support) == 4
    for path in support:
        result = read(path)
        assert result['support_margin'] == .08 and result['episodes'] == 100
        assert result['safety_filter']
    gpu = state.get('gate_gpu_seconds_upper_bound', 0) + sum(r.get('seconds', 0) for r in state['jobs'] if r['gpu'])
    diagnosis = sum(r.get('seconds', 0) for r in state['jobs'] if r['gpu'] and (
        r['label'] == 'val_initial' or r['label'].startswith(('diag_', 'val_diag_'))))
    ppo = sum(r.get('seconds', 0) for r in state['jobs'] if r['gpu'] and (
        r['label'].startswith(('memory_ppo_', 'ppo_train_', 'val_ppo'))))
    assert gpu <= 36000 and diagnosis <= 3600 and ppo <= 3600
    assert all(r['seconds'] < 10800 for r in state['jobs'])
    result = dict(passed=True, final_conditions=len(conditions), support_conditions=len(support),
                  model=best, gpu_process_hours=gpu / 3600,
                  diagnosis_gpu_seconds=diagnosis, ppo_gpu_seconds=ppo,
                  gate_gpu_seconds_upper_bound=state.get('gate_gpu_seconds_upper_bound', 0),
                  note='Includes the conservative elapsed interval for the two initial planner gates.')
    with Path('metrics/lwy_r7_acceptance.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
