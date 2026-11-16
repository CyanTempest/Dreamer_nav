"""Summarize bounded interventions without treating one validation as causality."""
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def paired(reference, candidate):
    assert [(r['map'], r['noise_seed']) for r in reference['rows']] == [
        (r['map'], r['noise_seed']) for r in candidate['rows']]
    gained = sum(not a['success'] and b['success']
                 for a, b in zip(reference['rows'], candidate['rows']))
    lost = sum(a['success'] and not b['success']
               for a, b in zip(reference['rows'], candidate['rows']))
    discordant = gained + lost
    p = min(1., 2 * sum(math.comb(discordant, k)
                       for k in range(min(gained, lost) + 1)) / 2 ** discordant)
    return dict(gained=gained, lost=lost, exact_mcnemar_p=p,
                note='Exploratory paired comparison; no multiple-comparison correction.')


def main():
    state = read('metrics/lwy_r7_queue.json')
    initial = read('metrics/lwy_r7_val_initial.json')
    control_path = Path('metrics/lwy_r7_val_diag_baseline.json')
    control = read(control_path) if control_path.exists() else None
    arms = []
    for config in ['baseline', 'low_lr', 'uniform']:
        result_path = Path(f'metrics/lwy_r7_val_diag_{config}.json')
        if not result_path.exists():
            continue
        result = read(result_path)
        name = 'dreamer_lwy_r7_diag_' + config
        rows = [json.loads(s) for s in Path(f'metrics/train_{name}.jsonl').read_text().splitlines()]
        traces = [r for r in rows if r.get('event') == 'update_trace']
        reports = [r for r in rows if r.get('event') == 'train']
        last_report = reports[-1]['metrics'] if reports else {}
        hi = last_report.get('train/retnorm/hi/value')
        lo = last_report.get('train/retnorm/lo/value')
        job = next(r for r in state['jobs'] if r['label'] == 'diag_' + config)
        arm = dict(config=config, success_rate=result['success_rate'],
                   mean_gates=result['mean_gates'], failures=result['failures'],
                   process_seconds=job['seconds'], updates=max(r.get('updates', 0) for r in rows),
                   first_update_seconds=traces[0]['wall_seconds'] if traces else None,
                   last_update_seconds=traces[-1]['wall_seconds'] if traces else None,
                   versus_initial=paired(initial, result),
                   versus_baseline=paired(control, result) if control else None,
                   first_metrics=next((r['metrics'] for r in traces if r['metrics']), {}),
                   last_metrics=traces[-1]['metrics'] if traces else {},
                   last_return_scale=max(1., hi - lo) if hi is not None and lo is not None else None,
                   replay_reports=[{k: r.get(k) for k in [
                       'wall_seconds', 'updates', 'replay_old_sample_fraction',
                       'replay_old_cumulative_fraction', 'online_priority_sample_fraction']}
                       for r in reports])
        arms.append(arm)
    output = dict(arms=arms, initial_success_rate=initial['success_rate'],
                  gpu_seconds=state.get('diagnosis', {}).get('gpu_seconds'),
                  skipped=state.get('diagnosis', {}).get('skipped', []),
                  limitations=[
                      'Only one training seed and one fixed 80-episode validation set.',
                      'Initialization and replay restore are included in bounded process time.',
                      'Collector count and repaired reset placements are common to all arms.',
                      'A short intervention cannot uniquely identify the historical cause.'])
    Path('metrics/lwy_r7_diagnosis_analysis.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({**output, 'arms': [{k: v for k, v in arm.items()
        if k not in ['first_metrics', 'last_metrics', 'replay_reports']} for arm in arms]}, indent=2))


if __name__ == '__main__':
    main()
