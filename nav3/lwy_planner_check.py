"""Lock down the actual closed-loop narrow-road failure before adopting a fix."""
import json
import argparse
from pathlib import Path


def read(tag): return json.loads(Path(f'metrics/lwy_{tag}.json').read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dynamic', action='store_true'); a = p.parse_args()
    old = read('diag_single_legacy'); new = read('diag_single_lwy')
    tag = 'g1_original_dynamic' if a.dynamic else 'g1_original_lwy'
    full = read(tag); repeat = read(tag+'_repeat')
    assert old['rows'][0]['progress_m'] < 999 and new['rows'][0]['progress_m'] >= 999
    assert all(r['progress_m'] >= 999 for r in full['rows']), 'The full-route premature failure remains'
    assert full['rows'] == repeat['rows'] and full['runtime'] == repeat['runtime']
    result = dict(passed=True, adopted_for_lwy_g1=True, single_seed=940004,
        baseline_single_failure_s=old['rows'][0]['progress_m'], fixed_single_failure_s=new['rows'][0]['progress_m'],
        baseline_full_early_failures=sum(r['progress_m'] < 999 for r in read('g1_original_legacy')['rows']),
        fixed_full_early_failures=sum(r['progress_m'] < 999 for r in full['rows']),
        fixed_full_exact_repeat=True, remaining_endpoint_failures=sum(not r['success'] for r in full['rows']),
        only_planner_change='Observed obstacle 3x3 max pooling -> identity; offsets, prediction footprint, dynamics, scores and speed unchanged',
        physical_checker_changed=False, noise_changed=False, endpoint_gate_changed=False)
    output = 'metrics/lwy_planner_fix_dynamic.json' if a.dynamic else 'metrics/lwy_planner_fix.json'
    with Path(output).open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
