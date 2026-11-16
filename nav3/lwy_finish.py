"""Summarize observed G1 outcomes; never converts a near-finish into success."""
import json
from pathlib import Path


def main():
    from .lwy_eval import protocol_hash
    manifest = json.loads(Path('metrics/lwy_split_manifest.json').read_text())
    summary = {}
    for split in ['train', 'val', 'blind', 'test']:
        maps = [r for r in manifest['maps'] if r['split'] == split]
        specs = [json.loads((Path('maps_lwy')/r['name']/'spec.json').read_text()) for r in maps if split != 'test']
        summary[split] = dict(count=len(maps), seeds=[r['seed'] for r in maps],
            raster_bytes=sum(r['raster_bytes'] for r in maps), total_crop_length_m=sum(r['length_m'] for r in maps))
        if specs:
            widths = [z['width_m'] for spec in specs for z in spec['track']['zones']]
            rises = [abs(v) for r in maps for v in r['actual_stair_rises_m']]
            summary[split].update(mirrored=sum(s['lwy_variation']['mirrored'] for s in specs),
                width_range_m=[min(widths), max(widths)], stair_rise_range_m=[min(rises), max(rises)],
                stair_flight_count_range=[min(len(s['track']['stair_flights']) for s in specs), max(len(s['track']['stair_flights']) for s in specs)],
                unique_zone_orders=len({tuple(z['label'] for z in s['track']['zones']) for s in specs}))
    with Path('metrics/lwy_dataset_summary.json').open('x') as f: json.dump(summary, f, indent=2)
    original = json.loads(Path('metrics/lwy_g1_original_dynamic.json').read_text())
    repeat = json.loads(Path('metrics/lwy_g1_original_dynamic_repeat.json').read_text())
    blind = json.loads(Path('metrics/lwy_g1_blind_dynamic.json').read_text())
    assert original['rows'] == repeat['rows'], 'Deterministic full-route repeat differs'
    assert original['runtime'] == repeat['runtime']
    successes = sum(r['success'] for r in original['rows'])
    gate = dict(passed=successes == 10 and blind['success_rate'] >= .95,
        original_episodes=10, original_successes=successes, original_mean_gates=original['mean_gates'],
        blind_episodes=blind['episodes'], blind_success_rate=blind['success_rate'], blind_mean_gates=blind['mean_gates'],
        deterministic_repeat_exact=True, repeat_compared_fields='all rows including every-step action/pose SHA256',
        original_failure_s_m=[r['progress_m'] for r in original['rows']],
        blind_failure_s_m=[r['progress_m'] for r in blind['rows']],
        original_source='metrics/lwy_g1_original_dynamic.json', blind_source='metrics/lwy_g1_blind_dynamic.json',
        rl_started=False, checkpoint_copies_created=False,
        environment_sha256=protocol_hash(),
        blocked_reason='last gate at s=1000 requires robot center on the physical road boundary; no footprint fits')
    for label, result in [('original', original), ('blind', blind)]:
        gate[label+'_before_999m_failures'] = [dict(map=r['map'], noise_seed=r['noise_seed'], s=r['progress_m'], failure=r['failure'])
            for r in result['rows'] if not r['success'] and r['progress_m'] < 999.]
        gate[label+'_near_finish_7gate_runs'] = sum(r['gates'] == 7 and r['progress_m'] >= 999. for r in result['rows'])
        histogram = {}
        for row in result['rows']:
            spec_path = Path('../../ATEC_Env/config/track_spec.json') if label == 'original' else Path('maps_lwy')/row['map']/'spec.json'
            zones = json.loads(spec_path.read_text())['track']['zones']
            index = next((i for i, zone in enumerate(zones) if row['progress_m'] < zone['end_m']), 9)
            key = f"{index+1}:{zones[index]['label']}:{row['failure']}"
            histogram[key] = histogram.get(key, 0)+1
        gate[label+'_failure_by_zone'] = histogram
    with Path('metrics/lwy_g1.json').open('x') as f: json.dump(gate, f, indent=2)
    print(json.dumps(gate), flush=True)


if __name__ == '__main__': main()
