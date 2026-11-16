"""Round-six isolation and geometry regression; independent of G1 acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--protected', default='metrics/lwy_protected_remote.json')
    p.add_argument('--v2', action='store_true')
    p.add_argument('--output', default='metrics/lwy_checks.json')
    a = p.parse_args()
    from nav3.provenance import fingerprint
    assert fingerprint() == 'a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d'
    manifest = json.loads(Path('metrics/lwy_split_manifest.json').read_text())
    records = manifest['maps']; sets = {split: {r['seed'] for r in records if r['split'] == split} for split in ['train', 'val', 'blind', 'test']}
    assert len(sets['train']) >= 200 and len(sets['val']) == len(sets['blind']) == 20
    assert sets['test'] == {862428}
    for s, group in sets.items():
        for t, other in sets.items():
            if s != t: assert not group & other, (s, t)
    old = json.loads(Path('maps/manifest.json').read_text())['maps']
    legacy_train = {r['seed'] for r in old if r['split'] == 'train'}
    train = sets['train'] | set(manifest['training_mix']['legacy_seeds'])
    assert set(manifest['training_mix']['legacy_seeds']) <= legacy_train
    forbidden = sets['test'] | sets['val'] | sets['blind'] | {871002} | set(range(30000, 30004))
    forbidden |= {r['seed'] for r in old if r['split'] != 'train'}
    assert not train & forbidden and all(not r['mini'] for r in records)
    for r in records:
        directory = Path('maps_lwy')/r['name']
        assert digest(directory/'map.npz') == r['sha256'], r['name']
        with np.load(directory/'map.npz') as data:
            assert data['res'] <= .05+1e-8 and len(data['gates']) == 8
            assert data['width'].min() >= 1.05-1e-6
            assert data['tiles'].shape[1:] == (32, 32, 7)
            assert np.isfinite(data['tiles']).all()
        if r['split'] != 'test':
            assert digest(directory/'spec.json') == r['spec_sha256']
            assert all(.12-1e-5 <= abs(x) <= .16+1e-5 for x in r['actual_stair_rises_m'])
    geometry = json.loads(Path('metrics/lwy_geometry.json').read_text())
    assert geometry['envelope_m'] == [1.03, .58] and geometry['unsafe_false_free'] == 0
    assert geometry['probe_count'] >= 3000 and not geometry['bev']['changed']
    random_check = json.loads(Path('metrics/lwy_random_obstacles.json').read_text())
    assert random_check['passed'] and random_check['active_obstacles'] == 48
    planner_fix = json.loads(Path('metrics/lwy_planner_fix_dynamic.json').read_text())
    assert planner_fix['passed'] and planner_fix['fixed_full_early_failures'] == 0
    assert planner_fix['baseline_full_early_failures'] == 3 and planner_fix['fixed_full_exact_repeat']
    canonical = json.loads(Path('metrics/lwy_canonical_specs.json').read_text())
    assert canonical['passed'] and all(r['raster_arrays_identical'] and r['raster_file_unchanged'] for r in canonical['rows'])
    gate = json.loads(Path('metrics/lwy_g1.json').read_text())
    assert gate['passed'] == (gate['original_successes'] == 10 and gate['blind_success_rate'] >= .95)
    if not gate['passed']:
        from nav3.lwy_eval import require_g1
        try: require_g1()
        except AssertionError: pass
        else: raise AssertionError('G1 failure must block training')
        if not Path('metrics/lwy_v2_g1.json').exists() or not json.loads(Path('metrics/lwy_v2_g1.json').read_text())['passed']:
            assert not Path('checkpoints/dreamer_lwy').exists() and not Path('checkpoints/ppo_lwy').exists()
    protected = json.loads(Path(a.protected).read_text())
    changed = [f for f, h in protected.items() if not Path(f).is_file() or digest(f) != h]
    assert not changed, changed
    added = [str(f) for directory in ['maps', 'maps_v2', 'maps_mix_v2'] for f in Path(directory).rglob('*') if f.is_file() and str(f) not in protected]
    assert not added, added
    result = dict(passed=True, protected_files=len(protected), protected_changed=[], protected_maps_added=[],
        split_sizes={k: len(v) for k, v in sets.items()}, mini_in_train=False, test_in_train=False,
        validation_blind_in_train=False, old_environment_fingerprint=fingerprint(), g1_passed=gate['passed'])
    if a.v2:
        result['v2'] = check_v2()
        result['legacy_g1_passed'] = result['g1_passed']
        current = json.loads(Path('metrics/lwy_v2_g1.json').read_text())
        from nav3.lwy_eval_v2 import protocol_hash
        assert current['passed'] == (current['original_successes'] == 10 and current['blind_success_rate'] >= .95)
        assert current['environment_sha256'] == protocol_hash()
        result['g1_passed'] = current['passed']
        result['g1_protocol_current'] = True
    output = Path(a.output)
    if not output.exists():
        with output.open('x') as f: json.dump(result, f, indent=2)
    else:
        assert json.loads(output.read_text()) == result, 'Existing audit result differs; use a new output artifact'
    print(json.dumps(result), flush=True)


def check_v2():
    from nav3.lwy_dataset_v2 import gates, reflect_scene, clearance_check, VERSION
    from nav3.lwy_geometry import primitive_polygon
    from shapely import affinity
    import shapely as sh
    root = Path('maps_lwy_v2')
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest == json.loads(Path('metrics/lwy_v2_split_manifest.json').read_text())
    assert manifest['dataset_version'] == VERSION
    assert np.array_equal(gates(1000), [125,250,375,500,625,750,875,998.5])
    assert gates(1000,2.)[-1] == 998.
    assert gates(250,road_after_end=0)[-1] == 248.5
    assert gates(250,road_after_end=3)[-1] == 250
    sets = {s: set() for s in ['train','val','blind','test']}
    rejected = {s: set() for s in sets}
    for r in manifest['rejected_candidates']:
        rejected[r['split']].add(r['seed'])
    assert manifest['discarded_variants'] == sum(len(s) for s in rejected.values())
    for r in manifest['maps']:
        sets[r['split']].add(r['seed'])
        assert r['dataset_version'] == VERSION
        d = root/r['name']
        assert digest(d/'map.npz') == r['sha256']
        with np.load(d/'map.npz') as data:
            expected = gates(r['length_m'],manifest['terminal_gate_offset_m'],1000-r['source_end_m'])
            assert np.array_equal(data['gates'],expected.astype(np.float32))
        if r['split'] == 'test':
            # Original source geometry/raster remain bytewise-array identical.
            with np.load(d/'map.npz') as data, np.load('maps_lwy/test_862428/map.npz') as old:
                assert all(np.array_equal(data[k],old[k]) for k in old.files if k!='gates')
            continue
        check=r['clearance_check']
        assert check['passed'] and check['minimum_m']>=manifest['clearance_threshold_m']-1e-9
        assert check['step_m']==.2 and check['envelope_m']==[1.03,.58]
        assert check['samples']>=4985 and check['domain_m']==[1.5,998.5]
        rejected[r['split']].update(x['seed'] for x in r['rejected_candidates'])
        assert r['discarded_variants']==len(r['rejected_candidates'])
    assert len(sets['train'])>=200 and len(sets['val'])==len(sets['blind'])==20 and sets['test']=={862428}
    for s in sets:
        assert not sets[s]&rejected[s]
        for t in sets:
            if s!=t:assert not (sets[s]|rejected[s])&(sets[t]|rejected[t])
    for split,first in [('train',610000),('val',620000),('blind',630000)]:
        assert sets[split]|rejected[split] == set(range(first,max(sets[split])+1))
        assert max(sets[split])<first+10000
    assert not sets['train']&({871002,862428}|set(range(30000,30004)))
    record=next(r for r in manifest['maps'] if r.get('strict_mirror'))
    d=root/record['name'];bundle=json.loads((d/'scene/scene_bundle.json').read_text())
    arrays=dict(np.load(d/'scene/trail_surfaces.npz'))
    reflected,ra=reflect_scene(bundle,arrays);twice,ta=reflect_scene(reflected,ra)
    assert twice==bundle and all(np.array_equal(arrays[k],ta[k]) for k in arrays)
    reflection=np.diag([1.,-1.,1.])
    for p,q in zip(bundle['colliders'],reflected['colliders']):
        assert np.array_equal(np.asarray(p['vertices'])@reflection,np.asarray(q['vertices']))
        assert np.array_equal(np.asarray(p['faces'])[:,[0,2,1]],q['faces'])
    for key,value in arrays.items():
        if key.endswith('_vertices'):assert np.array_equal(value@reflection,ra[key])
        if key.endswith('_faces'):assert np.array_equal(value[:,[0,2,1]],ra[key])
    # Asymmetric asset vertices require reflected handedness as well as yaw.
    def asset_matrix(p):
        c,s=np.cos(p['yaw']),np.sin(p['yaw'])
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])@np.diag([1,p.get('local_scale_y',1),1])
    for p,q in zip(bundle['plants'],reflected['plants']):
        assert np.allclose(reflection@asset_matrix(p),asset_matrix(q),rtol=0,atol=1e-12)
        assert np.array_equal(np.asarray(p['position'])@reflection,q['position'])
    # Check actual geometric reflection, in addition to the involution property.
    for p,q in zip(bundle['primitives'],reflected['primitives']):
        expected=affinity.scale(primitive_polygon(p),xfact=1,yfact=-1,origin=(0,0))
        assert expected.hausdorff_distance(primitive_polygon(q))<1e-8
    spec=json.loads((d/'spec.json').read_text())
    good=clearance_check(spec,bundle,manifest['clearance_threshold_m'])
    assert good['passed']
    assert not clearance_check(spec,bundle,good['minimum_m']+.001)['passed']
    # Every old LWY file is immutable, including surface and spec files.
    old=json.loads(Path('metrics/lwy_v2_old_maps.json').read_text())
    assert all(Path(f).is_file() and digest(f)==h for f,h in old.items())
    assert {str(p) for p in Path('maps_lwy').rglob('*') if p.is_file()}==set(old)
    return dict(passed=True,version=VERSION,split_sizes={s:len(v) for s,v in sets.items()},
                mirror_involution=True,mirror_projected_geometry=True,clearance_threshold_enforced=True,
                endpoint_offset_m=manifest['terminal_gate_offset_m'],old_lwy_files_unchanged=len(old))


if __name__ == '__main__': main()
