"""Version two: whole-scene reflection, exact clearance rejection and end gates.

Legacy generation and maps_lwy are deliberately left intact. Rejected variants
advance to the next unused seed inside their split's exclusive 10,000-seed band.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import shapely as sh

from .lwy_dataset import variant, vary_props, build_bundle
from .lwy_geometry import Geometry
from .lwy_import import Route, export, ROOT

VERSION = 'lwy-v2.1'


def gates(length, offset=1.5, road_after_end=0.):
    assert offset >= 1.5, 'Terminal envelope needs the approved 1.5 m setback'
    result = np.linspace(length/8, length, 8)
    # Crops backed by the full source road may cross their nominal crop end.
    result[-1] -= max(0., offset-road_after_end)
    assert np.all(np.diff(result) > 0)
    return result


def reflect_spec(spec):
    spec = copy.deepcopy(spec)
    xyz = np.asarray(spec['track']['control_points_xyz']); xyz[:, 1] *= -1
    spec['track']['control_points_xyz'] = xyz.tolist()
    for zone in spec['track']['zones']:
        zone['rail_sides'] = [-s for s in zone.get('rail_sides', [])]
        zone['left_bank_slope'], zone['right_bank_slope'] = zone['right_bank_slope'], zone['left_bank_slope']
    return spec


def reflect_scene(bundle, arrays):
    """Apply diag(1,-1,1) to world geometry, including asset handedness.

Plant instances use Rz(yaw) @ diag(1, local_scale_y, 1); changing both yaw
and local handedness reflects every vertex of an asymmetric source asset.
"""
    result = copy.deepcopy(bundle)
    def xyz(value):
        a = np.asarray(value, float).copy(); a[..., 1] *= -1
        return a.tolist()
    for p in result['colliders']:
        p['vertices'] = xyz(p['vertices'])
        p['faces'] = np.asarray(p['faces'])[:, [0, 2, 1]].tolist()
    for p in result['primitives'] + result['plants']:
        for key in ['position', 'axis_x', 'axis_y']:
            if key in p: p[key] = xyz(p[key])
        if 'fromto' in p: p['fromto'] = np.asarray(xyz(np.asarray(p['fromto']).reshape(2, 3))).ravel().tolist()
        if 'yaw' in p: p['yaw'] = -p['yaw']
        if 'asset' in p:
            if 'local_scale_y' in p: p.pop('local_scale_y')
            else: p['local_scale_y'] = -1
    reflected = {}
    for key, value in arrays.items():
        a = value.copy()
        if key.endswith('_vertices'): a[:, 1] *= -1
        if key.endswith('_faces'): a = a[:, [0, 2, 1]]
        reflected[key] = a
    return result, reflected


def clearance_check(spec, bundle, threshold=.10, step=.2, endpoint_margin=1.5):
    assert threshold >= 0 and step > 0
    route = Route(spec); geometry = Geometry(bundle, route)
    # End caps cannot support a finite centered rectangle. Audit every 0.2 m
    # of the executable domain; the excluded caps are explicit in the manifest.
    stations = np.unique(np.r_[np.arange(endpoint_margin, route.length-endpoint_margin, step), route.length-endpoint_margin])
    centers = np.array([route.position(float(s))[:2] for s in stations])
    ts = np.array([route.frame(float(s))[0][:2] for s in stations])
    ns = np.stack([-ts[:, 1], ts[:, 0]], -1)
    local = np.array([[-.515, -.29], [.515, -.29], [.515, .29], [-.515, .29]])
    footprints = sh.polygons(centers[:, None]+local[None, :, :1]*ts[:, None]+local[None, :, 1:]*ns[:, None])
    pair, ed = geometry.edge_tree.query_nearest(footprints, return_distance=True, all_matches=False)
    edge = np.empty(len(stations)); edge[pair[0]] = ed
    covered = sh.covers(geometry.support, footprints)
    edge[~covered] = -1.
    pair, ob = geometry.block_tree.query_nearest(footprints, return_distance=True, all_matches=False)
    obstacle = np.empty(len(stations)); obstacle[pair[0]] = ob
    clearance = np.minimum(edge, obstacle)
    bad = np.flatnonzero(clearance < threshold-1e-9)
    ids = geometry.block_tree.nearest(footprints[bad]) if len(bad) else []
    names = [geometry.props[geometry.block_ids[i]]['name'] for i in ids]
    return dict(passed=not len(bad), threshold_m=threshold, step_m=step,
        envelope_m=[1.03, .58], samples=len(stations), domain_m=[endpoint_margin, route.length-endpoint_margin],
        excluded_end_caps_reason='finite envelope; spawn s=2, terminal gate s=L-1.5',
        minimum_m=float(clearance.min()), support_minimum_m=float(edge.min()), obstacle_minimum_m=float(obstacle.min()),
        bad_stations_m=stations[bad].tolist(), blocking_names=sorted(set(names)),
        support_failures=int(np.sum(edge < threshold-1e-9)))


def repair(spec, bundle, threshold):
    route = Route(spec); history = []; count = 0
    for attempt in range(4):
        check = clearance_check(spec, bundle, threshold)
        history.append(check)
        if check['passed']: break
        if attempt == 3 or check['support_failures']: break
        # Only interfering movable stones are resampled; fixed walls/rails,
        # road width and robot geometry are never changed to pass this audit.
        names = set(check['blocking_names'])
        moved = 0
        for index, p in enumerate(bundle['primitives']):
            if p['name'] not in names or not p['name'].startswith('rock_'): continue
            rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], 45191, attempt, index]))
            nominal = int(p['name'].split('_')[1])*2.
            station = float(np.clip(nominal+rng.uniform(-2.5, 2.5), 1.5, 998.5))
            side = int(rng.choice([-1, 1]))
            lateral = side*(route.width(station)/2+rng.uniform(.24, .40))
            p['position'] = (route.bank_position(station, lateral)+[0, 0, .13]).tolist()
            p['yaw'] = float(rng.uniform(-np.pi, np.pi))
            moved += 1
        count += moved
        if not moved: break
    return check, dict(resampled_objects=count, checks=history)


def generate(item):
    split, requested_seed, package, root, threshold, offset = item
    root = Path(root); directory = root/f'{split}_{requested_seed}'
    if (directory/'metadata.json').exists():
        old = json.loads((directory/'metadata.json').read_text())
        assert old['dataset_version'] == VERSION and old['clearance_check']['threshold_m'] == threshold
        assert hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest() == old['sha256']
        return old
    base = json.loads((Path(package)/'config/track_spec.json').read_text())
    rejected = []; resampled = 0
    for attempt in range(1):
        seed = requested_seed
        final_spec = variant(base, seed)
        mirrored = final_spec['lwy_variation']['mirrored']
        spec = reflect_spec(final_spec) if mirrored else copy.deepcopy(final_spec)
        staging = root/'candidates'/f'{split}_{seed}'
        route, bundle, arrays = build_bundle(spec, staging)
        vary_props(spec, route, bundle)
        check, audit = repair(spec, bundle, threshold)
        resampled += audit['resampled_objects']
        if not check['passed']:
            rejected.append(dict(seed=seed, clearance=check, resampling=audit))
            (staging/'rejected.json').write_text(json.dumps(rejected[-1], indent=2))
            return dict(split=split, seed=seed, rejected=True, clearance=check, resampling=audit)
        if mirrored:
            bundle, arrays = reflect_scene(bundle, arrays)
            check = clearance_check(final_spec, bundle, threshold)
            assert check['passed'], 'Reflection must preserve geometric feasibility'
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'scene').mkdir(exist_ok=True)
        (directory/'spec.json').write_text(json.dumps(final_spec, indent=2))
        bundle['spec_digest'] = hashlib.sha256(json.dumps(final_spec, sort_keys=True).encode()).hexdigest()
        (directory/'scene/scene_bundle.json').write_text(json.dumps(bundle, separators=(',', ':')))
        np.savez_compressed(directory/'scene/trail_surfaces.npz', **arrays)
        start = float(np.random.default_rng(seed+7).choice([0, 249, 498, 747])) if split == 'train' else 0.
        end = start+250 if split == 'train' else 1000.
        metadata = export(final_spec, bundle, directory, start, end, gates(end-start, offset, 1000-end))
        metadata.update(split=split, dataset_version=VERSION, requested_seed=requested_seed,
            variation=final_spec['lwy_variation'], actual_stair_rises_m=[f['rise'] for f in route.stairs],
            spec_sha256=hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest(),
            clearance_check=check, resampling=audit, resampled_objects_total=resampled,
            discarded_variants=len(rejected), rejected_candidates=rejected,
            terminal_gate_offset_m=offset, strict_mirror=mirrored,
            mirror_transform='world diag(1,-1,1), reversed face winding, plant local_scale_y=-1')
        (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
        shutil.rmtree(staging)  # Only this new candidate's redundant generation copy.
        print(json.dumps(dict(completed=directory.name, accepted_seed=seed, minimum_m=check['minimum_m'], rejected=len(rejected), resampled=resampled)), flush=True)
        return metadata
    raise AssertionError('Unreachable candidate state')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', default='../../ATEC_Env'); p.add_argument('--root', default='maps_lwy_v2')
    p.add_argument('--workers', type=int, default=8); p.add_argument('--clearance', type=float, default=.10)
    p.add_argument('--terminal-offset', type=float, default=1.5)
    a = p.parse_args(); root = Path(a.root); root.mkdir(exist_ok=True)
    test_dir = root/'test_862428'
    if not (test_dir/'metadata.json').exists():
        spec = json.loads((Path(a.package)/'config/track_spec.json').read_text())
        bundle = json.loads((Path(a.package)/'scene/scene_bundle.json').read_text())
        test = export(spec, bundle, test_dir, gate_stations=gates(1000, a.terminal_offset))
        test.update(split='test', dataset_version=VERSION, terminal_gate_offset_m=a.terminal_offset,
                    source_geometry_unmodified=True)
        (test_dir/'metadata.json').write_text(json.dumps(test, indent=2))
    test = json.loads((test_dir/'metadata.json').read_text())
    # Consecutive seeds in each original split band; accept exactly the first N
    # feasible variants. The acceptance decision never reads planner results.
    records=[]; rejected=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for split, first, n in [('blind',630000,20),('val',620000,20),('train',610000,200)]:
            accepted=[]; next_seed=first
            while len(accepted)<n:
                count=n-len(accepted)
                assert next_seed+count<=first+10000, 'Split seed band exhausted'
                items=[(split,s,a.package,a.root,a.clearance,a.terminal_offset) for s in range(next_seed,next_seed+count)]
                for result in pool.map(generate,items):
                    (rejected if result.get('rejected') else accepted).append(result)
                next_seed+=count
            records.extend(accepted)
    manifest = dict(schema_version=2, dataset_version=VERSION, maps=records+[test],
        generator='nav3.lwy_dataset_v2', clearance_threshold_m=a.clearance, terminal_gate_offset_m=a.terminal_offset,
        candidate_seed_rule='next consecutive unused seed from 610000/620000/630000; exclusive 10000-seed split bands',
        discarded_variants=len(rejected), rejected_candidates=rejected,
        resampled_objects=sum(r['resampled_objects_total'] for r in records)+sum(r['resampling']['resampled_objects'] for r in rejected),
        source_sha256=hashlib.sha256((ROOT/'vendor/lwy/trail_scene.py').read_bytes()).hexdigest(),
        training_mix=dict(lwy_probability=.75, legacy_probability=.25, legacy_root='maps', legacy_split='train', legacy_seeds=list(range(10000,10200))),
        test_usage='G1 and final evaluation only; never selection or training')
    with (root/'manifest.json').open('x') as f: json.dump(manifest, f, indent=2)
    with Path('metrics/lwy_v2_split_manifest.json').open('x') as f: json.dump(manifest, f, indent=2)
    print(json.dumps(dict(completed=True, maps=len(records)+1, discarded=manifest['discarded_variants'], resampled=manifest['resampled_objects'])), flush=True)


if __name__ == '__main__': main()
