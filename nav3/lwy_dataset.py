"""Deterministic LWY train/validation/blind variants, with a separate test copy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from .lwy_import import export, Route, ROOT
sys.path.insert(0, str(ROOT/'vendor/lwy'))
from trail_scene import build_bundle


def variant(base, seed):
    rng = np.random.default_rng(seed); spec = copy.deepcopy(base)
    spec['seed'] = seed; spec['name'] = f'lwy_variant_{seed}'
    points = np.array(spec['track']['control_points_xyz'], float)
    # Shift stair starts, tread runs, lateral shape and curvature independently.
    points[:, 1] += rng.normal(0, 2.0, len(points))
    points[:5, 1] = 0
    points[1:5, 0] = np.cumsum([rng.uniform(1.5, 4.5), rng.uniform(5, 8), rng.uniform(1.2, 2.4), rng.uniform(5, 8)])
    points[5:, 0] += points[4, 0]-16.12
    points[5:, 1] *= rng.uniform(.65, 1.4)
    flights = copy.deepcopy(spec['track']['stair_flights'])
    # Two entry flights are required by the upstream cheek-wall generator.
    # Additional flights change count and placement, not only vegetation seed.
    for i in rng.choice(np.arange(7, 27), size=int(rng.integers(0, 3)), replace=False):
        flights.append(dict(label=f'added_{i}', start_anchor=int(i), end_anchor=int(i+1), count=int(rng.integers(12, 23))))
    flights.sort(key=lambda f: f['start_anchor'])
    targets = []
    for flight in flights:
        flight['count'] = int(rng.integers(12, 24))
        targets.append(float(rng.uniform(.12, .16)) * (-1 if flight['label'] == 'final_descent' else 1))
    spec['track']['stair_flights'] = flights
    # Account for Route's 3-D arclength scaling, rather than label raw z as rise.
    scale = .85
    for _ in range(5):
        for flight, rise in zip(flights, targets):
            i, j = flight['start_anchor'], flight['end_anchor']
            change = points[i, 2]+rise*flight['count']/scale-points[j, 2]
            points[j:, 2] += change
        spec['track']['control_points_xyz'] = points.tolist()
        scale = Route(spec).scale
    zones = [copy.deepcopy(base['track']['zones'][i]) for i in rng.permutation(10)]
    lengths = np.array([z['end_m']-z['start_m'] for z in zones])*rng.uniform(.65, 1.4, 10)
    limits = np.r_[0, np.cumsum(lengths)/lengths.sum()*1000]
    limits[-1] = 1000.  # zone_at requires exact inclusion of the terminal station.
    mirror = bool(rng.integers(2))
    for i, z in enumerate(zones):
        z['start_m'], z['end_m'] = float(limits[i]), float(limits[i+1])
        z['width_m'] = float(max(spec['track']['minimum_width_m'], rng.uniform(1., 1.35)))
        z['vegetation_density'] = float(rng.uniform(.45, 1.))
        z['rail_sides'] = [int(rng.choice([-1, 1]))] if z['rail'] else []
        if mirror:
            z['rail_sides'] = [-x for x in z['rail_sides']]
            z['left_bank_slope'], z['right_bank_slope'] = z['right_bank_slope'], z['left_bank_slope']
    spec['track']['zones'] = zones
    if mirror: points[:, 1] *= -1
    spec['track']['control_points_xyz'] = points.tolist()
    spec['track']['landmarks'] = [dict(s_m=float(s), label=f'variant_sign_{i}') for i, s in enumerate(sorted(rng.uniform(30, 970, 3)))]
    spec['lwy_variation'] = dict(seed=seed, mirrored=mirror, rock_density=float(rng.uniform(.4, 1.8)),
        root_density=float(rng.uniform(.4, 1.8)), target_stair_rises_m=targets)
    return spec


def vary_props(spec, route, bundle):
    rng = np.random.default_rng(spec['seed']+3000000)
    props = []
    for p in bundle['primitives']:
        typ = p['name'].split('_')[0]
        if typ not in ['root', 'rock']:
            props.append(p); continue
        factor = spec['lwy_variation'][typ+'_density']
        count = int(factor)+(rng.random() < factor % 1)
        station = int(p['name'].split('_')[1])*2.
        for k in range(count):
            q = copy.deepcopy(p); q['name'] += f'_v{k}'
            new_s = float(np.clip(station+rng.uniform(-2.5, 2.5), .5, 999.))
            if typ == 'rock':
                side = int(rng.choice([-1, 1]))
                q['position'] = (route.bank_position(new_s, side*(route.width(new_s)/2+.16))+[0, 0, .13]).tolist()
                q['yaw'] = float(rng.uniform(-np.pi, np.pi))
            else:
                a = route.position(new_s, -.6)+[0, 0, .018]
                b = route.position(min(1000, new_s+.35), .6)+[0, 0, .018]
                q['fromto'] = np.r_[a, b].tolist()
            props.append(q)
    bundle['primitives'] = props


def generate(item):
    split, seed, package, root = item
    directory = Path(root)/f'{split}_{seed}'
    if (directory/'metadata.json').exists():
        old = json.loads((directory/'metadata.json').read_text())
        if old.get('split') == split and hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest() == old['sha256']:
            return old
    base = json.loads((Path(package)/'config/track_spec.json').read_text())
    spec = variant(base, seed); directory.mkdir(parents=True, exist_ok=True)
    (directory/'spec.json').write_text(json.dumps(spec, indent=2))
    route, bundle, _ = build_bundle(spec, directory)
    vary_props(spec, route, bundle)
    (directory/'scene/scene_bundle.json').write_text(json.dumps(bundle, separators=(',', ':')))
    # Full route variants exist independently; only training crops use 250 m.
    rng = np.random.default_rng(seed+7)
    start = float(rng.choice([0, 249, 498, 747])) if split == 'train' else 0.
    end = start+250 if split == 'train' else 1000.
    metadata = export(spec, bundle, directory, start, end)
    metadata.update(split=split, variation=spec['lwy_variation'],
        actual_stair_rises_m=[f['rise'] for f in route.stairs],
        spec_sha256=hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest())
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(name=directory.name, seconds=metadata['seconds'], tiles=metadata['tile_count'])), flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package', default='../../ATEC_Env')
    p.add_argument('--root', default='maps_lwy'); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--train', type=int, default=200)
    a = p.parse_args()
    items = [('train', s, a.package, a.root) for s in range(610000, 610000+a.train)]
    items += [(split, s, a.package, a.root) for split, first in [('val', 620000), ('blind', 630000)] for s in range(first, first+20)]
    with ProcessPoolExecutor(a.workers) as pool: records = list(pool.map(generate, items))
    test = json.loads((Path(a.root)/'test_862428/metadata.json').read_text())
    manifest = dict(schema_version=1, maps=records+[test], generator='vendor/lwy/trail_scene.py plus deterministic nav3 variants',
        source_sha256=hashlib.sha256((ROOT/'vendor/lwy/trail_scene.py').read_bytes()).hexdigest(),
        training_mix=dict(lwy_probability=.75, legacy_probability=.25,
            legacy_root='maps', legacy_split='train', legacy_seeds=list(range(10000, 10200)),
            note='planned per-episode sampling mixture; no training authorized unless G1 passes'),
        test_usage='original and blind only G1 geometry/planner qualification and final evaluation; never model selection')
    (Path(a.root)/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with Path('metrics/lwy_split_manifest.json').open('x') as f: json.dump(manifest, f, indent=2)
    print(json.dumps(dict(completed=True, maps=len(records)+1)), flush=True)


if __name__ == '__main__': main()
