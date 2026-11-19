"""Build a reproducible bank of CPU-validated starts using full_v2's own fitter.
Run from the private source checkout with PYTHONDONTWRITEBYTECODE=1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from nav4.matrix.collision_spawn import prepare, stable
from nav4.matrix.full_scan import route
from nav4.matrix.stairs import heading, path_at


def tread_center(point, direction, polygons):
    # For a convex tread intersect the route-parallel line with both risers.
    side = np.array([-direction[1], direction[0]])
    for polygon in polygons:
        if np.any(point < polygon.min(0)) or np.any(point > polygon.max(0)):
            continue
        relative = polygon - point
        u, v = relative @ direction, relative @ side
        other = np.roll(v, -1)
        hits = (v > 0) != (other > 0)
        if hits.sum() == 2:
            cross = u[hits] - v[hits] * (np.roll(u, -1)-u)[hits] / (other-v)[hits]
            if cross.min() <= 0 <= cross.max():
                return point + direction * cross.mean(), True
    return point, False


def build(window, count, out, seed):
    base = Path('data/trail_a_mujoco/full_v2')
    spec = json.loads((base/'assembly.json').read_text())
    row = next(r for r in spec['statics'] if Path(r['window']).stem == window)
    polygons = [np.array(p['vertices'])[4:8, :2] for p in json.loads((base/'stair_geometry.json').read_text())]
    _, path = route()
    shift = np.array(spec['shared_shift_m'])
    rng = np.random.default_rng(seed)
    records = []
    for attempt in range(count*12):
        station = rng.uniform(row['station_m'][0]+.8, min(row['station_m'][1], 722)-1)
        unit = heading(path, station)
        side = np.array([-unit[1], unit[0]])
        offset = rng.uniform(-.2, .2)
        delta = rng.uniform(-np.pi/18, np.pi/18)
        p = path_at(path, station)
        pos, tread = tread_center(p+side*offset, unit, polygons)
        rotation = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
        # prepare() evaluates the transformed path only at the three local stations.
        try:
            pose = prepare(window, station, (path-p) @ rotation.T + pos, shift, False, tread)
        except (ValueError, AssertionError, FloatingPointError):
            continue
        if stable(pose['stand']):
            records.append(dict(pose, station_m=float(station + (pos-p) @ unit),
                                lateral_m=offset, yaw_error_rad=delta, tread=tread, fit_seed=seed, attempt=attempt))
            tmp = out.with_suffix('.tmp')
            tmp.write_text(json.dumps(records, default=lambda x: x.tolist()))
            tmp.replace(out)
            if len(records) == count:
                return
    raise RuntimeError(f'{window}: only {len(records)}/{count} supported starts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slot', type=int)
    parser.add_argument('out', type=Path)
    parser.add_argument('--count', type=int, default=48)
    args = parser.parse_args()
    build(f'window_{args.slot:02}', args.count, args.out, 20261119+args.slot)
