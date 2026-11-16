"""Independent polygon-vs-raster clearance audit for the fixed 1 km test scene."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .lwy_import import Route, RES
from .lwy_geometry import Geometry


def intervals(s, mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1)
    return [[float(s[g[0]]), float(s[g[-1]])] for g in groups]


def stats(x):
    return dict(zip(['min', 'p01', 'p05', 'median', 'p95', 'max'], np.quantile(x, [0, .01, .05, .5, .95, 1]).tolist()))


def sample(a, xy):
    ij = np.floor((xy-a['origin'])/a['res']+.5).astype(int)
    tile = ij//32
    dims = a['dimensions']; lookup = np.zeros(tuple(dims), int)
    indices = a['tile_indices']; lookup[indices[:, 0], indices[:, 1]] = np.arange(len(indices))+1
    valid = ((tile >= 0)&(tile < dims)).all(-1)
    ti = lookup[np.clip(tile[..., 0], 0, dims[0]-1), np.clip(tile[..., 1], 0, dims[1]-1)]
    valid &= ti > 0
    values = a['tiles'][np.maximum(ti-1, 0), ij[..., 0]%32, ij[..., 1]%32].astype(float)
    values[~valid, 2:4] = 20; values[~valid, 5] = 100
    return values


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package', default='../../ATEC_Env')
    p.add_argument('--map', default='maps_lwy/test_862428/map.npz')
    a = p.parse_args(); root = Path(a.package)
    spec = json.loads((root/'config/track_spec.json').read_text())
    bundle = json.loads((root/'scene/scene_bundle.json').read_text())
    route = Route(spec); g = Geometry(bundle, route); data = dict(np.load(a.map))
    ss = np.linspace(0, 1000, 5001)
    pose = np.array([route.position(float(s))[:2] for s in ss])
    yaw = np.array([np.arctan2(*route.frame(float(s))[0][1::-1]) for s in ss])
    exact = [g.clearance(g.footprint(x, y)) for x, y in zip(pose, yaw)]
    edge, obstacle = np.array([x[:2] for x in exact]).T
    feasible = np.array([x[2] for x in exact])
    xx, yy = np.meshgrid(np.linspace(-.515, .515, 43), np.linspace(-.29, .29, 25), indexing='ij')
    foot = np.stack([xx, yy], -1).reshape(-1, 2)
    def raster_margins(poses, yaws):
        all_edge, all_ob = [], []
        for first in range(0, len(poses), 64):
            xy, theta = poses[first:first+64], yaws[first:first+64]
            rot = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], -1).reshape(-1, 2, 2)
            fp = xy[:, None]+np.einsum('bij,kj->bki', rot, foot)
            values = sample(data, fp.reshape(-1, 2)).reshape(len(xy), -1, 7)
            all_edge.extend(.025-values[..., 2].max(-1)); all_ob.extend(values[..., 5].min(-1))
        return np.array(all_edge), np.array(all_ob)
    re, ro = raster_margins(pose, yaw)
    # Independent safety probes include orientations and lateral boundary cases,
    # not merely easy centerline points.
    rng = np.random.default_rng(930601)
    probe_s = rng.uniform(1, 999, 3000)
    lat = rng.uniform(-.65, .65, len(probe_s)); angle = rng.uniform(-.45, .45, len(probe_s))
    probe_xy = np.array([route.position(float(s), float(d))[:2] for s, d in zip(probe_s, lat)])
    probe_yaw = np.array([np.arctan2(*route.frame(float(s))[0][1::-1]) for s in probe_s])+angle
    probe_exact = np.array([g.clearance(g.footprint(x, y))[2] for x, y in zip(probe_xy, probe_yaw)])
    pe, po = raster_margins(probe_xy, probe_yaw)
    probe_raster = (pe >= 0)&(po >= 0)
    interior = (ss >= 1)&(ss <= 999)
    out = dict(resolution_m=RES, envelope_m=[1.03, .58], sample_step_m=.2,
        support_clearance_m=stats(edge), obstacle_clearance_m=stats(obstacle),
        interior_min_clearance_m=float(np.minimum(edge, obstacle)[interior].min()),
        below_5cm_intervals_m=intervals(ss, np.minimum(edge, obstacle) < .05),
        exact_infeasible_intervals_m=intervals(ss, ~feasible),
        raster_infeasible_intervals_m=intervals(ss, (re < 0)|(ro < 0)),
        support_conservatism_m=stats(edge[interior]-re[interior]),
        obstacle_conservatism_m=stats(obstacle[interior]-ro[interior]),
        probe_count=len(probe_s), unsafe_false_free=int(np.sum(probe_raster&~probe_exact)),
        conservative_rejections=int(np.sum(~probe_raster&probe_exact)),
        probe_exact_free=int(probe_exact.sum()), probe_raster_free=int(probe_raster.sum()),
        gate_endpoint=dict(last_gate_m=1000, source_road_end_m=1000,
            center_crossing_required=True, support_extension_m=0,
            conclusion='center must reach the physical road boundary; a nonzero rectangular footprint cannot be supported there'),
        bev=dict(shape=[64, 64, 6], resolution_m=.125, width_cells_1_1m=1.1/.125,
            lateral_margin_cells=(1.1-.58)/2/.125,
            alternative='128x128 at 0.0625m with same 8m view; 4x image pixels and replay image bytes; requires both models from scratch',
            changed=False))
    with Path('metrics/lwy_geometry.json').open('x') as f: json.dump(out, f, indent=2)
    np.savez_compressed('metrics/lwy_clearance_samples.npz', s=ss, pose=pose, yaw=yaw,
        support=edge, obstacle=obstacle, raster_support=re, raster_obstacle=ro, feasible=feasible)
    print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
