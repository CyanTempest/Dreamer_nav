"""Read-only LWY import into sparse world-coordinate 5 cm tiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import shapely as sh

from .lwy_geometry import Geometry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/lwy'))
from trail_scene import Route

RES = .05
BLOCK = 32
HALF_GAP = np.hypot(1.03/42, .58/24)/2
PAD = RES*np.sqrt(2)/2+HALF_GAP+.0002


def project(route, xy):
    ss = np.arange(0, route.length+.025, .05)
    xyz = np.array([np.interp(ss, route.s, route.xyz[:, k]) for k in range(3)]).T
    _, idx = cKDTree(xyz[:, :2]).query(xy, workers=1)
    idx = np.clip(idx, 0, len(ss)-2)
    candidates = np.clip(idx[:, None]+np.array([-1, 0, 1]), 0, len(ss)-2)
    aa, bb = xyz[candidates, :2], xyz[candidates+1, :2]
    v = bb-aa
    u = np.clip(((xy[:, None]-aa)*v).sum(-1)/np.maximum((v*v).sum(-1), 1e-12), 0, 1)
    error = ((xy[:, None]-aa-u[..., None]*v)**2).sum(-1)
    j = error.argmin(-1); i = np.arange(len(xy))
    return ss[candidates[i, j]]+u[i, j]*.05


def export(spec, bundle, directory, start=0., end=None, gate_stations=None):
    begin = time.monotonic(); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'map.npz').exists()
    route = Route(spec); geometry = Geometry(bundle, route)
    end = route.length if end is None else end
    length = end-start
    ss = np.r_[np.arange(0, length, .2), length]
    points = np.array([route.position(float(s+start)) for s in ss])
    yaws = np.unwrap([np.arctan2(*route.frame(float(s+start))[0][1::-1]) for s in ss])
    widths = np.array([route.width(float(s+start)) for s in ss])
    origin = np.floor((points[:, :2].min(0)-8)/RES)*RES
    # Select whole Cartesian tiles near the route; there is no Frenet image warp.
    tile_ij = np.floor((points[:, :2]-origin)/(RES*BLOCK)).astype(int)
    offsets = np.array([(i, j) for i in range(-5, 6) for j in range(-5, 6)])
    candidates = np.unique((tile_ij[:, None]+offsets).reshape(-1, 2), axis=0)
    candidates = candidates[(candidates >= 0).all(-1)]
    lo = origin+candidates*(RES*BLOCK)
    corridor = sh.LineString(points[:, :2])
    boxes = sh.box(lo[:, 0], lo[:, 1], lo[:, 0]+RES*BLOCK, lo[:, 1]+RES*BLOCK)
    candidates = candidates[sh.distance(boxes, corridor) <= 6.]
    dims = candidates.max(0)+1
    grid = np.stack(np.meshgrid(np.arange(BLOCK), np.arange(BLOCK), indexing='ij'), -1).reshape(-1, 2)
    tiles, stations = [], []
    roots = [p for p in geometry.props if p['name'].startswith('root_')]
    root_poly = None
    if roots:
        from .lwy_geometry import primitive_polygon
        root_poly = sh.union_all([primitive_polygon(p) for p in roots]).buffer(RES*np.sqrt(2)/2)
    for first in range(0, len(candidates), 64):
        ij = candidates[first:first+64]
        xy = (origin+(ij[:, None]*BLOCK+grid)*RES).reshape(-1, 2)
        ground = geometry.ground(xy)
        st = project(route, xy)
        # No extrapolated support before/after the source road; missing terrain
        # outside the source is a finite background for rendering only.
        ground = np.where(np.isfinite(ground), ground, np.interp(st, route.s, route.xyz[:, 2])-2)
        edge = geometry.signed_edge(xy)
        distance, top, typ = geometry.obstacle_layers(xy, ground, PAD)
        # A 2.5 cm root is traversable, but retained in sensing and speed limiting.
        if root_poly is not None:
            typ = np.where(sh.intersects(root_poly, sh.points(xy)), 2, typ)
        # Legacy fall threshold is .025. Offset support by that threshold plus
        # a proved quantization/sampling bound; neither envelope nor threshold shrinks.
        layer = np.stack([ground, top, edge+.025+PAD, edge, st-start, distance, typ], -1)
        tiles.append(layer.reshape(-1, BLOCK, BLOCK, 7).astype(np.float16))
        stations.append((st-start).reshape(-1, BLOCK, BLOCK).astype(np.float32))
        if first % 1024 == 0:
            print(json.dumps(dict(map=directory.name, tiles_done=first+len(ij), tiles_total=len(candidates), seconds=time.monotonic()-begin)), flush=True)
    tiles = np.concatenate(tiles); stations = np.concatenate(stations)
    gates = np.asarray(gate_stations if gate_stations is not None else np.linspace(length/8, length, 8), np.float32)
    assert len(gates) == 8 and np.all(np.diff(gates) > 0)
    np.savez_compressed(directory/'map.npz', tiles=tiles, station_tiles=stations,
        tile_indices=candidates.astype(np.int32), dimensions=dims.astype(np.int32),
        origin=origin.astype(np.float32), res=np.float32(RES), route=points.astype(np.float32),
        s=ss.astype(np.float32), yaw=yaws.astype(np.float32), width=widths.astype(np.float32), gates=gates)
    metadata = dict(name=directory.name, seed=spec['seed'], mini=False, source_start_m=start,
        source_end_m=end, length_m=length, resolution_m=RES, block=BLOCK,
        tile_count=len(tiles), raster_bytes=int(tiles.nbytes+stations.nbytes),
        support='road and stair treads only; banks excluded', seam_tolerance_m=6e-5,
        roots='nonblocking; speed capped at 0.28 m/s; sensing retains capsule top',
        collision='all other primitive XY projections, including low rocks and both rail bars',
        raster_pad_m=PAD, footprint_sampling_half_gap_m=HALF_GAP,
        gates_m=gates.tolist(), seconds=time.monotonic()-begin,
        sha256=hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest())
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', default='../../ATEC_Env')
    p.add_argument('--out', default='maps_lwy/test_862428')
    p.add_argument('--gates', nargs=8, type=float)
    a = p.parse_args(); package = Path(a.package)
    spec = json.loads((package/'config/track_spec.json').read_text())
    bundle = json.loads((package/'scene/scene_bundle.json').read_text())
    result = export(spec, bundle, a.out, gate_stations=a.gates)
    result['split'] = 'test'
    (Path(a.out)/'metadata.json').write_text(json.dumps(result, indent=2))
    manifest = Path(a.out).parent/'manifest.json'
    with manifest.open('x') as f:
        json.dump(dict(maps=[result], generator='LWY original bundle; world tiles', schema_version=1), f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
