"""Faithful re-implementation of r9 build_scan_patches.build_patch over the whole of
segment A (one dense grid instead of 40 m windows), for diagnosis only.

Only the route changes: the segment-A centreline (scanner z on the centreline) is
used for the 'nearest route station -> expected scanner z' lookup.  All thresholds
are the old ones: ground candidates z in [scanner-2.8, scanner-0.6] with
|normal_z|>0.65, 5x5 median outlier test 0.25 m, 0.2 m interpolation, slope
|grad|<0.8 over a 0.4 m baseline, obstacle = >=2 points 0.22-0.8 m above ground.

Writes data/trail_a/old_proxy/map.npz (+ old_proxy_masks.npz with the separate
criteria) and old_proxy.json.
"""
import time

import numpy as np
from scipy import ndimage as ndi

from nav4.mapping.common import DATA, RES, load_config, signed_distance
from nav4.mapping.grid import CellIndex, band_atlas, load_centreline, load_points, make_grid, save_json


def old_proxy(cells, nz_sorted, scanner_z_cell, shape):
    expected = scanner_z_cell.reshape(-1)[cells.flat]
    candidate = (cells.z >= expected - 2.8) & (cells.z <= expected - .6) & (np.abs(nz_sorted) > .65)
    low = cells.first(cells.z, candidate, fill=np.inf)
    observed = np.isfinite(low)
    distance, indices = ndi.distance_transform_edt(~observed, return_indices=True)
    filled = low[tuple(indices)]
    median = ndi.median_filter(filled, size=5)
    observed &= np.abs(filled - median) < .25
    distance, indices = ndi.distance_transform_edt(~observed, return_indices=True)
    ground = low[tuple(indices)]
    known = distance * RES <= .2
    atlas, relative = band_atlas(cells, ground, known)
    body = (relative > .22) & (relative < .8)
    body_hits = cells.count(body)
    obstacle = (body_hits >= 2) & known
    gx = (np.roll(ground, -2, axis=0) - np.roll(ground, 2, axis=0)) / (4 * RES)
    gy = (np.roll(ground, -2, axis=1) - np.roll(ground, 2, axis=1)) / (4 * RES)
    slope = np.hypot(gx, gy)
    support = known & (slope < .8)
    support[[0, 1, -2, -1], :] = False
    support[:, [0, 1, -2, -1]] = False
    return dict(ground=ground.astype(np.float32), known=known, observed=observed, slope=slope.astype(np.float32),
                support=support, obstacle=obstacle, body_hits=body_hits.astype(np.int32), atlas=atlas)


def main():
    begin = time.time()
    cfg, _ = load_config()
    line = load_centreline()
    grid = make_grid(line['xy'], cfg['grid']['bbox_margin_m'], cfg['grid']['coverage_radius_m'])
    shape, origin = grid['shape'], grid['origin']
    pts = load_points(('x', 'y', 'z', 'nz'))
    cells = CellIndex(pts['x'], pts['y'], pts['z'], origin, shape)
    nz = cells.take(pts['nz']).astype(np.float32)
    scanner_z_cell = line['z_scanner'][grid['near']].astype(np.float32)
    r = old_proxy(cells, nz, scanner_z_cell, shape)
    support_sdf = signed_distance(r['support'])
    obstacle_sdf = signed_distance(r['obstacle'])
    layer = np.zeros((*shape, 7), np.float32)
    layer[..., 0] = r['ground']
    layer[..., 1] = r['ground'] + np.maximum(0, r['atlas'][..., 2:9:3].max(-1))
    layer[..., 2] = support_sdf
    layer[..., 3] = grid['dist'] - 1.
    layer[..., 4] = grid['near'] * .2
    layer[..., 5] = obstacle_sdf
    out = DATA / 'old_proxy'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'map.npz', layers=layer, atlas=r['atlas'].astype(np.float16),
                        origin=origin.astype(np.float32), res=np.float32(RES))
    np.savez_compressed(out / 'old_proxy_masks.npz', known=r['known'], observed=r['observed'],
                        slope=r['slope'], support=r['support'], obstacle=r['obstacle'],
                        body_hits=r['body_hits'], coverage=grid['coverage'])
    save_json(out / 'old_proxy.json', dict(
        note='Faithful re-run of r9 build_scan_patches criteria on the segment-A grid (diagnosis only).',
        shape=list(shape), origin=origin.tolist(), points=int(len(cells.z)),
        thresholds=dict(scanner_ground_interval_m=[.6, 2.8], ground_normal_abs_z_min=.65, median_window=5,
                        median_tolerance_m=.25, interpolate_radius_m=.2, slope_max=.8, slope_baseline_m=.4,
                        body_band_m=[.22, .8], body_min_points=2),
        seconds=time.time() - begin))
    print('old proxy done', shape, time.time() - begin)


if __name__ == '__main__':
    main()
