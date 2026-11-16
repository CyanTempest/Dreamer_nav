"""Build the segment-A static simulation map (task 3/4).

Inputs : data/trail_a/segment/centreline.npz, data/trail_a/cache/crop/*.npy,
         nav4/mapping/trail_map_config.json
Outputs: data/trail_a/forward/{map.npz,aux.npz,metadata.json}
         data/trail_a/reverse/{map.npz,aux.npz,metadata.json}   (same geometry, reversed route)
         data/trail_a/manifest.json, data/trail_a/cache/model_cells.npz (diagnostics)

map.npz keys/semantics follow r9 build_scan_patches.py, except: layers float32,
layers[...,6] = obstacle type flag (0 none, 1 hard/colliding, 2 soft-only i.e.
visible but non-colliding), atlas float32 (channel 0 == layers[...,0]), gates =
8 evenly spaced stations ending at length-1 m, width = measured feasible width
clipped to [1,3] m, layers[...,3] = distance to centreline - width/2.

Usage: python -m nav4.mapping.build_trail_map [--no-reverse]
"""
import argparse
import hashlib
import json
import resource
import time

import numpy as np
from scipy import ndimage as ndi

from nav4.mapping.analysis import corridor_path, feasible_width, lateral_feasibility, validate_path
from nav4.mapping.common import DATA, PLANNING_PLY, PLANNING_SHA256, RES, load_config, load_poses, write_json
from nav4.mapping.grid import CellIndex, load_centreline, load_points, make_grid
from nav4.mapping.core import run_model
from nav4.mapping.ground import device_height
from nav4.mapping.model import lowest_ground_layer


def log(msg, t0):
    print(f'[{time.time() - t0:7.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB] {msg}', flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pose_speed(rows):
    sp = np.r_[0, np.linalg.norm(np.diff(rows[:, 1:3], axis=0), axis=1) / np.diff(rows[:, 0])]
    return ndi.median_filter(sp, size=5)


def build(cfg, t0):
    line = load_centreline()
    rows = load_poses()
    s, xy, yaw, zs = line['s'], line['xy'], line['yaw'], line['z_scanner']
    grid = make_grid(xy, cfg['grid']['bbox_margin_m'], cfg['grid']['coverage_radius_m'])
    origin, shape, coverage = grid['origin'], grid['shape'], grid['coverage']
    log(f'grid {shape} origin {origin}', t0)
    pts = load_points(('x', 'y', 'z', 'nz', 'cls'))
    cells = CellIndex(pts['x'], pts['y'], pts['z'], origin, shape)
    absnz = np.abs(cells.take(pts['nz']).astype(np.float32))
    cls = cells.take(pts['cls'])
    log(f'indexed {len(cells.z)} points', t0)

    walked = rows[line['walked_index']]
    speed = pose_speed(rows)[line['walked_index']]
    moving = speed >= cfg['ground']['device_height_min_speed_mps']
    g = cfg['ground']
    z0, n0 = lowest_ground_layer(cells, cfg)
    dh, _ = device_height(z0, origin, walked[moving, 1:4], g['device_height_radius_m'])
    ok = np.isfinite(dh)
    h_dev = float(np.median(dh[ok]))
    dev_stats = dict(estimate_m=h_dev, poses_used=int(ok.sum()), poses_moving=int(moving.sum()),
                     quantiles_m={str(q): float(np.percentile(dh[ok], q)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
                     robust_sigma_m=float((np.percentile(dh[ok], 75) - np.percentile(dh[ok], 25)) / 1.349),
                     all_poses_quantiles_m={str(q): float(np.nanpercentile(device_height(z0, origin, walked[:, 1:4], g['device_height_radius_m'])[0], q)) for q in (1, 5, 50, 95, 99)},
                     method='scanner z minus median lowest-layer ground within 0.2 m, moving poses (>=0.3 m/s)')
    log(f'device height {h_dev:.3f} m', t0)
    lateral_yaw = yaw[grid['near']]
    r = run_model(cells, cells.take(pts['x']), cells.take(pts['y']), absnz, cls, origin, shape, coverage, z0, n0,
                  walked[moving, 1:4], walked[:, 1:3], lateral_yaw, h_dev, cfg, log=lambda msg: log(msg, t0))
    ground, known, atlas, gr, feat, obs = r['ground'], r['known'], r['atlas'], r['gr'], r['feat'], r['obs']
    grade, step, cap_ok, cross, along = r['grade'], r['step'], r['cap_ok'], r['cross'], r['along']
    zone_conn, dist_walk, exempt, support, support_cap = r['zone_conn'], r['dist_walk'], r['exempt'], r['support'], r['support_cap']
    corridor, hard, soft, flag, sdf, wmask = r['corridor'], r['hard'], r['soft'], r['flag'], r['sdf'], r['wmask']
    wc = cfg['walked_corridor']
    near = grid['near']
    # route z = map ground at the route cell (old contract: inferred surface, never scanner z)
    ri = np.floor((xy - origin) / RES + .5).astype(int)
    route = np.c_[xy, ground[ri[:, 0], ri[:, 1]]].astype(np.float32)
    c = cfg['corridor']
    offsets = np.arange(-c['lateral_search_m'], c['lateral_search_m'] + 1e-9, .05)
    feas = lateral_feasibility(sdf['support'], sdf['obstacle_c' if wc['apply_in_saved_map'] else 'obstacle_b'],
                               origin, xy, yaw, offsets)
    fw = feasible_width(feas, offsets, c['acceptance_band_m'])
    width = np.clip(fw, *c['width_clip_m']).astype(np.float32)
    zero = int(np.argmin(np.abs(offsets)))
    band = np.abs(offsets) <= c['acceptance_band_m'] + 1e-9
    connected, reach, path, last = corridor_path(feas, offsets, int(round(c['dp_max_shift_per_station_m'] / .05)),
                                                 c['acceptance_band_m'])
    swept = bool(validate_path(sdf['support'], sdf['obstacle_c' if wc['apply_in_saved_map'] else 'obstacle_b'],
                               origin, xy, yaw, path)['swept_ok']) if connected else False
    qual = dict(centre=feas[:, zero], band=feas[:, band].any(1), connected=bool(connected), swept=swept,
                reach_s=float(s[last]) if last >= 0 else 0.)
    log(f'width done; continuous={connected} swept={swept} reach={qual["reach_s"]:.1f} m', t0)

    layers = np.zeros((*shape, 7), np.float32)
    layers[..., 0] = ground
    layers[..., 1] = ground + np.maximum(0, atlas[..., 2:9:3].max(-1))
    layers[..., 2] = sdf['support']
    layers[..., 3] = grid['dist'] - width[near] / 2
    layers[..., 4] = s[near]
    layers[..., 5] = sdf['obstacle_c' if wc['apply_in_saved_map'] else 'obstacle_b']
    layers[..., 6] = flag
    aux = dict(known=known, observed=gr['observed'], prior_fill=gr['prior_fill'], veg_top=gr['veg_top'],
               sub_surface=gr['sub_surface'], connected=gr['connected'],
               spike=gr['spike'], small_island=gr['small_island'], coverage=coverage, z0=z0,
               grade_deg=grade.astype(np.float32), step_m=step, cross_grade_deg=cross.astype(np.float32),
               along_grade_deg=along.astype(np.float32), cap_ok=cap_ok, trail_zone=zone_conn, exempt=exempt,
               support=support, support_cap=support_cap, obstacle=obs['obstacle'], hard=hard, soft=soft,
               walked_corridor=corridor, walked_cells=wmask, dist_walked=dist_walk.astype(np.float32),
               obstacle_sdf_b=sdf['obstacle_b'], obstacle_sdf_c=sdf['obstacle_c'],
               obstacle_sdf_all=sdf['obstacle_all'], obstacle_sdf_all_c=sdf['obstacle_all_c'],
               support_sdf_cap=sdf['support_cap'], feasible_width_m=fw.astype(np.float32),
               route_dist=grid['dist'], route_near=near, device_height_m=np.float32(h_dev),
               n_body=feat['n_body'], frac_vert=obs['frac_vert'], occ_bins=feat['occ_bins'], n_above=feat['n_above'],
               n_rail=feat['n_rail'], vertical_line=obs['vertical_line'], vertical_plane=obs['vertical_plane'],
               surface=obs['surface'], continuous=obs['continuous'], railing=obs['railing'], diffuse=obs['diffuse'],
               hard_evidence=obs['hard_evidence'], conc=gr['conc'], above_env=gr['above_env'],
               **{k: v for k, v in feat.items() if k.startswith('vox_')})
    return dict(layers=layers, atlas=atlas, origin=origin, route=route, s=s, yaw=yaw, width=width, aux=aux, qual=qual,
                device=dev_stats, points=len(cells.z), shape=shape, t_route=line['t'], raw_index=line['raw_index'])


def gates_for(length):
    return np.linspace((length - 1.) / 8, length - 1., 8).astype(np.float32)


def save_map(directory, b, reverse=False):
    directory.mkdir(parents=True, exist_ok=True)
    layers, route, s, yaw, width = b['layers'], b['route'], b['s'], b['yaw'], b['width']
    if reverse:
        L = float(s[-1])
        layers = layers.copy()
        layers[..., 4] = L - layers[..., 4]
        route = route[::-1].copy()
        s = (L - s[::-1]).astype(np.float64)
        yaw = np.unwrap(yaw[::-1] + np.pi)
        width = width[::-1].copy()
    payload = dict(layers=layers.astype(np.float32), atlas=b['atlas'].astype(np.float32),
                   origin=b['origin'].astype(np.float32), res=np.float32(RES), route=route.astype(np.float32),
                   s=s.astype(np.float32), yaw=yaw.astype(np.float32), width=width.astype(np.float32),
                   gates=gates_for(float(s[-1])))
    np.savez_compressed(directory / 'map.npz', **payload)
    if reverse:
        return payload   # diagnostics layers are direction independent: see forward/aux.npz
    aux = dict(b['aux'])
    aux['station_t'] = b['t_route'][::-1] if reverse else b['t_route']
    aux['station_raw_index'] = b['raw_index'][::-1] if reverse else b['raw_index']
    aux['feasible_width_m'] = aux['feasible_width_m'][::-1] if reverse else aux['feasible_width_m']
    np.savez_compressed(directory / 'aux.npz', **aux)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-reverse', action='store_true')
    p.add_argument('--skip-sha', action='store_true')
    a = p.parse_args()
    t0 = time.time()
    cfg, raw_cfg = load_config()
    sha = None if a.skip_sha else file_sha256(PLANNING_PLY)
    if sha is not None and sha != PLANNING_SHA256:
        raise ValueError('planning cloud checksum mismatch')
    log(f'source sha256 {sha}', t0)
    b = build(cfg, t0)
    seg = json.loads((DATA / 'segment' / 'segment.json').read_text())
    records = []
    for name, rev in [('forward', False), ('reverse', True)]:
        if rev and a.no_reverse:
            continue
        d = DATA / name
        payload = save_map(d, b, rev)
        digest = hashlib.sha256((d / 'map.npz').read_bytes()).hexdigest()
        aux_digest = None if rev else hashlib.sha256((d / 'aux.npz').read_bytes()).hexdigest()
        ri = np.floor((payload['route'][:, :2] - b['origin']) / RES + .5).astype(int)
        known_on_route = b['aux']['known'][ri[:, 0], ri[:, 1]]
        q = b['qual']
        centre = q['centre'][::-1] if rev else q['centre']
        bandok = q['band'][::-1] if rev else q['band']
        record = dict(
            name=name, source_sha256=sha or PLANNING_SHA256, station_m=0.0,
            segment_length_m=float(payload['s'][-1]), stations=int(len(payload['s'])), points=int(b['points']),
            ground_known_on_route=float(known_on_route.mean()),
            footprint_feasible_on_route=float(centre.mean()),
            footprint_feasible_within_0p75m=float(bandok.mean()),
            start_at_2m_feasible=bool(centre[10]),
            continuous_path_within_0p75m=q['connected'], continuous_swept_feasible=q['swept'],
            conservative_proxy_qualified=bool(q['swept'] and centre[10]),
            route_adaptation=None,
            scope='Whole segment A (no sub-segment selection, no lateral recentring); see report.json for acceptance',
            dimensions=list(b['shape']), origin=[float(v) for v in b['origin']], res=RES,
            start_xy=payload['route'][0, :2].tolist(), end_xy=payload['route'][-1, :2].tolist(),
            direction='same as scan trajectory (race direction, user 2026-11-16)' if not rev else
                      'reversed scan trajectory (optional; not used for acceptance)',
            gates=payload['gates'].tolist(), width_clip=[1.0, 3.0],
            layers_dtype='float32', atlas_dtype='float32',
            layer_semantics=['ground z (m)', 'ground + max(0, top of atlas bands 0-2) (m)',
                             'support signed distance (m, <=0.025 = supported)',
                             'distance to centreline - width(nearest station)/2 (m)',
                             's of nearest station (m, along this map direction)',
                             'obstacle signed distance (m, <0 = colliding; walked-corridor prior applied)'
                             if cfg['walked_corridor']['apply_in_saved_map'] else 'obstacle signed distance (no prior)',
                             'obstacle type flag: 0 none, 1 hard (colliding), 2 soft-only (visible, non-colliding)'],
            map_sha256=digest, aux_sha256=aux_digest, device_height=b['device'],
            seconds=time.time() - t0,
            assumptions=dict(frame='Identity CSV-to-PLY (trajectory->cloud nearest-XY median 7 cm); no surveyed registration',
                             segment=seg['definition'], classification='Used only as hard evidence for codes 13/14 (posts/rails); never to make anything soft',
                             route='Cleaned scanner centreline (pauses/back-and-forth removed), not an official race-route annotation',
                             terrain='User: all walked-trail terrain is traversable in stair mode; step/grade limits exempted only in the trail zone',
                             outside_coverage='known=0, unsupported, empty atlas beyond 12 m of the centreline'))
        write_json(d / 'metadata.json', dict(record, config=raw_cfg))
        records.append(record)
        log(f'saved {name} {digest[:12]}', t0)
    manifest = dict(source=str(PLANNING_PLY), source_sha256=sha or PLANNING_SHA256, upstream_source=None,
                    scope='Segment A (mountain trail) static map from the official planning cloud; geometric model, not a semantic survey',
                    segment=seg, maps=records, parameters=raw_cfg,
                    acceptance_direction='forward')
    write_json(DATA / 'manifest.json', manifest)
    log('done', t0)


if __name__ == '__main__':
    main()
