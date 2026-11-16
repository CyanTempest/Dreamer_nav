"""Recompute v6/v7 blockers and measure remaining gaps directly in collision returns.

Usage: python -m nav4.mapping.diagnose_subcell
All official-cloud derivatives stay under data/trail_a/v7_diag.
"""
import json

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.analysis import blocking_runs, lateral_feasibility, lateral_grid
from nav4.mapping.common import DATA, FOOT_X, FOOT_Y, RES, footprint_ok, load_config, write_json
from nav4.mapping.grid import load_points
from nav4.mapping.route import lateral_violations


def feasibility(m, offsets, linear=True, xy=None, yaw=None):
    xy = m['route'][:, :2].astype(float) if xy is None else xy
    yaw = m['yaw'].astype(float) if yaw is None else yaw
    cand = lateral_grid(xy, yaw, offsets)
    return footprint_ok(m['layers'][..., 2], m['layers'][..., 5], m['origin'].astype(float),
                        cand.reshape(-1, 2), np.broadcast_to(yaw[:, None], cand.shape[:2]).ravel(),
                        obstacle_linear=linear).reshape(cand.shape[:2])


def main():
    cfg, _ = load_config()
    out = DATA / 'v7_diag'
    out.mkdir(exist_ok=True)
    keys = ('layers', 'origin', 'route', 'yaw', 's')
    with np.load(DATA / 'forward/map.npz') as f:
        m = {k: f[k] for k in keys}
    with np.load(DATA / 'v6_before_subcell/map.npz') as f:
        old = {k: f[k] for k in keys}
    with np.load(DATA / 'forward/aux.npz') as f:
        aux = {k: f[k] for k in ('known', 'hard', 'walked_corridor', 'railing', 'surface', 'continuous', 'vertical_line', 'vertical_plane')}
    origin = np.asarray(json.loads((DATA / 'forward/metadata.json').read_text())['origin'])
    offsets = np.arange(-cfg['corridor']['acceptance_band_m'], cfg['corridor']['acceptance_band_m']+1e-9, .05)
    records = {}
    for name, mm, linear in [('v6_nearest', old, False), ('v6_linear', old, True), ('v7_linear', m, True)]:
        fea = feasibility(mm, offsets, linear)
        ok = fea.any(1)
        records[name] = dict(blocking_stations=int((~ok).sum()), blocking_runs=blocking_runs(ok, mm['s'], mm['route'][:, :2]))
        np.savez_compressed(out/f'{name}_feasibility.npz', feasible=fea, offsets=offsets, s=mm['s'])
        if name == 'v7_linear':
            current = fea
        print(name, records[name], flush=True)
    # Compare on the unchanged v6 route as well (separates route adaptation from layer interpolation).
    fixed = feasibility(m, offsets, True, old['route'][:, :2].astype(float), old['yaw'].astype(float))
    records['v7_on_v6_route'] = dict(blocking_stations=int((~fixed.any(1)).sum()),
                                   blocking_runs=blocking_runs(fixed.any(1), old['s'], old['route'][:, :2]))
    sites = []
    descriptions = [
        (5.4, 6.8, '扶手立柱 / 陡坎（交接记录；物体名称为推断）'),
        (19., 20., '硬物 / 土坎（交接记录；推断）'),
        (94., 96., '密实物体，可能石头或树桩（交接记录；推断）'),
        (616., 619., '扶手及后方落差 / 密实点（交接记录；推断）'),
        (663., 665., '台坎 / 悬挑物（交接记录；推断）')]
    _, v6_station = cKDTree(old['route'][:, :2]).query(m['route'][:, :2])
    mapped = old['s'][v6_station]
    for lo, hi, description in descriptions:
        use = (mapped >= lo-.2) & (mapped <= hi+.2)
        sites.append(dict(v6_location_m=[lo, hi], description_inferred=description,
                          v7_station_range_m=[float(m['s'][use].min()), float(m['s'][use].max())],
                          stations=int(use.sum()), blocked_stations=int((~current[use].any(1)).sum())))
    blocked = np.flatnonzero(~current.any(1))
    xy, yaw = m['route'][:, :2].astype(float), m['yaw'].astype(float)
    best = offsets[lateral_violations(m['layers'][..., 2], m['layers'][..., 5], m['origin'].astype(float),
                                      xy[blocked], yaw[blocked], offsets).argmin(1)] if len(blocked) else []
    # Load only point-cloud subsets near the blocked stations. No point quantisation in measured widths.
    pts = load_points(('x', 'y', 'z', 'cls'))
    radius = cfg['corridor']['lateral_search_m']+max(abs(offsets))+FOOT_X[-1]
    groups = records['v7_linear']['blocking_runs']
    bounds = []
    for r in groups:
        line = xy[r['first_station']:r['last_station']+1]
        bounds.append((line.min(0)-radius, line.max(0)+radius))
    chosen = []
    for a in range(0, len(pts['x']), 1_000_000):
        xx, yy = pts['x'][a:a+1_000_000], pts['y'][a:a+1_000_000]
        keep = np.zeros(len(xx), bool)
        for low, high in bounds:
            keep |= (xx >= low[0]) & (xx <= high[0]) & (yy >= low[1]) & (yy <= high[1])
        chosen.append(np.flatnonzero(keep)+a)
    idx = np.concatenate(chosen)
    xyz = np.column_stack([pts[k][idx] for k in ('x', 'y', 'z')]).astype(float)
    ij = np.floor((xyz[:, :2]-origin)/RES+.5).astype(int)
    L = m['layers']
    known, ground = aux['known'], L[..., 0]
    gmax = ndi.maximum_filter(np.where(known, ground, -np.inf).astype(np.float32), size=3)
    gmax = np.where(np.isfinite(gmax), gmax, ground)
    i, j = ij.T
    height = xyz[:, 2]-ground[i, j]
    lo, hi = cfg['obstacles']['body_band_m']
    body = (xyz[:, 2]-gmax[i, j] > lo) & (height < hi)
    collide = aux['hard'][i, j] & body
    if cfg['walked_corridor']['apply_in_saved_map']:
        collide &= ~aux['walked_corridor'][i, j]
    xyz, idx, ij, height = xyz[collide], idx[collide], ij[collide], height[collide]
    noise = cfg['obstacles']['distance_noise_m']

    def point(k, centre, normal, tangent):
        a, b = ij[k]
        d = xyz[k, :2]-centre
        return dict(crop_index=int(idx[k]), xyz_m=xyz[k], lateral_m=float(d@normal), along_m=float(d@tangent),
                    height_above_cell_ground_m=float(height[k]), cls=int(pts['cls'][idx[k]]),
                    evidence=[name for name in ('railing', 'surface', 'continuous', 'vertical_line', 'vertical_plane') if aux[name][a, b]])

    measurements = []
    no_obstacle = np.full(L.shape[:2], 100., np.float32)
    all_support = -no_obstacle
    for st, offset in zip(blocked, best):
        n = np.array([-np.sin(yaw[st]), np.cos(yaw[st])])
        t = np.array([np.cos(yaw[st]), np.sin(yaw[st])])
        centre = xy[st]+offset*n
        d = xyz[:, :2]-centre
        lateral, along = d@n, d@t
        candidates = xy[st]+offsets[:, None]*n
        angles = np.full(len(offsets), yaw[st])
        supported = footprint_ok(L[..., 2], no_obstacle, m['origin'].astype(float), candidates, angles)
        field_free = footprint_ok(all_support, L[..., 5], m['origin'].astype(float), candidates, angles)
        # Independent raw-point audit: exact Euclidean distance of collision returns
        # to the rectangle, with the same measured noise radius. This explains cases
        # where a wide two-sided point gap still fails support or the sampled field.
        raw_clearance = []
        for candidate_offset in offsets:
            cross = lateral + offset - candidate_offset
            q = np.stack([np.abs(along)-FOOT_X[-1], np.abs(cross)-FOOT_Y[-1]], -1)
            point_to_body = np.linalg.norm(np.maximum(q, 0), axis=1)
            raw_clearance.append(float(point_to_body.min(initial=np.inf))-noise)
        raw_free = np.asarray(raw_clearance) >= 0
        # Sections span one robot body length; slab thickness one source map cell.
        sections = []
        for longitudinal in FOOT_X:
            sel = (np.abs(along-longitudinal) <= RES/2) & (np.abs(lateral) <= cfg['corridor']['lateral_search_m'])
            left = np.flatnonzero(sel & (lateral >= 0))
            right = np.flatnonzero(sel & (lateral < 0))
            if not len(left) or not len(right):
                continue
            l = left[lateral[left].argmin()]
            r = right[lateral[right].argmax()]
            width = float(lateral[l]-lateral[r])
            sections.append(dict(longitudinal_m=float(longitudinal), width_m=width,
                                 width_minus_two_noise_m=width-2*noise,
                                 left=point(l, centre, n, t), right=point(r, centre, n, t)))
        # Independent footprint counterfactuals identify support versus obstacle restrictions.
        one = st+np.zeros(1, int)
        sup_only = footprint_ok(L[..., 2], no_obstacle, m['origin'].astype(float),
                                centre[None], yaw[one])[0]
        obs_only = footprint_ok(all_support, L[..., 5], m['origin'].astype(float),
                                centre[None], yaw[one])[0]
        measurements.append(dict(station=int(st), s_m=float(m['s'][st]), v6_s_m=float(mapped[st]),
                                 xy_m=xy[st], best_offset_m=float(offset), support_ok=bool(sup_only),
                                 obstacle_ok=bool(obs_only),
                                 supported_offsets_m=offsets[supported],
                                 field_obstacle_free_offsets_m=offsets[field_free],
                                 raw_point_free_offsets_m=offsets[raw_free],
                                 supported_and_raw_free_offsets_m=offsets[supported & raw_free],
                                 raw_point_clearance_m=raw_clearance,
                                 narrowest=min(sections, key=lambda q:q['width_m']) if sections else None,
                                 sections=sections))
    for r in groups:
        rows = [v for v in measurements if r['first_station'] <= v['station'] <= r['last_station']]
        widths = [v for v in rows if v['narrowest'] is not None]
        r['measurement'] = min(widths, key=lambda v:v['narrowest']['width_m']) if widths else None
        if r['measurement']:
            r['measurement'] = {k:v for k,v in r['measurement'].items() if k != 'sections'}
        r['support_failing_stations_at_best_offset'] = sum(not v['support_ok'] for v in rows)
        r['obstacle_failing_stations_at_best_offset'] = sum(not v['obstacle_ok'] for v in rows)
        r['stations_feasible_with_raw_points_and_same_support'] = sum(bool(len(v['supported_and_raw_free_offsets_m'])) for v in rows)
    records.update(sites=sites, measurements=measurements,
                   method=dict(collision_points='Exactly the model body band: z > max known 3x3 ground + .22, z < own ground + .80; hard cell, excluding walked prior cells. Raw crop XY/Z are retained.',
                               clearance='At each blocked station choose the least-violating lateral offset within +/-0.75 m. For each of 43 sections across the 1.03 m body length, collect collision returns in a 0.10 m-thick slab normal to route, within +/-1.5 m laterally; width = min positive lateral minus max negative lateral. Report the narrowest two-sided section. Missing side gives null, not an invented width.',
                               caution='Projected width of finite-thickness cloud slabs, not a proof of a continuous drivable path. Endpoints may differ in along-route coordinate by up to 0.10 m. Terrain support and turning can block a gap wider than the envelope. Semantic object names come from HANDOFF, not cloud classification labels.',
                               noise_m=noise, footprint_m=[2*FOOT_X[-1], 2*FOOT_Y[-1]], slab_m=RES,
                               samples_along_body=len(FOOT_X), section_spacing_m=float(FOOT_X[1]-FOOT_X[0])))
    write_json(out/'diagnosis.json', records)
    if groups:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, axs = plt.subplots(len(groups), 1, figsize=(8, 3*len(groups)), squeeze=False, layout='constrained')
        for ax, run in zip(axs[:, 0], groups):
            row = run['measurement'] or next(v for v in measurements if v['station'] == run['first_station'])
            st, offset = row['station'], row['best_offset_m']
            n = np.array([-np.sin(yaw[st]), np.cos(yaw[st])])
            t = np.array([np.cos(yaw[st]), np.sin(yaw[st])])
            d = xyz[:, :2] - (xy[st]+offset*n)
            lateral, along = d@n, d@t
            use = (np.abs(lateral) < cfg['corridor']['lateral_search_m']) & (np.abs(along) < 2*FOOT_X[-1])
            ax.scatter(lateral[use], along[use], s=2, c=height[use], vmin=lo, vmax=hi, cmap='viridis')
            ax.add_patch(Rectangle((-FOOT_Y[-1], -FOOT_X[-1]), 2*FOOT_Y[-1], 2*FOOT_X[-1],
                                   facecolor='none', edgecolor='red', label='1.03 x 0.58 m body'))
            q = row['narrowest']
            if q:
                ax.plot([q['right']['lateral_m'], q['left']['lateral_m']],
                        [q['right']['along_m'], q['left']['along_m']], 'r.-')
            ax.set_title(f"v7 s={row['s_m']:.2f} m; projected point gap={q['width_m']:.3f} m" if q else
                         f"v7 s={row['s_m']:.2f} m; no two-sided collision returns in slabs")
            ax.set_xlabel('lateral from best footprint centre (m, +left)')
            ax.set_ylabel('along route (m)')
            ax.set_aspect('equal')
            ax.legend(fontsize=8)
        fig.savefig(out/'remaining_gaps.png', dpi=150)
        plt.close(fig)
    print('wrote', out/'diagnosis.json', flush=True)


if __name__ == '__main__':
    main()
