"""Task 5: acceptance report recomputed from the SAVED maps.

Reads data/trail_a/forward/{map.npz,aux.npz}, data/trail_a/old_proxy/map.npz,
the segment-A centreline/poses and (for cross-sections) the cropped cloud.
Writes data/trail_a/report.json and data/trail_a/figures/*.png.

Variants (footprint 1.03x0.58 m, 43x25 samples, support sdf<=0.025 & obstacle sdf>=0):
  a  old proxy (r9 criteria re-run on the segment-A grid)
  b  new model, no walked-corridor prior
  c  new model, with walked-corridor prior   (= map.npz layers 2 and 5)
  plus sensitivity variants: soft treated as hard (b_soft_hard, c_soft_hard) and
  capability-only support without the trail-terrain exemption (c_no_exemption).

  fix-effect variants: c_fixA_off_sparse_as_hard (sparse counted as hard), c_fixB_off_v5_exemption
  (v5 trail exemption: 0.3 m-disc cross grade, 0.20 m connectivity); before = v5_b/v5_c read from
  data/trail_a/v5_before_fix/.  All variants are evaluated along the scanner centreline (same
  stations as v5); 'saved_route' answers the acceptance question along the saved (fix D) route.

Usage: python -m nav4.mapping.report_trail_map [--no-sections]
"""
import argparse
import json
import time

import numpy as np

from nav4.mapping.analysis import (ROBOT_W, blocking_runs, corridor_path, feasible_width, lateral_feasibility, lookup,
                                   validate_path)
from nav4.mapping.common import DATA, RES, footprint_ok, load_config, load_poses, write_json
from nav4.mapping.grid import load_centreline
from nav4.mapping.report_fixes import (blocker_causes, blocker_sections, fig_before_after, fix_effects, fix_sections,
                                       load_v5)

FIG = DATA / 'figures'


def load_maps():
    with np.load(DATA / 'forward' / 'map.npz') as f:
        m = {k: f[k] for k in f.files}
    with np.load(DATA / 'forward' / 'aux.npz') as f:
        aux = {k: f[k] for k in f.files}
    with np.load(DATA / 'old_proxy' / 'map.npz') as f:
        old = {k: f[k] for k in ('layers', 'origin')}
    return m, aux, old


def evaluate(name, support_sdf, obstacle_sdf, origin, xy, yaw, s, cfg):
    c = cfg['corridor']
    offsets = np.arange(-c['lateral_search_m'], c['lateral_search_m'] + 1e-9, .05)
    feas = lateral_feasibility(support_sdf, obstacle_sdf, origin, xy, yaw, offsets)
    centre = feas[:, np.argmin(np.abs(offsets))]
    band = np.abs(offsets) <= c['acceptance_band_m'] + 1e-9
    any_band = feas[:, band].any(1)
    width = feasible_width(feas, offsets, c['acceptance_band_m'])
    shift = int(round(c['dp_max_shift_per_station_m'] / .05))
    connected, reach, path, last = corridor_path(feas, offsets, shift, c['acceptance_band_m'])
    rec = dict(variant=name, centreline_feasible_fraction=float(centre.mean()),
               station_feasible_fraction_within_band=float(any_band.mean()),
               continuous_path_within_band=bool(connected), reach_fraction=float(reach),
               reach_until_s_m=float(s[last]) if last >= 0 else 0.,
               feasible_width_m=dict(min=float(width.min()), p5=float(np.percentile(width, 5)),
                                     median=float(np.median(width))),
               stations_width_below_fit=int((width < ROBOT_W + c['fit_margin_m']).sum()),
               blocking_stations=int((~any_band).sum()))
    blocked = np.flatnonzero(~any_band)
    rec['blocking_runs'] = blocking_runs(any_band, s, xy)
    rec['blocking_station_list'] = [dict(station=int(k), s_m=float(s[k]), x=float(xy[k, 0]), y=float(xy[k, 1]))
                                    for k in blocked] if len(blocked) <= 400 else 'see blocking_runs (too many)'
    narrow = width < ROBOT_W + c['fit_margin_m']
    rec['narrow_runs'] = blocking_runs(~narrow, s, xy)
    if connected:
        v = validate_path(support_sdf, obstacle_sdf, origin, xy, yaw, path)
        rec['path_validation'] = dict(fine_pass_fraction=v['fine_pass_fraction'],
                                      swept_pass_fraction=v['swept_pass_fraction'], swept_ok=v['swept_ok'],
                                      max_abs_offset_m=float(np.abs(path).max()),
                                      fine_fail_s_m=v['fine_fail_s'][:50])
        rec['continuous_swept_feasible'] = bool(v['swept_ok'])
    else:
        rec['continuous_swept_feasible'] = False
    return rec, dict(centre=centre, any_band=any_band, width=width, path=path, feas=feas, offsets=offsets)


def ground_quality(m, aux, line, rows, h_dev, cfg):
    origin = m['origin'].astype(np.float64)
    ground = m['layers'][..., 0]
    known = m['atlas'][..., 19] > .5
    w = line['walked_index']
    from scipy import ndimage as ndi
    sp_all = ndi.median_filter(np.r_[0, np.linalg.norm(np.diff(rows[:, 1:3], axis=0), axis=1) / np.diff(rows[:, 0])], size=5)
    # pauses: >= 3 s below 0.15 m/s; the operator lowered the scanner during several stops,
    # so poses within 5 s of a pause are excluded from the prior residual
    slow = sp_all < .15
    lab, n = ndi.label(slow)
    pause = np.zeros(len(rows), bool)
    if n:
        sizes = ndi.sum(slow, lab, index=np.arange(1, n + 1))
        long_ = np.zeros(n + 1, bool)
        long_[1:] = sizes >= 30
        pause = long_[lab]
        pause = ndi.binary_dilation(pause, iterations=50)
    sp = sp_all[w]
    moving = (sp >= cfg['ground']['device_height_min_speed_mps']) & ~pause[w]
    P = rows[w][moving]
    g = lookup(ground, origin, P[:, 1:3], np.nan)
    kn = lookup(known, origin, P[:, 1:3], False)
    prior = P[:, 3] - h_dev
    res = g - prior
    ok = kn & np.isfinite(res)
    a = np.abs(res[ok])
    k = np.clip(np.searchsorted(line['t'], P[:, 0]), 0, len(line['s']) - 1)
    st = line['s'][k]
    order = np.argsort(-np.where(ok, np.abs(res), -1))
    worst = []
    taken = []
    for idx in order:
        if not ok[idx] or len(worst) >= 12:
            break
        if any(abs(st[idx] - t) < 3 for t in taken):
            continue
        taken.append(st[idx])
        worst.append(dict(s_m=float(st[idx]), x=float(P[idx, 1]), y=float(P[idx, 2]), residual_m=float(res[idx]),
                          scanner_z=float(P[idx, 3])))
    d = aux['route_dist']
    near2 = d <= 2.0
    near2 &= aux['coverage']
    return dict(device_height_m=float(h_dev), poses=int(ok.sum()),
                pose_filter='moving (>=0.3 m/s) and >5 s away from pauses (>=3 s below 0.15 m/s)',
                residual_signed_median_m=float(np.median(res[ok])),
                abs_residual_median_m=float(np.median(a)), abs_residual_p95_m=float(np.percentile(a, 95)),
                abs_residual_max_m=float(a.max()), frac_abs_residual_gt_0p3=float((a > .3).mean()),
                worst_places=worst,
                unknown_fraction_within_2m_of_centreline=float((~known[near2]).mean()),
                prior_filled_cells=int(aux['prior_fill'].sum()),
                prior_filled_fraction_of_walked_cells=float(aux['prior_fill'][aux['walked_cells']].mean())), \
        dict(s=st[ok], res=res[ok])


def profile(m, line, step=.05):
    origin = m['origin'].astype(np.float64)
    xy = line['xy']
    s = line['s']
    fs = np.arange(0, s[-1], step)
    pts = np.stack([np.interp(fs, s, xy[:, 0]), np.interp(fs, s, xy[:, 1])], -1)
    z = lookup(m['layers'][..., 0], origin, pts, np.nan)
    return fs, z


def detect_stairs(fs, z, cfg):
    """Risers along the centreline profile (0.05 m samples of the 0.1 m ground)."""
    st = cfg['report']['stairs']
    dz = np.r_[z[2:] - z[:-2], 0, 0]                  # rise over 0.1 m
    cand = np.abs(dz) >= st['riser_min_m']
    risers = []
    k = 0
    n = len(z)
    while k < n:
        if not cand[k]:
            k += 1
            continue
        e = k
        while e + 1 < n and cand[e + 1] and np.sign(dz[e + 1]) == np.sign(dz[k]):
            e += 1
        a = max(0, k - 2)
        b = min(n - 1, e + 4)
        before = z[max(0, a - 2):a + 1]
        after = z[b:min(n, b + 3)]
        h = float(np.median(after) - np.median(before))
        if abs(h) >= st['riser_min_m'] and (fs[b] - fs[a]) <= st['riser_max_run_m']:
            risers.append(dict(s=float((fs[a] + fs[b]) / 2), h=h))
        k = e + 1
    flights = []
    cur = []
    for r in risers:
        if cur and (r['s'] - cur[-1]['s'] > st['tread_max_m'] or np.sign(r['h']) != np.sign(cur[-1]['h'])):
            if len(cur) >= st['min_risers']:
                flights.append(cur)
            cur = []
        cur.append(r)
    if len(cur) >= st['min_risers']:
        flights.append(cur)
    out = []
    for f in flights:
        h = np.abs([r['h'] for r in f])
        tread = np.diff([r['s'] for r in f])
        out.append(dict(s_from_m=f[0]['s'], s_to_m=f[-1]['s'], steps=len(f),
                        direction='up' if f[0]['h'] > 0 else 'down',
                        riser_median_m=float(np.median(h)), riser_max_m=float(h.max()), riser_min_m=float(h.min()),
                        tread_median_along_route_m=float(np.median(tread)) if len(tread) else None,
                        risers_over_0p20=int((h > .20).sum())))
    return out, risers


def terrain_stats(m, aux, line, fs, z, cfg):
    origin = m['origin'].astype(np.float64)
    out = {}
    for base in (1.0, 3.0):
        k = int(round(base / .05 / 2))
        g = (z[2 * k:] - z[:-2 * k]) / base
        ang = np.degrees(np.arctan(np.abs(g)))
        i = int(np.nanargmax(ang))
        out[f'max_sustained_grade_{base:.0f}m_deg'] = float(ang[i])
        out[f'max_sustained_grade_{base:.0f}m_at_s_m'] = float(fs[i + k])
        out[f'grade_{base:.0f}m_quantiles_deg'] = {str(q): float(np.nanpercentile(ang, q)) for q in (50, 90, 95, 99)}
    k = int(round(1.0 / .05 / 2))
    g1 = np.degrees(np.arctan((z[2 * k:] - z[:-2 * k]) / 1.0))
    hist, edges = np.histogram(np.abs(g1[np.isfinite(g1)]), bins=np.arange(0, 65, 5))
    out['grade_1m_histogram'] = dict(edges_deg=edges.tolist(), fraction=(hist / max(1, hist.sum())).tolist())
    cross = lookup(aux['cross_grade_deg'], origin, line['xy'], np.nan)
    plane = lookup(aux['grade_deg'], origin, line['xy'], np.nan)
    out['cross_slope_deg_quantiles'] = {str(q): float(np.nanpercentile(cross, q)) for q in (50, 90, 95, 99, 100)}
    out['plane_grade_1m_disc_deg_quantiles'] = {str(q): float(np.nanpercentile(plane, q)) for q in (50, 90, 95, 99, 100)}
    out['elevation_gain_m'] = float(np.nansum(np.clip(np.diff(z), 0, None)))
    out['elevation_loss_m'] = float(-np.nansum(np.clip(np.diff(z), None, 0)))
    return out, g1, cross


def capability_diagnostic(m, aux, line, cfg):
    """Stations where 0.20 m / 35 deg (no trail exemption) would block the trail: robot
    footprint at the centreline contains capability-unsupported but known cells."""
    origin = m['origin'].astype(np.float64)
    from nav4.mapping.analysis import footprint_cells
    shape = aux['known'].shape
    ij, inside = footprint_cells(origin, shape, line['xy'], line['yaw'])
    kn = aux['known'][ij[..., 0], ij[..., 1]] & inside
    cap = aux['cap_ok'][ij[..., 0], ij[..., 1]] & inside
    blocked = (kn & ~cap).any(1)
    step = np.where(kn, aux['step_m'][ij[..., 0], ij[..., 1]], 0).max(1)
    grade = np.where(kn, np.nan_to_num(aux['grade_deg'][ij[..., 0], ij[..., 1]]), 0).max(1)
    t = cfg['terrain']
    rows = []
    for r in blocking_runs(~blocked, line['s'], line['xy']):
        a, b = r['first_station'], r['last_station'] + 1
        r.update(max_step_m=float(step[a:b].max()), max_plane_grade_deg=float(grade[a:b].max()),
                 cause=('step' if step[a:b].max() > t['max_step_m'] else '') +
                       ('+grade' if grade[a:b].max() > t['max_grade_deg'] else ''))
        rows.append(r)
    return dict(stations_blocked_without_exemption=int(blocked.sum()), runs=rows), blocked, step, grade


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-sections', action='store_true')
    a = p.parse_args()
    t0 = time.time()
    cfg, raw = load_config()
    m, aux, old = load_maps()
    line = load_centreline()
    rows = load_poses()
    # variants (a)/(b)/(c) are evaluated along the scanner centreline (same stations as v5) so that
    # before/after is comparable; the acceptance question is answered along the SAVED route.
    s, xy, yaw = line['s'].astype(np.float64), line['xy'].astype(np.float64), line['yaw'].astype(np.float64)
    assert np.allclose(aux['centreline_xy'], xy, atol=1e-6), 'aux centreline differs from segment/centreline.npz'
    rs, rxy, ryaw = m['s'].astype(np.float64), m['route'][:, :2].astype(np.float64), m['yaw'].astype(np.float64)
    origin = m['origin'].astype(np.float64)
    L = m['layers']
    v5 = load_v5()
    variants = {
        'a_old_proxy': (old['layers'][..., 2], old['layers'][..., 5], old['origin'].astype(np.float64)),
        'b_new_no_prior': (L[..., 2], aux['obstacle_sdf_b'], origin),
        'c_new_with_prior': (L[..., 2], L[..., 5], origin),
        'b_soft_as_hard': (L[..., 2], aux['obstacle_sdf_all'], origin),
        'c_soft_as_hard': (L[..., 2], aux['obstacle_sdf_all_c'], origin),
        'c_no_trail_exemption': (aux['support_sdf_cap'], L[..., 5], origin),
        'c_fixA_off_sparse_as_hard': (L[..., 2], aux['obstacle_sdf_default_c'], origin),
        'c_fixB_off_v5_exemption': (aux['support_sdf_v5_exemption'], L[..., 5], origin),
    }
    if v5 is not None:
        variants['v5_b_no_prior'] = (v5['map']['layers'][..., 2], v5['aux']['obstacle_sdf_b'], v5['map']['origin'].astype(np.float64))
        variants['v5_c_with_prior'] = (v5['map']['layers'][..., 2], v5['map']['layers'][..., 5], v5['map']['origin'].astype(np.float64))
    results, per = {}, {}
    for name, (sup, obs, org) in variants.items():
        rec, d = evaluate(name, sup, obs, org, xy, yaw, s, cfg)
        results[name] = rec
        per[name] = d
        print(name, {k: rec[k] for k in ('centreline_feasible_fraction', 'station_feasible_fraction_within_band',
                                          'continuous_path_within_band', 'continuous_swept_feasible', 'reach_until_s_m',
                                          'blocking_stations', 'stations_width_below_fit')}, len(rec['blocking_runs']), flush=True)
    # acceptance along the saved route: DP within +-0.75 m of it, and the route itself (offset 0)
    rec, d = evaluate('saved_route', L[..., 2], L[..., 5], origin, rxy, ryaw, rs, cfg)
    itself = validate_path(L[..., 2], L[..., 5], origin, rxy, ryaw, np.zeros(len(rs)))
    rec['route_itself'] = dict(fine_pass_fraction=itself['fine_pass_fraction'], swept_pass_fraction=itself['swept_pass_fraction'],
                               swept_ok=itself['swept_ok'], fine_fail_s_m=itself['fine_fail_s'][:200])
    results['saved_route'] = rec
    per['saved_route'] = d
    print('saved_route', {k: rec[k] for k in ('centreline_feasible_fraction', 'station_feasible_fraction_within_band',
                                              'continuous_path_within_band', 'continuous_swept_feasible', 'blocking_stations')},
          rec['route_itself'], flush=True)
    # where the soft/hard decision and the walked-corridor prior matter
    b, c = per['b_new_no_prior']['any_band'], per['c_new_with_prior']['any_band']
    prior_matters = np.flatnonzero(c & ~b)
    soft_matters = np.flatnonzero(per['c_new_with_prior']['any_band'] & ~per['c_soft_as_hard']['any_band'])
    results['walked_corridor_prior_matters'] = dict(
        stations=int(len(prior_matters)), runs=blocking_runs(~np.isin(np.arange(len(s)), prior_matters), s, xy))
    results['soft_decision_matters'] = dict(
        stations=int(len(soft_matters)), runs=blocking_runs(~np.isin(np.arange(len(s)), soft_matters), s, xy),
        note='stations feasible in (c) that become infeasible if every soft obstacle collided')
    h_dev = float(aux['device_height_m'])
    gq, gres = ground_quality(m, aux, line, rows, h_dev, cfg)
    results['ground_quality'] = gq
    fs, z = profile(m, line)
    stairs, risers = detect_stairs(fs, z, cfg)
    ts, g1, cross = terrain_stats(m, aux, line, fs, z, cfg)
    ts['stairs'] = stairs
    ts['stair_flights'] = len(stairs)
    ts['stair_steps_total'] = int(sum(f['steps'] for f in stairs))
    ts['max_riser_in_flights_m'] = float(max([f['riser_max_m'] for f in stairs], default=0.))
    ts['trail_zone_connect_max_step_m'] = cfg['terrain']['trail_zone_connect_max_step_m']
    ts['connect_step_covers_max_riser'] = bool(ts['max_riser_in_flights_m'] <= cfg['terrain']['trail_zone_connect_max_step_m'])
    results['terrain'] = ts
    capd, cap_blocked, cap_step, cap_grade = capability_diagnostic(m, aux, line, cfg)
    results['capability_limits_on_trail'] = capd
    results['acceptance_question'] = dict(
        question='Continuous swept-feasible path start->end within +-0.75 m for the 1.03x0.58 m footprint, trail terrain traversable (user)',
        a_old_proxy=results['a_old_proxy']['continuous_swept_feasible'],
        b_new_no_prior=results['b_new_no_prior']['continuous_swept_feasible'],
        c_new_with_prior=results['c_new_with_prior']['continuous_swept_feasible'],
        saved_route_within_band=results['saved_route']['continuous_swept_feasible'],
        saved_route_itself=results['saved_route']['route_itself']['swept_ok'])
    if v5 is not None:
        results['acceptance_question']['v5_c_with_prior'] = results['v5_c_with_prior']['continuous_swept_feasible']
    results['fixes'] = fix_effects(m, aux, v5, line, per, results, cfg)
    results['remaining_blockers'] = blocker_causes(m, aux, rxy, ryaw, rs, per['saved_route'], cfg)
    results['remaining_blockers_on_centreline'] = blocker_causes(m, aux, xy, yaw, s, per['c_new_with_prior'], cfg)
    results['map'] = dict(shape=list(L.shape), origin=origin.tolist(), length_m=float(rs[-1]), stations=len(rs),
                          centreline_length_m=float(s[-1]), centreline_stations=len(s), sparse_cells=int(aux['sparse'].sum()),
                          deck_cells=int(aux['deck'].sum()),
                          bytes_in_ram=int(sum(v.nbytes for v in m.values())),
                          known_fraction_in_coverage=float((m['atlas'][..., 19] > .5)[aux['coverage']].mean()),
                          hard_cells=int(aux['hard'].sum()), soft_cells=int(aux['soft'].sum()),
                          walked_corridor_cells=int(aux['walked_corridor'].sum()),
                          hard_cells_in_walked_corridor=int((aux['hard'] & aux['walked_corridor']).sum()))
    hc = aux['hard'] & aux['walked_corridor']
    results['walked_corridor_neutralised'] = dict(
        cells=int(hc.sum()),
        with_post_or_rail_class_13_14=int((hc & aux['railing']).sum()),
        with_vertical_line=int((hc & aux['vertical_line']).sum()),
        with_vertical_plane=int((hc & aux['vertical_plane']).sum()),
        with_surface=int((hc & aux['surface']).sum()),
        continuing_above_0p8m=int((hc & (aux['n_above'] >= 2)).sum()),
        sparse_no_shape_evidence=int((hc & ~aux['hard_evidence'] & ~aux['diffuse']).sum()),
        median_body_points=float(np.median(aux['n_body'][hc])) if hc.any() else 0.,
        note='hard cells inside +-0.30 m of the raw walked path that the prior turns non-colliding (layer 6 = 2)')
    near3 = aux['route_dist'] <= 3.
    ob = aux['obstacle'] & near3
    results['obstacle_evidence_within_3m'] = dict(
        obstacle_cells=int(ob.sum()), hard=int((aux['hard'] & near3).sum()), soft=int((aux['soft'] & near3).sum()),
        hard_by_evidence=dict(vertical_line=int((ob & aux['vertical_line']).sum()),
                              vertical_plane=int((ob & aux['vertical_plane']).sum()),
                              surface=int((ob & aux['surface']).sum()), continuous_column=int((ob & aux['continuous']).sum()),
                              class_13_14=int((ob & aux['railing']).sum()),
                              continues_above_0p8m=int((ob & (aux['n_above'] >= 2)).sum()),
                              no_shape_evidence_sparse=int((ob & ~aux['hard_evidence'] & ~aux['diffuse']).sum())),
        walked_corridor_obstacle_cells=int((aux['obstacle'] & aux['walked_corridor']).sum()),
        walked_corridor_labelled_soft=int((aux['soft'] & aux['walked_corridor']).sum()))
    FIG.mkdir(parents=True, exist_ok=True)
    fig_strip(s, per, results, FIG / 'feasibility_strip.png', cap_blocked)
    fig_profile(fs, z, stairs, risers, g1, s, gres, FIG / 'elevation_profile.png')
    fig_overview(m, aux, line, FIG / 'overview.png')
    fig_tiles(m, aux, line, per['c_new_with_prior'], FIG / 'overview_tiles.png')
    fig_blocking(m, aux, line, rows, per['c_new_with_prior'], results['c_new_with_prior']['blocking_runs'],
                 FIG / 'blocking_zooms.png')
    fig_before_after(s, rs, per, results, aux, FIG / 'fix_strip_before_after.png')
    if not a.no_sections:
        from nav4.mapping.report_sections import render_sections
        results['cross_sections'] = render_sections(m, aux, old, line, per, cfg, stairs, g1)
        from nav4.mapping.report_fixes import _slab_and_walked
        slab, _ = _slab_and_walked(line, rows)
        results['fix_sections'] = fix_sections(m, aux, v5, line, rows, FIG, cfg, slab=slab)
        results['remaining_blocker_sections'] = blocker_sections(m, aux, v5, line, rows, results['remaining_blockers'], FIG, cfg,
                                                                 slab=slab)
    results['seconds'] = time.time() - t0
    write_json(DATA / 'report.json', results)
    print(json.dumps(results['acceptance_question'], indent=1))
    print(json.dumps(results['ground_quality'], indent=1)[:2000])
    print(json.dumps({k: v for k, v in results['terrain'].items() if k != 'stairs'}, indent=1)[:3000])
    print('stairs', json.dumps(results['terrain']['stairs'], indent=0)[:4000])
    print('capability', json.dumps(capd, indent=0)[:3000])


# ------------------------------------------------------------------ figures
def fig_strip(s, per, results, path, cap_blocked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['a_old_proxy', 'b_new_no_prior', 'c_new_with_prior', 'c_soft_as_hard']
    fig, ax = plt.subplots(len(names) + 2, 1, figsize=(18, 11), sharex=True, layout='constrained')
    for k, n in enumerate(names):
        d = per[n]
        img = np.vstack([d['centre'], d['any_band']]).astype(float)
        ax[k].imshow(img, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest',
                     extent=[s[0], s[-1], 1.5, -.5])
        r = results[n]
        ax[k].set_yticks([0, 1], ['centre', '+-0.75'])
        ax[k].set_title(f"{n}: centre {r['centreline_feasible_fraction']:.1%}, any offset in band {r['station_feasible_fraction_within_band']:.1%}, "
                        f"continuous={r['continuous_path_within_band']} swept={r['continuous_swept_feasible']}", fontsize=9)
    k = len(names)
    ax[k].imshow(cap_blocked[None].astype(float), aspect='auto', cmap='Greys', vmin=0, vmax=1, extent=[s[0], s[-1], .5, -.5])
    ax[k].set_yticks([]); ax[k].set_title('stations where 0.20 m / 35 deg (without trail exemption) would put unsupported cells under the centreline footprint', fontsize=9)
    for n, col in [('a_old_proxy', '0.6'), ('b_new_no_prior', 'b'), ('c_new_with_prior', 'g')]:
        ax[k + 1].plot(s, per[n]['width'], color=col, lw=.7, label=n)
    ax[k + 1].axhline(ROBOT_W + .1, color='r', ls='--', lw=.8, label='0.58+0.10 m')
    ax[k + 1].set_ylim(0, 3.7); ax[k + 1].legend(fontsize=8, ncol=4); ax[k + 1].set_ylabel('feasible width (m)')
    ax[k + 1].set_xlabel('station s (m)')
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_profile(fs, z, stairs, risers, g1, s, gres, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True, layout='constrained')
    ax[0].plot(fs, z, 'k-', lw=.6, label='map ground under centreline')
    for f in stairs:
        ax[0].axvspan(f['s_from_m'], f['s_to_m'], color='orange', alpha=.4)
    ax[0].plot([r['s'] for r in risers], np.interp([r['s'] for r in risers], fs, z), 'r|', ms=6, label='detected risers')
    ax[0].set_ylabel('z (m)'); ax[0].legend(fontsize=8); ax[0].set_title(f'Elevation profile, {len(stairs)} stair flights (orange)')
    k = 10
    ax[1].plot(fs[k:-k], g1, lw=.5)
    ax[1].axhline(35, color='r', ls='--', lw=.8); ax[1].axhline(-35, color='r', ls='--', lw=.8)
    ax[1].set_ylabel('grade over 1 m (deg)')
    ax[2].plot(gres['s'], gres['res'], '.', ms=1)
    ax[2].set_ylabel('ground - (scanner z - h_dev) (m)'); ax[2].set_xlabel('station s (m)'); ax[2].set_ylim(-1, 1)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def class_image(aux, m):
    known = m['atlas'][..., 19] > .5
    img = np.ones((*known.shape, 3), np.float32)
    img[known] = (.95, .85, .45)                          # known but unsupported
    img[m['layers'][..., 2] <= .025] = (.55, .8, .55)       # supported
    img[aux['soft']] = (.95, .6, .2)
    img[aux['hard']] = (.8, .1, .1)
    img[aux['hard'] & aux['walked_corridor']] = (.8, .2, .8)
    img[~aux['coverage']] = (1, 1, 1)
    img[aux['coverage'] & ~known] = (.6, .6, .6)
    return img


def fig_overview(m, aux, line, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    origin = m['origin'].astype(np.float64)
    sh = m['layers'].shape[:2]
    ext = [origin[0] - RES / 2, origin[0] + sh[0] * RES - RES / 2, origin[1] - RES / 2, origin[1] + sh[1] * RES - RES / 2]
    g = np.where(aux['coverage'], m['layers'][..., 0], np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(26, 13), layout='constrained')
    ls = LightSource(azdeg=315, altdeg=45)
    gg = np.nan_to_num(g, nan=np.nanmin(g))
    rgb = ls.shade(gg.T, plt.cm.terrain, vert_exag=2, blend_mode='soft', vmin=np.nanmin(g), vmax=np.nanmax(g))
    rgb[~aux['coverage'].T] = 1
    ax[0].imshow(rgb, origin='lower', extent=ext, interpolation='nearest')
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(np.nanmin(g), np.nanmax(g)))
    plt.colorbar(sm, ax=ax[0], shrink=.6, label='ground z (m)')
    ax[1].imshow(np.transpose(class_image(aux, m), (1, 0, 2)), origin='lower', extent=ext, interpolation='nearest')
    for a in ax:
        a.plot(line['xy'][:, 0], line['xy'][:, 1], 'b-', lw=.4)
        for k in range(0, len(line['s']), 250):
            a.annotate(f"{line['s'][k]:.0f}", line['xy'][k], fontsize=8, color='b')
        a.set_aspect('equal')
    ax[0].set_title('Ground elevation (hillshade), route with station labels every 50 m')
    ax[1].set_title('green support, yellow known-unsupported, grey unknown, red hard, orange soft, magenta hard in walked corridor (prior)')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_tiles(m, aux, line, d, path, tile=50.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = m['origin'].astype(np.float64)
    img = class_image(aux, m)
    s, xy = line['s'], line['xy']
    n = int(np.ceil(s[-1] / tile))
    cols = 5
    rows_ = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(5 * cols, 5 * rows_), layout='constrained')
    for k, ax in enumerate(axs.ravel()):
        if k >= n:
            ax.axis('off'); continue
        sel = (s >= k * tile) & (s <= (k + 1) * tile)
        c = xy[sel].mean(0)
        half = max(np.ptp(xy[sel], 0).max() / 2 + 4, 10)
        i0 = int(np.floor((c[0] - half - origin[0]) / RES)); i1 = int(np.ceil((c[0] + half - origin[0]) / RES))
        j0 = int(np.floor((c[1] - half - origin[1]) / RES)); j1 = int(np.ceil((c[1] + half - origin[1]) / RES))
        i0, j0 = max(i0, 0), max(j0, 0)
        sub = img[i0:i1, j0:j1]
        ext = [origin[0] + i0 * RES - RES / 2, origin[0] + i1 * RES - RES / 2, origin[1] + j0 * RES - RES / 2, origin[1] + j1 * RES - RES / 2]
        ax.imshow(np.transpose(sub, (1, 0, 2)), origin='lower', extent=ext, interpolation='nearest')
        ax.plot(xy[sel, 0], xy[sel, 1], 'b-', lw=.6)
        if d['path'] is not None:
            nrm = np.stack([-np.sin(line['yaw']), np.cos(line['yaw'])], -1)
            pp = xy + nrm * d['path'][:, None]
            ax.plot(pp[sel, 0], pp[sel, 1], 'k--', lw=.6)
        bl = sel & ~d['any_band']
        ax.plot(xy[bl, 0], xy[bl, 1], 'x', color='k', ms=4)
        for q in range(0, len(s), 50):
            if sel[q]:
                ax.annotate(f'{s[q]:.0f}', xy[q], fontsize=6, color='b')
        ax.set_title(f's {k * tile:.0f}-{(k + 1) * tile:.0f} m', fontsize=9)
        ax.set_aspect('equal')
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_blocking(m, aux, line, rows, d, runs, path, half=3.0, maxn=30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = m['origin'].astype(np.float64)
    img = class_image(aux, m)
    s, xy, yaw = line['s'], line['xy'], line['yaw']
    walked = rows[line['walked_index'], 1:3]
    # merge runs closer than 2 m
    groups = []
    for r in runs:
        if groups and r['s_from_m'] - groups[-1][-1]['s_to_m'] < 2.:
            groups[-1].append(r)
        else:
            groups.append([r])
    groups = groups[:maxn]
    if not groups:
        return
    cols = 5
    rows_ = int(np.ceil(len(groups) / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(4.2 * cols, 4.2 * rows_), layout='constrained', squeeze=False)
    nrm = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    for ax, g in zip(axs.ravel(), groups):
        a, b = g[0]['first_station'], g[-1]['last_station']
        c = xy[(a + b) // 2]
        i0 = int(np.floor((c[0] - half - origin[0]) / RES)); i1 = int(np.ceil((c[0] + half - origin[0]) / RES))
        j0 = int(np.floor((c[1] - half - origin[1]) / RES)); j1 = int(np.ceil((c[1] + half - origin[1]) / RES))
        sub = img[i0:i1, j0:j1]
        ext = [origin[0] + i0 * RES - RES / 2, origin[0] + i1 * RES - RES / 2, origin[1] + j0 * RES - RES / 2, origin[1] + j1 * RES - RES / 2]
        ax.imshow(np.transpose(sub, (1, 0, 2)), origin='lower', extent=ext, interpolation='nearest')
        near = np.all(np.abs(walked - c) < half + 1, axis=1)
        ax.plot(walked[near, 0], walked[near, 1], '.', color='k', ms=1)
        sel = np.all(np.abs(xy - c) < half + 1, axis=1)
        ax.plot(xy[sel, 0], xy[sel, 1], 'b-', lw=.8)
        for o in (-.75, .75):
            pp = xy + nrm * o
            ax.plot(pp[sel, 0], pp[sel, 1], 'b:', lw=.5)
        ax.plot(xy[a:b + 1, 0], xy[a:b + 1, 1], 'c-', lw=2.5, alpha=.7)
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_title(f"blocked s {s[a]:.1f}-{s[b]:.1f} m  ({c[0]:.1f},{c[1]:.1f})", fontsize=8)
        ax.set_aspect('equal')
    for ax in axs.ravel()[len(groups):]:
        ax.axis('off')
    fig.suptitle('(c) blocking places: green support, yellow known-unsupported, grey unknown, red hard, orange soft, '
                 'magenta hard in walked corridor; black dots raw walked poses; blue centreline +-0.75 m; cyan blocked stations', fontsize=9)
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    main()
