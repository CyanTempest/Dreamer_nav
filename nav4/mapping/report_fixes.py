"""Report parts for fixes A-D: before/after numbers, remaining-blocker causes, figures.

Before = data/trail_a/v5_before_fix/{map,aux}.npz (copied before the fixes); after = the saved map.
"""
import json

import numpy as np

from nav4.mapping.analysis import ROBOT_W, blocking_runs, lookup
from nav4.mapping.common import DATA, RES, bilinear, footprint_world
from nav4.mapping.route import lateral_violations

V5 = DATA / 'v5_before_fix'


def load_v5():
    if not (V5 / 'map.npz').exists():
        return None
    with np.load(V5 / 'map.npz') as f:
        mm = {k: f[k] for k in ('layers', 'origin', 'route', 's', 'yaw')}
    keep = ('known', 'hard', 'soft', 'obstacle', 'support', 'exempt', 'cap_ok', 'walked_corridor', 'walked_cells',
            'obstacle_sdf_b', 'hard_evidence', 'dist_walked', 'cross_grade_deg', 'coverage')
    with np.load(V5 / 'aux.npz') as f:
        aa = {k: f[k] for k in keep}
    return dict(map=mm, aux=aa)


def _q(v):
    return dict(n=int(len(v)), **({f'p{q}': float(np.percentile(v, q)) for q in (5, 10, 25, 50, 75, 90)} if len(v) else {}))


def density_tables(aux, cfg):
    """Returns per range bin for: bare ground (returns within +-layer_thickness of the ground per 0.1 m cell)
    and obstacle cells by evidence (peak sub-bin returns)."""
    o = cfg['obstacles']
    edges = np.asarray(o['solid_density_range_bins_m'], float)
    d = aux['dist_walked']
    ob = aux['obstacle']
    bare = aux['observed'] & (aux['conc'] >= cfg['ground']['bare_concentration_min']) & (aux['n_body'] == 0) & \
        (aux['n_above'] == 0) & aux['known']
    pops = dict(bare_ground_layer_returns=(bare, aux['n_ground_layer']),
                continuous_column_peak=(ob & aux['continuous'], aux['peak_subbin']),
                vertical_line_peak=(ob & aux['vertical_line'], aux['peak_subbin']),
                vertical_plane_peak=(ob & aux['vertical_plane'], aux['peak_subbin']),
                surface_peak=(ob & aux['surface'], aux['peak_subbin']),
                railing_peak=(ob & aux['railing'], aux['peak_subbin']),
                soft_peak=(aux['soft'], aux['peak_subbin']),
                sparse_peak=(aux['sparse'], aux['peak_subbin']),
                hard_no_evidence_peak=(aux['hard'] & ~aux['hard_evidence'], aux['peak_subbin']))
    out = dict(range_bins_m=edges.tolist())
    for name, (mask, val) in pops.items():
        out[name] = [_q(val[mask & (d >= a) & (d < b)]) for a, b in zip(edges[:-1], edges[1:])]
    return out


def best_offset_cells(sup_sdf, obs_sdf, origin, xy, yaw, i, offsets):
    viol = lateral_violations(sup_sdf, obs_sdf, origin, xy[i:i + 1], yaw[i:i + 1], offsets)[0]
    k = int(np.argmin(viol))
    nrm = np.array([-np.sin(yaw[i]), np.cos(yaw[i])])
    w = footprint_world((xy[i] + nrm * offsets[k])[None], yaw[i:i + 1])[0]
    ij = np.floor((w - origin) / RES + .5).astype(np.int64)
    ij[:, 0] = ij[:, 0].clip(0, sup_sdf.shape[0] - 1)
    ij[:, 1] = ij[:, 1].clip(0, sup_sdf.shape[1] - 1)
    bs = sup_sdf[ij[:, 0], ij[:, 1]] > .025
    bo = bilinear(obs_sdf, origin, w) < 0
    return float(offsets[k]), int(viol[k]), np.unique(ij[bs], axis=0), np.unique(ij[bo], axis=0)


def blocker_causes(m, aux, xy, yaw, s, d, cfg):
    """Blocked stations (no feasible offset within the band): best offset, failing cells and their causes."""
    L = m['layers']
    origin = m['origin'].astype(np.float64)
    band = cfg['corridor']['acceptance_band_m']
    offsets = np.arange(-band, band + 1e-9, .05)
    t = cfg['terrain']
    blocked = np.flatnonzero(~d['any_band'])
    rows = []
    for i in blocked:
        o, nviol, cs, co = best_offset_cells(L[..., 2], L[..., 5], origin, xy, yaw, i, offsets)
        sc = {}
        if len(cs):
            a, b = cs[:, 0], cs[:, 1]
            sc = dict(cells=int(len(cs)), unknown=int((~aux['known'][a, b]).sum()),
                      step_over_trail_riser=int((aux['step_m'][a, b] > t['trail_zone_connect_max_step_m']).sum()),
                      grade_over_limit=int((np.nan_to_num(aux['grade_deg'][a, b], nan=90.) > t['max_grade_deg']).sum()),
                      outside_trail_zone=int((aux['dist_walked'][a, b] > t['trail_zone_halfwidth_m']).sum()),
                      not_connected_in_zone=int(((aux['dist_walked'][a, b] <= t['trail_zone_halfwidth_m']) & ~aux['trail_zone'][a, b]).sum()),
                      cross_grade_over_limit=int((np.nan_to_num(aux['cross_grade_deg'][a, b], nan=90.) > t['trail_zone_max_cross_grade_deg']).sum()),
                      max_step_m=float(aux['step_m'][a, b].max()))
        oc = {}
        if len(co):
            a, b = co[:, 0], co[:, 1]
            oc = dict(cells=int(len(co)), vertical_line=int(aux['vertical_line'][a, b].sum()),
                      vertical_plane=int(aux['vertical_plane'][a, b].sum()), surface=int(aux['surface'][a, b].sum()),
                      continuous=int(aux['continuous'][a, b].sum()), railing=int(aux['railing'][a, b].sum()),
                      solid_density_only=int((aux['solid_density'][a, b] & ~aux['hard_evidence'][a, b] & ~aux['near_evidence'][a, b]).sum()),
                      next_to_evidence_only=int((aux['near_evidence'][a, b] & ~aux['hard_evidence'][a, b]).sum()),
                      dist_walked_m=[float(v) for v in np.round(aux['dist_walked'][a, b], 2)],
                      height_above_ground_m=[float(v) for v in np.round(L[a, b, 1] - L[a, b, 0], 2)],
                      n_body=[int(v) for v in aux['n_body'][a, b]])
        cause = '+'.join(k for k, v in (('support', sc), ('obstacle', oc)) if v)
        rows.append(dict(station=int(i), s_m=float(s[i]), x=float(xy[i, 0]), y=float(xy[i, 1]), best_offset_m=o,
                         failing_samples=nviol, cause=cause, support=sc, obstacle=oc))
    runs = blocking_runs(d['any_band'], s, xy)
    for r in runs:
        sel = [q for q in rows if r['first_station'] <= q['station'] <= r['last_station']]
        r['cause'] = '/'.join(sorted({q['cause'] for q in sel}))
        r['support_cells_max'] = max([q['support'].get('cells', 0) for q in sel], default=0)
        r['obstacle_cells_max'] = max([q['obstacle'].get('cells', 0) for q in sel], default=0)
    return dict(stations=int(len(blocked)), runs=runs, station_detail=rows)


def fix_effects(m, aux, v5, line, per, results, cfg):
    L = m['layers']
    near2 = aux['dist_walked'] <= 2.
    zone = aux['dist_walked'] <= cfg['terrain']['trail_zone_halfwidth_m']
    out = {}
    # ---------------- A
    a = dict(sparse_cells=int(aux['sparse'].sum()), sparse_within_2m_of_walked_line=int((aux['sparse'] & near2).sum()),
             hard_cells=int(aux['hard'].sum()), soft_cells=int(aux['soft'].sum()),
             hard_default_within_2m=int((aux['hard_default'] & near2).sum()),
             hard_within_2m=int((aux['hard'] & near2).sum()),
             hard_to_sparse_within_2m_same_ground=int((aux['hard_default'] & aux['sparse'] & near2).sum()),
             solid_density_only_hard_within_2m=int((aux['hard'] & aux['solid_density'] & ~aux['hard_evidence'] & ~aux['near_evidence'] & near2).sum()),
             density_tables=density_tables(aux, cfg))
    if v5 is not None:
        va = v5['aux']
        a.update(v5_hard_within_2m=int((va['hard'] & near2).sum()),
                 v5_hard_now_sparse_within_2m=int((va['hard'] & aux['sparse'] & near2).sum()),
                 v5_hard_now_soft_within_2m=int((va['hard'] & aux['soft'] & near2).sum()),
                 v5_hard_now_not_obstacle_within_2m=int((va['hard'] & ~aux['obstacle'] & near2).sum()),
                 new_hard_not_v5_hard_within_2m=int((aux['hard'] & ~va['hard'] & near2).sum()))
    with open(DATA / 'forward' / 'metadata.json') as f:
        meta = json.load(f)
    a['solid_density_threshold_table'] = (meta.get('route_adaptation') or {}).get('solid_density')
    out['A_sparse'] = a
    # ---------------- B
    sup, sup5 = aux['support'], aux['support_v5_exemption']
    out['B_stair_exemption'] = dict(
        trail_zone_connect_max_step_m=cfg['terrain']['trail_zone_connect_max_step_m'],
        max_riser_in_flights_m=results['terrain']['max_riser_in_flights_m'],
        exempt_cells=int(aux['exempt'].sum()), exempt_cells_v5_rule=int(aux['exempt_v5'].sum()),
        zone_connected_cells=int(aux['trail_zone'].sum()), zone_connected_cells_v5_rule=int(aux['trail_zone_v5'].sum()),
        support_gained=int((sup & ~sup5).sum()), support_lost=int((~sup & sup5).sum()),
        support_changed_in_zone=int(((sup != sup5) & zone).sum()),
        cross_grade_on_walked_cells_deg=dict(robot_scale=_q(aux['cross_grade_deg'][aux['walked_cells'] & np.isfinite(aux['cross_grade_deg'])]),
                                            disc_0p3m=_q(aux['cross_grade_disc_deg'][aux['walked_cells']])),
        blocking_stations_with_B=results['c_new_with_prior']['blocking_stations'],
        blocking_stations_B_off=results['c_fixB_off_v5_exemption']['blocking_stations'])
    # ---------------- C
    deck = aux['deck']
    dz = L[..., 0] - aux['ground_before_deck']
    s_c = line['s'][aux['centreline_near']]
    void_list = []
    for i, j in np.argwhere(aux['void_walked']):
        void_list.append(dict(s_m=float(s_c[i, j]), x=float(m['origin'][0] + i * RES), y=float(m['origin'][1] + j * RES),
                              depth_below_prior_m=float(aux['prior_z'][i, j] - aux['ground_before_deck'][i, j]),
                              deck_found=bool(deck[i, j])))
    places = {}
    for lo, hi in ((676., 678.5), (717., 721.)):
        sel = aux['walked_cells'] & (s_c >= lo) & (s_c <= hi)
        rec = dict(walked_cells=int(sel.sum()), deck_cells_near=int((deck & (s_c >= lo - 1) & (s_c <= hi + 1)).sum()),
                   walked_supported_after=float((L[..., 2][sel] <= .025).mean()) if sel.any() else None,
                   walked_ground_minus_prior_after=_q((L[..., 0] - aux['prior_z'])[sel & np.isfinite(aux['prior_z'])]),
                   walked_ground_minus_prior_before=_q((aux['ground_before_deck'] - aux['prior_z'])[sel & np.isfinite(aux['prior_z'])]))
        if v5 is not None:
            rec['walked_supported_before_v5'] = float((v5['map']['layers'][..., 2][sel] <= .025).mean()) if sel.any() else None
        places[f's_{lo:g}_{hi:g}'] = rec
    c = dict(void_walked_cells=int(aux['void_walked'].sum()), seed_cells=int(aux['deck_seeds'].sum()),
             deck_cells=int(deck.sum()), void_without_deck=int(aux['void_without_deck'].sum()),
             ground_raised_cells=int((dz > .05).sum()), ground_raised_max_m=float(dz.max()),
             void_cells=sorted(void_list, key=lambda r: r['s_m']), places=places)
    if v5 is not None:
        ring = deck.copy()
        from scipy import ndimage as ndi
        ring = ndi.binary_dilation(deck, iterations=3)
        c.update(support_changed_near_deck_vs_v5=int(((L[..., 2] <= .025) != (v5['map']['layers'][..., 2] <= .025))[ring].sum()),
                 hard_changed_near_deck_vs_v5=int((aux['hard'] != v5['aux']['hard'])[ring].sum()))
    out['C_decks'] = c
    # ---------------- table
    table = {}
    names = [('before_v5_c', 'v5_c_with_prior'), ('after_c_on_centreline', 'c_new_with_prior'),
             ('after_fixA_off', 'c_fixA_off_sparse_as_hard'), ('after_fixB_off', 'c_fixB_off_v5_exemption'),
             ('after_saved_route', 'saved_route')]
    for label, key in names:
        if key not in results:
            continue
        r = results[key]
        table[label] = dict(blocking_stations=r['blocking_stations'], blocking_runs=len(r['blocking_runs']),
                            centreline_feasible_fraction=r['centreline_feasible_fraction'],
                            station_feasible_fraction_within_band=r['station_feasible_fraction_within_band'],
                            feasible_width_m=r['feasible_width_m'], continuous_path_within_band=r['continuous_path_within_band'],
                            continuous_swept_feasible=r['continuous_swept_feasible'])
    out['before_after'] = table
    shift = aux['route_shift_m']
    out['D_route'] = dict(length_m=float(m['s'][-1]), stations=int(len(m['s'])), max_abs_shift_m=float(np.abs(shift).max()),
                          shift_quantiles_m={str(q): float(np.percentile(np.abs(shift), q)) for q in (50, 90, 95, 99)},
                          shift_per_station_m=[round(float(v), 3) for v in shift],
                          route_centreline_s_m=[round(float(v), 2) for v in aux['route_centreline_s']],
                          adaptation={k: v for k, v in (meta.get('route_adaptation') or {}).items() if k != 'solid_density'})
    return out


# ------------------------------------------------------------------ figures
def class_image(aux, L, sparse=True, deck=True):
    known = aux['known']
    img = np.ones((*known.shape, 3), np.float32)
    img[known] = (.95, .85, .45)                          # known but unsupported
    img[L[..., 2] <= .025] = (.55, .8, .55)               # supported
    img[aux['soft']] = (.95, .6, .2)
    if sparse and 'sparse' in aux:
        img[aux['sparse']] = (.55, .75, .95)
    img[aux['hard']] = (.8, .1, .1)
    img[aux['hard'] & aux['walked_corridor']] = (.8, .2, .8)
    if deck and 'deck' in aux:
        img[aux['deck'] & (L[..., 2] <= .025)] = (.3, .65, .65)
    img[~aux['coverage']] = (1, 1, 1)
    img[aux['coverage'] & ~known] = (.6, .6, .6)
    return img


def fig_before_after(s, rs, per, results, aux, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(n, s) for n in ('v5_c_with_prior', 'c_new_with_prior') if n in per] + [('saved_route', rs)]
    fig, ax = plt.subplots(len(rows) + 2, 1, figsize=(18, 10), sharex=True, layout='constrained')
    for k, (n, ss) in enumerate(rows):
        d = per[n]
        sx = ss if n != 'saved_route' else np.interp(rs, rs, aux['route_centreline_s'])
        img = np.vstack([d['centre'], d['any_band']]).astype(float)
        ax[k].imshow(img, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest', extent=[sx[0], sx[-1], 1.5, -.5])
        r = results[n]
        ax[k].set_yticks([0, 1], ['offset 0', '+-0.75'])
        ax[k].set_title(f"{n}: offset-0 feasible {r['centreline_feasible_fraction']:.1%}, any offset {r['station_feasible_fraction_within_band']:.1%}, "
                        f"blocking stations {r['blocking_stations']} in {len(r['blocking_runs'])} runs, swept path={r['continuous_swept_feasible']}"
                        + (" (x = centreline s)" if n == 'saved_route' else ''), fontsize=9)
    k = len(rows)
    for n, col in (('v5_c_with_prior', '0.5'), ('c_new_with_prior', 'b')):
        if n in per:
            ax[k].plot(s, per[n]['width'], color=col, lw=.6, label=f'{n} (centreline)')
    ax[k].plot(aux['route_centreline_s'], per['saved_route']['width'], color='g', lw=.6, label='saved route')
    ax[k].axhline(ROBOT_W + .1, color='r', ls='--', lw=.8, label='0.58+0.10 m')
    ax[k].set_ylim(0, 3.7); ax[k].legend(fontsize=8, ncol=4); ax[k].set_ylabel('feasible width (m)')
    ax[k + 1].plot(aux['route_centreline_s'], aux['route_shift_m'], 'm-', lw=.6)
    ax[k + 1].axhline(0, color='k', lw=.5)
    ax[k + 1].set_ylabel('route - centreline (m, +left)'); ax[k + 1].set_xlabel('centreline station s (m)')
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _section_panel(ax, slab, centre, yaw, ground_of, labels, title, route_off=None, walked_off=(), half=2.5):
    idx, lat = slab.slab(centre, yaw, half_lateral=half, half_along=.1)
    z = slab.z[idx]
    offs = np.arange(-half, half + 1e-6, .05)
    g = ground_of(offs)
    gi = np.interp(lat, offs, np.nan_to_num(g, nan=np.nanmin(g) if np.isfinite(g).any() else 0.))
    sc = ax.scatter(lat, z, c=np.clip(z - gi, -.3, 1.5), s=2, cmap='turbo', vmin=-.3, vmax=1.5, linewidths=0)
    ax.plot(offs, g, 'k-', lw=1.2, label='map ground')
    if route_off is not None:
        ax.axvspan(route_off - ROBOT_W / 2, route_off + ROBOT_W / 2, color='m', alpha=.12, zorder=0)
        ax.axvline(route_off, color='m', ls='--', lw=.8)
    ax.axvline(0, color='b', ls=':', lw=.8)
    for w in walked_off:
        ax.axvline(w, color='g', lw=.5, ls='-', alpha=.4)
    base = np.nanmin(g) if np.isfinite(g).any() else z.min()
    for k, (name, (mask, col)) in enumerate(labels.items()):
        y0 = base - .2 - .1 * k
        ax.scatter(offs[mask], np.full(mask.sum(), y0), s=5, marker='s', color=col, linewidths=0)
        ax.text(-half, y0, name, fontsize=6, va='center', ha='left')
    ax.set_ylim(base - .3 - .1 * len(labels), base + 2.2)
    ax.set_xlim(-half, half)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('lateral offset from scanner centreline (m, +left)')
    ax.set_ylabel('z (m)')
    return sc


def _labels(aux, L, origin, pts, v5=False):
    lk = lambda a: lookup(a, origin, pts, False)
    lab = {'support': (lk(L[..., 2] <= .025), '#2a9d8f'),
           'exempt (trail)': (lk(aux['exempt'] & ~aux['cap_ok']), '#4361ee'),
           'hard': (lk(aux['hard'] & ~aux['walked_corridor']), '#c1121f'),
           'hard in walked corr.': (lk(aux['hard'] & aux['walked_corridor']), '#c77dff'),
           'soft': (lk(aux['soft']), '#f4a261')}
    if not v5:
        lab['sparse (A)'] = (lk(aux['sparse']), '#74a9e8')
        lab['deck (C)'] = (lk(aux['deck']), '#1b7f7f')
    lab['unknown'] = (~lk(aux['known']), '#8d99ae')
    return lab


def _topview(ax, img, origin, c, half, line, route_xy, walked, title, cut=None):
    i0 = int(np.floor((c[0] - half - origin[0]) / RES)); i1 = int(np.ceil((c[0] + half - origin[0]) / RES))
    j0 = int(np.floor((c[1] - half - origin[1]) / RES)); j1 = int(np.ceil((c[1] + half - origin[1]) / RES))
    sub = img[i0:i1, j0:j1]
    ext = [origin[0] + i0 * RES - RES / 2, origin[0] + i1 * RES - RES / 2, origin[1] + j0 * RES - RES / 2, origin[1] + j1 * RES - RES / 2]
    ax.imshow(np.transpose(sub, (1, 0, 2)), origin='lower', extent=ext, interpolation='nearest')
    near = np.all(np.abs(walked - c) < half + 1, axis=1)
    ax.plot(walked[near, 0], walked[near, 1], '.', color='k', ms=1)
    sel = np.all(np.abs(line['xy'] - c) < half + 1, axis=1)
    ax.plot(line['xy'][sel, 0], line['xy'][sel, 1], 'b:', lw=1)
    if route_xy is not None:
        sel = np.all(np.abs(route_xy - c) < half + 1, axis=1)
        ax.plot(route_xy[sel, 0], route_xy[sel, 1], 'm-', lw=1.2)
    if cut is not None:
        ax.plot(cut[:, 0], cut[:, 1], 'k-', lw=.6)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)


def _route_offset_at(aux, line, i):
    """Lateral offset (from the centreline at station i) where the saved route crosses the section."""
    return float(aux['route_offsets_at_centreline'][i])


def _station_figure(path, slab, m, aux, v5, line, walked_all, i, title_extra=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    L = m['layers']
    origin = m['origin'].astype(np.float64)
    c, yw = line['xy'][i], line['yaw'][i]
    nrm = np.array([-np.sin(yw), np.cos(yw)])
    half = 2.5
    offs = np.arange(-half, half + 1e-6, .05)
    pts = c[None] + offs[:, None] * nrm[None]
    d = walked_all - c
    along = d @ np.array([np.cos(yw), np.sin(yw)])
    wl = (d @ nrm)[np.abs(along) <= .15]
    ro = _route_offset_at(aux, line, i)
    fig, axs = plt.subplots(2, 2, figsize=(17, 9), layout='constrained', gridspec_kw=dict(width_ratios=[2.2, 1]))
    sc = _section_panel(axs[0, 0], slab, c, yw, lambda o: lookup(L[..., 0], origin, c[None] + o[:, None] * nrm[None], np.nan),
                        _labels(aux, L, origin, pts), f'AFTER (fixes A-D) s={line["s"][i]:.1f} m ({c[0]:.1f},{c[1]:.1f}); '
                        f'magenta = saved route ({ro:+.2f} m) with robot width, green = walked poses within 0.15 m{title_extra}',
                        route_off=ro, walked_off=wl, half=half)
    _topview(axs[0, 1], class_image(aux, L), origin, c, 3., line, m['route'][:, :2], walked_all,
             'after: green support, teal deck, yellow known-unsupp., grey unknown,\nred hard, magenta hard in corridor, orange soft, blue sparse',
             cut=pts)
    if v5 is not None:
        L5, a5 = v5['map']['layers'], v5['aux']
        o5 = v5['map']['origin'].astype(np.float64)
        _section_panel(axs[1, 0], slab, c, yw, lambda o: lookup(L5[..., 0], o5, c[None] + o[:, None] * nrm[None], np.nan),
                       _labels(a5, L5, o5, pts, v5=True), 'BEFORE (v5): ground and labels of the v5 map, colours = height above v5 ground',
                       walked_off=wl, half=half)
        _topview(axs[1, 1], class_image(a5, L5, sparse=False, deck=False), o5, c, 3., line, None, walked_all, 'before (v5)', cut=pts)
    fig.colorbar(sc, ax=axs[:, 0], shrink=.5, label='height above map ground (m)')
    fig.savefig(path, dpi=85)
    plt.close(fig)


def _slab_and_walked(line, rows):
    from nav4.mapping.grid import load_points
    from nav4.mapping.sections import PointSlab
    return PointSlab(load_points(('x', 'y', 'z', 'cls', 'nz'))), rows[line['walked_index'], 1:3]


def fix_sections(m, aux, v5, line, rows, fig_dir, cfg, slab=None):
    slab_, walked = _slab_and_walked(line, rows) if slab is None else (slab, rows[line['walked_index'], 1:3])
    out = []
    for s0 in cfg['report']['fix_sections_s_m']:
        i = int(np.argmin(np.abs(line['s'] - s0)))
        path = fig_dir / f'fix_sections_s{int(round(s0)):03d}.png'
        _station_figure(path, slab_, m, aux, v5, line, walked, i)
        out.append(dict(s_m=float(line['s'][i]), x=float(line['xy'][i, 0]), y=float(line['xy'][i, 1]), file=path.name))
    return out


def blocker_sections(m, aux, v5, line, rows, blockers, fig_dir, cfg, slab=None, maxn=40):
    """One figure per remaining blocking run (middle station, section across the scanner centreline)."""
    if not blockers['runs']:
        return []
    slab_, walked = _slab_and_walked(line, rows) if slab is None else (slab, rows[line['walked_index'], 1:3])
    out = []
    cs = aux['route_centreline_s']
    for r in blockers['runs'][:maxn]:
        mid = (r['first_station'] + r['last_station']) // 2
        s_c = float(cs[mid])
        i = int(np.argmin(np.abs(line['s'] - s_c)))
        path = fig_dir / f'fix_blocker_s{line["s"][i]:06.1f}.png'
        _station_figure(path, slab_, m, aux, v5, line, walked, i,
                        title_extra=f'\nREMAINING BLOCKER route s {r["s_from_m"]:.1f}-{r["s_to_m"]:.1f}: {r["cause"]}')
        out.append(dict(route_s_from_m=r['s_from_m'], route_s_to_m=r['s_to_m'], centreline_s_m=float(line['s'][i]),
                        cause=r['cause'], file=path.name))
    return out
