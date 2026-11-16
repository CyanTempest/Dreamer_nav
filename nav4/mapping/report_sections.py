"""Cross-sections at failing places: old-proxy labels (task 2) and new labels (task 5)."""
import numpy as np

from nav4.mapping.analysis import lookup
from nav4.mapping.common import DATA
from nav4.mapping.grid import load_points
from nav4.mapping.sections import PointSlab, cut_profile, draw_section

FIG = DATA / 'figures'


def pick_stations(line, per, aux, origin, stairs, g1):
    """Deterministic picks; returns list of (station, reason)."""
    s = line['s']
    with np.load(DATA / 'diagnosis' / 'old_proxy_stations.npz') as f:
        cat = f['category']
        lane = f['lane']
    picks = []

    def add(i, why):
        i = int(i)
        if all(abs(s[i] - s[j]) > 5 for j, _ in picks):
            picks.append((i, why))

    for c in ('slope_only', 'obstacle_only', 'combined'):
        m = cat == c
        e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        if len(e):
            runs = sorted(zip(e[1::2] - e[::2], e[::2], e[1::2]), reverse=True)
            for L, a, b in runs[:2]:
                add((a + b - 1) // 2, f'old proxy longest {c} run (s {s[a]:.1f}-{s[b - 1]:.1f} m)')
    add(np.argmin(lane), f'old proxy narrowest clear lane ({lane.min():.2f} m)')
    if stairs:
        f0 = max(stairs, key=lambda f: f['steps'])
        add(np.argmin(np.abs(s - (f0['s_from_m'] + f0['s_to_m']) / 2)), f"longest stair flight ({f0['steps']} risers)")
    k = int(np.nanargmax(np.abs(g1)))
    add(np.argmin(np.abs(s - (k + 10) * .05)), f'steepest 1 m grade ({abs(g1[k]):.0f} deg)')
    # railing: most linear-structure (13/14) body cells within 1 m of the centreline
    rail = aux['railing'] if 'railing' in aux else aux['n_rail'] >= 2
    nrm = np.stack([-np.sin(line['yaw']), np.cos(line['yaw'])], -1)
    cnt = np.zeros(len(s))
    for o in np.arange(-1., 1.01, .1):
        cnt += lookup(rail, origin, line['xy'] + nrm * o, False)
    add(np.argmax(cnt), 'most posts/handrail (codes 13/14) within 1 m')
    c = per['c_new_with_prior']
    if (~c['any_band']).any():
        bad = np.flatnonzero(~c['any_band'])
        add(bad[len(bad) // 2], 'new model (c) blocking station')
    w = c['width']
    add(np.argmin(np.where(c['any_band'], w, np.inf)) if c['any_band'].any() else 0, f'new model (c) narrowest feasible width')
    return picks


def render_sections(m, aux, old, line, per, cfg, stairs, g1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = m['origin'].astype(np.float64)
    oorigin = old['origin'].astype(np.float64)
    with np.load(DATA / 'old_proxy' / 'old_proxy_masks.npz') as f:
        om = {k: f[k] for k in ('support', 'obstacle', 'known')}
    picks = pick_stations(line, per, aux, origin, stairs, g1)
    slab = PointSlab(load_points(('x', 'y', 'z', 'cls', 'nz')))
    offs = np.arange(-4, 4.0001, .05)
    s, xy, yaw = line['s'], line['xy'], line['yaw']
    out = []
    for label, which in (('old', 0), ('new', 1)):
        n = len(picks)
        fig, axs = plt.subplots(n, 2, figsize=(17, 3.7 * n), layout='constrained')
        axs = np.atleast_2d(axs)
        for r, (i, why) in enumerate(picks):
            idx, lat = slab.slab(xy[i], yaw[i])
            walked = []
            nrm = np.array([-np.sin(yaw[i]), np.cos(yaw[i])])
            cut = xy[i][None] + offs[:, None] * nrm[None]
            corr = lookup(aux['walked_corridor'], origin, cut, False)
            if which == 0:
                g = cut_profile(old['layers'][..., 0], oorigin, xy[i], yaw[i], offs)
                lab = {'old support': (cut_profile(om['support'], oorigin, xy[i], yaw[i], offs, False).astype(bool), '#2a9d8f'),
                       'old obstacle': (cut_profile(om['obstacle'], oorigin, xy[i], yaw[i], offs, False).astype(bool), '#e76f51'),
                       'old unknown': (~cut_profile(om['known'], oorigin, xy[i], yaw[i], offs, False).astype(bool), '#8d99ae')}
                title = f'[OLD] st {i} s={s[i]:.1f} m ({xy[i, 0]:.1f},{xy[i, 1]:.1f}): {why}'
                path_off = None
            else:
                g = cut_profile(m['layers'][..., 0], origin, xy[i], yaw[i], offs)
                known = lookup(m['atlas'][..., 19] > .5, origin, cut, False)
                lab = {'support': (lookup(m['layers'][..., 2] <= .025, origin, cut, False), '#2a9d8f'),
                       'exempt (trail)': (lookup(aux['exempt'] & ~aux['cap_ok'], origin, cut, False), '#4361ee'),
                       'hard': (lookup(aux['hard'] & ~aux['walked_corridor'], origin, cut, False), '#c1121f'),
                       'hard in corridor': (lookup(aux['hard'] & aux['walked_corridor'], origin, cut, False), '#c77dff'),
                       'soft': (lookup(aux['soft'], origin, cut, False), '#f4a261'),
                       'unknown': (~known, '#8d99ae'),
                       'walked corr.': (corr, '#2b9348')}
                title = (f'[NEW] st {i} s={s[i]:.1f} m: {why}; (c) width {per["c_new_with_prior"]["width"][i]:.2f} m, '
                         f'feasible in band={bool(per["c_new_with_prior"]["any_band"][i])}')
                pth = per['c_new_with_prior']['path']
                path_off = float(pth[i]) if pth is not None else None
            draw_section(axs[r, 0], axs[r, 1], slab, idx, lat, g, offs, title, lab, path_offset=path_off)
            if which == 1:
                out.append(dict(station=int(i), s_m=float(s[i]), x=float(xy[i, 0]), y=float(xy[i, 1]), reason=why))
        fig.savefig(FIG / f'sections_{label}.png', dpi=85)
        plt.close(fig)
    return out
