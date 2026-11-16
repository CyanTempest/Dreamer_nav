"""Task 2b: empirical characterisation of the undocumented classification codes.

Uses the cropped planning cloud and the NEW ground (forward/map.npz layers[...,0],
known flag) for height above ground.  Writes data/trail_a/classes.json and
figures/classes.png.  Nothing here feeds back into the model except the codes the
report lists as 'linear structures' (13, 14), which the config may use as HARD
evidence only.
"""
import time

import numpy as np

from nav4.mapping.common import DATA, RES, write_json
from nav4.mapping.grid import load_centreline, load_points

CODES = [1, 4, 5, 6, 7, 12, 13, 14, 23, 24]


def main():
    t0 = time.time()
    with np.load(DATA / 'forward' / 'map.npz') as f:
        layers = f['layers']
        origin = f['origin'].astype(np.float64)
    with np.load(DATA / 'forward' / 'aux.npz') as f:
        aux = {k: f[k] for k in ('known', 'route_dist', 'route_near', 'walked_corridor', 'obstacle', 'hard', 'soft')}
    ground = layers[..., 0]
    shape = ground.shape
    line = load_centreline()
    P = load_points(('x', 'y', 'z', 'nz', 'cls', 'intensity', 't'))
    x, y, z = np.asarray(P['x']), np.asarray(P['y']), np.asarray(P['z'])
    cls = np.asarray(P['cls'])
    i = np.floor((x - origin[0]) / RES + .5).astype(np.int64)
    j = np.floor((y - origin[1]) / RES + .5).astype(np.int64)
    inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    i[~inside] = 0
    j[~inside] = 0
    known = aux['known'][i, j] & inside
    hag = (z - ground[i, j]).astype(np.float32)
    dist = aux['route_dist'][i, j]
    station = line['s'][aux['route_near'][i, j]]
    absnz = np.abs(np.asarray(P['nz']).astype(np.float32))
    inten = np.asarray(P['intensity']).astype(np.float32)
    corridor = aux['walked_corridor'][i, j] & inside
    body = known & (hag > .22) & (hag < .8)
    hardc = aux['hard'][i, j] & body
    softc = aux['soft'][i, j] & body
    total = len(cls)
    out = dict(points=int(total), note='Codes are undocumented; statistics are empirical, HAG uses the new segment-A ground on known cells.')
    per = {}
    for c in CODES:
        m = cls == c
        mk = m & known
        n = int(m.sum())
        rec = dict(points=n, share=n / total,
                   share_within_3m_of_route=float((m & (dist < 3)).sum() / max(1, (dist < 3).sum())),
                   hag_quantiles_m={str(q): float(np.percentile(hag[mk], q)) for q in (5, 25, 50, 75, 95)} if mk.any() else None,
                   frac_hag_abs_lt_0p05=float((np.abs(hag[mk]) < .05).mean()) if mk.any() else None,
                   frac_body_band=float(body[m].mean()),
                   abs_nz_quantiles={str(q): float(np.percentile(absnz[m], q)) for q in (10, 50, 90)},
                   frac_abs_nz_gt_0p9=float((absnz[m] > .9).mean()), frac_abs_nz_lt_0p3=float((absnz[m] < .3).mean()),
                   intensity_median=float(np.median(inten[m])),
                   z_quantiles_m={str(q): float(np.percentile(z[m], q)) for q in (5, 50, 95)},
                   route_distance_median_m=float(np.median(dist[m & inside])),
                   share_of_walked_corridor_body_points=float((m & body & corridor).sum() / max(1, (body & corridor).sum())),
                   share_of_hard_body_points=float((m & hardc).sum() / max(1, hardc.sum())),
                   share_of_soft_body_points=float((m & softc).sum() / max(1, softc.sum())),
                   station_hist_50m=np.histogram(station[m & inside & (dist < 4)], bins=np.arange(0, 800, 50))[0].tolist())
        per[str(c)] = rec
    out['codes'] = per
    # Column test: are codes semantic (ground vs elevated get different codes) or location-like?
    near = inside & known & (dist < 4)
    col = (i // 10) * (shape[1] // 10 + 1) + (j // 10)   # 1 m columns
    gsel = near & (np.abs(hag) < .05)
    esel = near & (hag > .5) & (hag < 3.)
    def dominant(sel):
        key = col[sel].astype(np.int64) * 32 + cls[sel]
        u, cnt = np.unique(key, return_counts=True)
        cc, code = u // 32, u % 32
        order = np.lexsort((-cnt, cc))
        cc, code = cc[order], code[order]
        first = np.r_[True, np.diff(cc) > 0]
        return dict(zip(cc[first].tolist(), code[first].tolist()))
    dg, de = dominant(gsel), dominant(esel)
    common = [k for k in dg if k in de]
    same = np.array([dg[k] == de[k] for k in common])
    pairs = {}
    for k in common:
        pairs.setdefault(f'{dg[k]}->{de[k]}', 0)
        pairs[f'{dg[k]}->{de[k]}'] += 1
    out['column_test'] = dict(
        description='1 m columns within 4 m of the route: dominant code of ground-level points (|HAG|<5 cm) vs of elevated points (0.5-3 m). '
                    'A semantic ground/vegetation scheme would give different codes; a location/partition-like code gives the same.',
        columns=len(common), same_dominant_code_fraction=float(same.mean()) if len(same) else None,
        top_pairs=dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:12]))
    out['seconds'] = time.time() - t0
    write_json(DATA / 'classes.json', out)
    plot(per, DATA / 'figures' / 'classes.png', hag, absnz, cls, known)
    print(out['column_test'])
    for c in CODES:
        r = per[str(c)]
        print(c, f"{r['share']:.4f}", r['hag_quantiles_m'], f"nz>0.9 {r['frac_abs_nz_gt_0p9']:.2f} nz<0.3 {r['frac_abs_nz_lt_0p3']:.2f}",
              f"corr {r['share_of_walked_corridor_body_points']:.3f} hard {r['share_of_hard_body_points']:.3f} soft {r['share_of_soft_body_points']:.3f}")


def plot(per, path, hag, absnz, cls, known):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nav4.mapping.sections import CLASS_COLORS
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), layout='constrained')
    rng = np.random.default_rng(0)
    for c in CODES:
        m = np.flatnonzero((cls == c) & known)
        if len(m) > 200000:
            m = rng.choice(m, 200000, replace=False)
        if len(m) < 50:
            continue
        ax[0].hist(np.clip(hag[m], -.5, 6), bins=130, histtype='step', density=True, color=CLASS_COLORS[c], label=f'{c} ({per[str(c)]["share"]:.1%})')
        ax[1].hist(absnz[m], bins=50, histtype='step', density=True, color=CLASS_COLORS[c], label=str(c))
    ax[0].set(xlabel='height above new ground (m)', ylabel='density', title='HAG by code'); ax[0].legend(fontsize=7)
    ax[0].set_yscale('log')
    ax[1].set(xlabel='|normal_z|', title='|normal_z| by code'); ax[1].legend(fontsize=7)
    bins = np.arange(0, 800, 50)
    bottom = np.zeros(len(bins) - 1)
    for c in CODES:
        h = np.array(per[str(c)]['station_hist_50m'], float)
        tot = np.sum([per[str(k)]['station_hist_50m'] for k in CODES], 0).astype(float)
        frac = h / np.maximum(tot, 1)
        ax[2].bar(bins[:-1], frac, width=48, bottom=bottom, color=CLASS_COLORS[c], label=str(c), align='edge')
        bottom += frac
    ax[2].set(xlabel='route station (m)', ylabel='share of points within 4 m', title='Code mix along segment A'); ax[2].legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
