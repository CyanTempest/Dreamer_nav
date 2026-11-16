"""Task 1: objective definition of segment A and a clean centreline.

Segment A = the part of the scanner path that is off the official 2D nav map's
free (paved) area, i.e. the longest contiguous run of poses with no nav-map free
cell within ``navmap_free_dilation_m``, plus ``approach_m`` of centreline at each
end.  The centreline removes pauses (distance thinning) and local back-and-forth
(loop pruning: whenever the walker returns to within ``loop_close_radius_m`` (XY,
and ``loop_close_dz_m`` in z) of a kept point after walking at least
``loop_min_arc_m`` away, the excursion is cut), is smoothed along arc length and
resampled at 0.2 m XY arc length.  Each station records the original timestamp
and the nearest original poses.csv row.

Usage: python -m nav4.mapping.segment
"""
import argparse
import time

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d

from nav4.mapping.common import DATA, load_config, load_navmap, load_poses, navmap_lookup, write_json


def off_navmap_run(poses, nav, dilation_m):
    """Longest (in XY path length) contiguous run of poses not on dilated free nav-map cells."""
    free = nav['free']
    radius = int(round(dilation_m / nav['res']))
    if radius > 0:
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        free = binary_dilation(free, structure=(xx ** 2 + yy ** 2) <= radius ** 2)
    on = navmap_lookup(nav, free, poses[:, 1:3])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[:, 1:3], axis=0), axis=1))]
    edges = np.flatnonzero(np.diff(np.r_[1, on.astype(np.int8), 1]))
    runs = [(edges[k], edges[k + 1] - 1) for k in range(0, len(edges), 2)]
    lengths = [arc[j] - arc[i] for i, j in runs]
    best = int(np.argmax(lengths))
    i, j = runs[best]
    return dict(exit_index=int(i), entry_index=int(j + 1), exit_arc_raw_m=float(arc[i]),
                entry_arc_raw_m=float(arc[j + 1]), run_length_raw_m=float(lengths[best]),
                second_longest_run_m=float(sorted(lengths)[-2]) if len(lengths) > 1 else 0.,
                on_free=on, raw_arc=arc)


def thin_pauses(rows, min_step):
    """Keep a pose only once it is >= min_step (XY) from the last kept pose."""
    keep = [0]
    last = rows[0, 1:3]
    for k in range(1, len(rows)):
        if np.hypot(*(rows[k, 1:3] - last)) >= min_step:
            keep.append(k)
            last = rows[k, 1:3]
    return np.array(keep)


def prune_loops(xyz, radius, dz, min_arc, max_arc):
    """Remove excursions that come back to an already kept point.

    Returns the indices (into xyz) of the kept, monotonically-timed points and
    a list of pruned (first_removed, rejoin) index pairs.
    """
    step = np.r_[0, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)]
    arc = np.cumsum(step)
    kept = [0]
    pruned = []
    for i in range(1, len(xyz)):
        k_idx = np.asarray(kept)
        window = (arc[i] - arc[k_idx] >= min_arc) & (arc[i] - arc[k_idx] <= max_arc)
        if window.any():
            cand = k_idx[window]
            close = (np.hypot(*(xyz[cand, :2] - xyz[i, :2]).T) < radius) & (np.abs(xyz[cand, 2] - xyz[i, 2]) < dz)
            if close.any():
                k = int(cand[np.flatnonzero(close)[0]])
                cut = kept.index(k)
                if cut + 1 < len(kept):
                    pruned.append((int(kept[cut + 1]), int(i)))
                kept = kept[:cut + 1]
        kept.append(i)
    return np.array(kept), pruned


def build_centreline(rows, cfg):
    seg = cfg['segment']
    thin = thin_pauses(rows, seg['pause_min_step_m'])
    sub = rows[thin]
    kept, pruned = prune_loops(sub[:, 1:4], seg['loop_close_radius_m'], seg['loop_close_dz_m'],
                               seg['loop_min_arc_m'], seg['loop_max_arc_m'])
    path = sub[kept]
    # Smooth along XY arc length (uniform 5 cm reparametrisation first).
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, 1:3], axis=0), axis=1))]
    fine_s = np.arange(0, arc[-1], .05)
    fine = np.stack([np.interp(fine_s, arc, path[:, k]) for k in range(4)], -1)  # t x y z
    sigma = seg['smoothing_sigma_m'] / .05
    smooth = fine.copy()
    for k in (1, 2, 3):
        smooth[:, k] = gaussian_filter1d(fine[:, k], sigma, mode='nearest')
    arc2 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(smooth[:, 1:3], axis=0), axis=1))]
    s = np.arange(0, arc2[-1], seg['station_spacing_m'])
    stations = np.stack([np.interp(s, arc2, smooth[:, k]) for k in range(4)], -1)
    # Merged excursions: maximal runs of thinned poses that are not kept.
    removed = np.setdiff1d(np.arange(len(sub)), kept)
    runs = []
    if len(removed):
        breaks = np.flatnonzero(np.diff(removed) > 1)
        starts = np.r_[removed[0], removed[breaks + 1]]
        ends = np.r_[removed[breaks], removed[-1]]
        runs = [(int(thin[a]), int(thin[min(b + 1, len(thin) - 1)])) for a, b in zip(starts, ends)]
    return dict(s=s, t=stations[:, 0], xy=stations[:, 1:3], z_scanner=stations[:, 3],
                thin_index=thin, kept=kept, pruned=runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DATA / 'segment'))
    args = parser.parse_args()
    begin = time.time()
    cfg, _ = load_config()
    seg = cfg['segment']
    rows = load_poses()
    nav = load_navmap()
    run = off_navmap_run(rows, nav, seg['navmap_free_dilation_m'])
    sens = {}
    for d in seg['dilation_sensitivity_m']:
        r = off_navmap_run(rows, nav, d)
        sens[str(d)] = dict(exit_arc_raw_m=r['exit_arc_raw_m'], entry_arc_raw_m=r['entry_arc_raw_m'],
                            exit_index=r['exit_index'], entry_index=r['entry_index'])
    raw_arc = run['raw_arc']
    # Work window generously beyond both ends, then cut on the cleaned centreline.
    lo = int(np.searchsorted(raw_arc, raw_arc[run['exit_index']] - seg['work_margin_raw_m']))
    hi = int(np.searchsorted(raw_arc, raw_arc[run['entry_index']] + seg['work_margin_raw_m']))
    window = rows[lo:hi + 1]
    line = build_centreline(window, cfg)
    t_exit = rows[run['exit_index'], 0]
    t_entry = rows[run['entry_index'], 0]
    s_exit = float(line['s'][np.searchsorted(line['t'], t_exit)])
    s_entry = float(line['s'][np.searchsorted(line['t'], t_entry)])
    s0 = max(0., s_exit - seg['approach_m'])
    s1 = min(float(line['s'][-1]), s_entry + seg['approach_m'])
    take = (line['s'] >= s0 - 1e-6) & (line['s'] <= s1 + 1e-6)
    s = line['s'][take] - line['s'][take][0]
    xy = line['xy'][take]
    z_scanner = line['z_scanner'][take]
    t = line['t'][take]
    raw_index = np.clip(np.searchsorted(rows[:, 0], t), 1, len(rows) - 1)
    raw_index -= (np.abs(rows[raw_index - 1, 0] - t) < np.abs(rows[raw_index, 0] - t)).astype(int)
    yaw = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    # Raw poses physically walked within segment A (incl. pruned excursions): walked-corridor prior.
    t0, t1 = t[0], t[-1]
    walked = np.flatnonzero((rows[:, 0] >= t0) & (rows[:, 0] <= t1))
    pruned_abs = [(lo + a, lo + b) for a, b in line['pruned']]
    pruned_in = [(a, b) for a, b in pruned_abs if rows[b, 0] >= t0 and rows[a, 0] <= t1]
    raw_len = float(np.linalg.norm(np.diff(rows[walked, 1:3], axis=0), axis=1).sum())
    out = DATA / 'segment'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'centreline.npz', s=s.astype(np.float64), xy=xy, z_scanner=z_scanner,
                        t=t, raw_index=raw_index, yaw=yaw, walked_index=walked,
                        pruned=np.array(pruned_in, np.int64).reshape(-1, 2))
    summary = dict(
        definition=('Longest contiguous run of 10 Hz poses with no official nav-map FREE cell (PGM value 254) '
                    f'within {seg["navmap_free_dilation_m"]} m, plus {seg["approach_m"]} m of cleaned centreline '
                    'before the exit and after the re-entry.'),
        exit=dict(raw_index=run['exit_index'], timestamp=float(t_exit), raw_xy_path_m=run['exit_arc_raw_m'],
                  xyz=rows[run['exit_index'], 1:4].tolist(), centreline_s_m=s_exit - s0),
        entry=dict(raw_index=run['entry_index'], timestamp=float(t_entry), raw_xy_path_m=run['entry_arc_raw_m'],
                   xyz=rows[run['entry_index'], 1:4].tolist(), centreline_s_m=s_entry - s0),
        off_navmap_run_raw_m=run['run_length_raw_m'], second_longest_off_run_raw_m=run['second_longest_run_m'],
        dilation_sensitivity=sens,
        start=dict(xy=xy[0].tolist(), scanner_z=float(z_scanner[0]), timestamp=float(t[0]), raw_index=int(raw_index[0])),
        end=dict(xy=xy[-1].tolist(), scanner_z=float(z_scanner[-1]), timestamp=float(t[-1]), raw_index=int(raw_index[-1])),
        length_m=float(s[-1]), stations=len(s), station_spacing_m=seg['station_spacing_m'],
        raw_walked_xy_length_m=raw_len, raw_walked_poses=len(walked),
        duration_s=float(t[-1] - t[0]),
        pruned_excursions=[dict(from_raw_index=int(a), rejoin_raw_index=int(b),
                                raw_xy_length_m=float(raw_arc[b] - raw_arc[a]),
                                seconds=float(rows[b, 0] - rows[a, 0]),
                                xy=rows[a, 1:3].tolist()) for a, b in pruned_in],
        parameters=seg, seconds=time.time() - begin)
    write_json(out / 'segment.json', summary)
    figure(rows, nav, run, line, s0, s1, s, xy, z_scanner, t, walked, pruned_in, out / 'segment.png')
    print({k: v for k, v in summary.items() if k not in ('pruned_excursions', 'parameters')})
    print('pruned excursions', len(pruned_in), 'longest', max([e['raw_xy_length_m'] for e in summary['pruned_excursions']] or [0]))


def figure(rows, nav, run, line, s0, s1, s, xy, z_scanner, t, walked, pruned, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = nav['image']
    h, w = img.shape
    ext = [nav['origin'][0], nav['origin'][0] + w * nav['res'], nav['origin'][1], nav['origin'][1] + h * nav['res']]
    fig = plt.figure(figsize=(18, 11), layout='constrained')
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(np.where(img == 254, 1., np.where(img == 0, 0., .6)), extent=ext, cmap='gray', vmin=0, vmax=1)
    ax.plot(rows[:, 1], rows[:, 2], color='0.6', lw=.5, label='raw poses (whole run)')
    ax.plot(xy[:, 0], xy[:, 1], 'r-', lw=1., label='segment A centreline')
    for k in range(0, len(s), 250):
        ax.annotate(f'{s[k]:.0f}', xy[k], fontsize=7, color='b')
    ex, en = rows[run['exit_index'], 1:3], rows[run['entry_index'], 1:3]
    ax.plot(*ex, 'g^', ms=9, label='exit nav-map free area')
    ax.plot(*en, 'mv', ms=9, label='re-entry')
    ax.set_aspect('equal'); ax.legend(fontsize=8, loc='lower right'); ax.set_title('Segment A (nav map: white=free)')
    for col, (centre, title) in enumerate([(ex, 'exit'), (en, 're-entry')]):
        a = fig.add_subplot(gs[0, 1 + col])
        a.imshow(np.where(img == 254, 1., np.where(img == 0, 0., .6)), extent=ext, cmap='gray', vmin=0, vmax=1)
        sel = walked
        a.plot(rows[sel, 1], rows[sel, 2], color='orange', lw=.6, label='raw walked poses')
        a.plot(xy[:, 0], xy[:, 1], 'r-', lw=1.2, label='centreline')
        for a0, b0 in pruned:
            a.plot(rows[a0:b0 + 1, 1], rows[a0:b0 + 1, 2], 'c-', lw=.8)
        a.plot(*centre, 'g^' if col == 0 else 'mv', ms=10)
        a.plot(*xy[0], 'ko', ms=6); a.plot(*xy[-1], 'ks', ms=6)
        a.set_xlim(centre[0] - 25, centre[0] + 25); a.set_ylim(centre[1] - 25, centre[1] + 25)
        a.set_aspect('equal'); a.set_title(f'{title} (cyan = pruned excursions; o start, s end)')
    a = fig.add_subplot(gs[1, 1:])
    a.plot(s, z_scanner, 'k-', lw=1, label='scanner z on centreline')
    a.set_xlabel('station s (m)'); a.set_ylabel('scanner z (m)'); a.grid(alpha=.3)
    a2 = a.twinx(); a2.plot(s, (t - t[0]) / 60, 'b-', lw=.6); a2.set_ylabel('elapsed time (min)', color='b')
    a.set_title(f'Segment A: length {s[-1]:.1f} m, {len(pruned)} pruned excursions')
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
