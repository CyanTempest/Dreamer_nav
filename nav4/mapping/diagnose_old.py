"""Task 2a: why the old r9 proxy rejects segment A, per route station.

Reads data/trail_a/old_proxy/{map.npz, old_proxy_masks.npz} (faithful re-run of the
old criteria on the segment-A grid) and the segment-A centreline.  For each station
(robot footprint at the centreline pose) it records which old criterion is violated
inside the 43x25 footprint: unknown ground, slope>=0.8 (0.4 m baseline), body-band
obstacle (>=2 points 0.22-0.8 m), plus single-criterion counterfactuals and the
lateral clear width.  Also validates the re-implementation against the r9 survey
window maps (same XY, their own routes).

Writes data/trail_a/diagnosis/old_proxy_stations.npz, old_proxy_diagnosis.json and
figures/old_proxy_causes.png.
"""
import json
import time
from pathlib import Path

import numpy as np

from nav4.mapping.analysis import (ROBOT_W, blocking_runs, feasible_width, footprint_cells, lateral_feasibility,
                                   lookup)
from nav4.mapping.common import DATA, ROOT, bilinear, footprint_ok, footprint_world, signed_distance, write_json
from nav4.mapping.grid import load_centreline

OLD_SURVEY = ROOT / 'r9_official_runner' / 'maps_scan_survey'


def station_causes(masks, origin, xy, yaw):
    known, support, obstacle = masks['known'], masks['support'], masks['obstacle']
    shape = known.shape
    ij, inside = footprint_cells(origin, shape, xy, yaw)
    kn = known[ij[..., 0], ij[..., 1]] & inside
    su = support[ij[..., 0], ij[..., 1]] & inside
    ob = (bilinear(signed_distance(obstacle), origin, footprint_world(xy, yaw)) < 0) & inside
    unknown = ~kn
    steep = kn & ~su
    clear = su & ~ob
    return dict(
        feasible=clear.all(1),
        frac_unknown=unknown.mean(1), frac_steep=steep.mean(1), frac_obstacle=ob.mean(1),
        any_unknown=unknown.any(1), any_steep=steep.any(1), any_obstacle=ob.any(1),
        cf_no_unknown=((su | ~kn) & ~ob).all(1),
        cf_no_slope=(kn & ~ob).all(1),
        cf_no_obstacle=su.all(1),
        cf_no_slope_no_unknown=(~ob).all(1),
    )


def clear_lane(masks, origin, xy, yaw, half=3.0, step=.05, band=.75):
    """Contiguous lateral run of clear cells (support & not obstacle) through the
    station, measured on the lateral cut line; the run must overlap |o|<=band."""
    offs = np.arange(-half, half + 1e-9, step)
    normal = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    pts = xy[:, None] + normal[:, None] * offs[None, :, None]
    c = lookup(masks['support'], origin, pts, False) & (bilinear(signed_distance(masks['obstacle']), origin, pts) >= 0)
    out = np.zeros(len(xy))
    near = np.abs(offs) <= band
    for i, row in enumerate(c):
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if near[a:b].any():
                out[i] = max(out[i], (b - a) * step)
    return out


def validate_against_r9(old_map, origin):
    """Footprint feasibility along each r9 survey window route: r9 map vs this re-run."""
    rows = []
    if not OLD_SURVEY.exists():
        return rows
    sup = old_map['layers'][..., 2]
    obs = old_map['layers'][..., 5]
    for d in sorted(OLD_SURVEY.glob('segment_*')):
        with np.load(d / 'map.npz') as f:
            m = {k: f[k] for k in ('layers', 'origin', 'route', 'yaw')}
        xy = m['route'][:, :2].astype(np.float64)
        yaw = m['yaw'].astype(np.float64)
        theirs = footprint_ok(m['layers'][..., 2].astype(np.float32), m['layers'][..., 5].astype(np.float32),
                              m['origin'].astype(np.float64), xy, yaw)
        ours = footprint_ok(sup, obs, origin, xy, yaw)
        # cell agreement of the masks over the r9 window
        lay = m['layers'].astype(np.float32)
        gi = np.stack(np.meshgrid(np.arange(lay.shape[0]), np.arange(lay.shape[1]), indexing='ij'), -1)
        world = m['origin'] + gi * .1
        o_sup = lookup(sup, origin, world, 100.) <= .025
        o_obs = lookup(obs, origin, world, 100.) < 0
        t_sup = lay[..., 2] <= .025
        t_obs = lay[..., 5] < 0
        near = lay[..., 3] <= 2.   # within 3 m of the r9 window route (their layer 3 = dist-1)
        rows.append(dict(window=d.name, stations=len(xy), r9_feasible=float(theirs.mean()),
                         rerun_feasible=float(ours.mean()), station_agreement=float((theirs == ours).mean()),
                         support_cell_agreement_within_3m=float((o_sup == t_sup)[near].mean()),
                         obstacle_cell_agreement_within_3m=float((o_obs == t_obs)[near].mean())))
    return rows


def main():
    begin = time.time()
    line = load_centreline()
    s, xy, yaw = line['s'], line['xy'], line['yaw']
    with np.load(DATA / 'old_proxy' / 'map.npz') as f:
        old_map = {k: f[k] for k in ('layers', 'origin')}
    origin = old_map['origin'].astype(np.float64)
    with np.load(DATA / 'old_proxy' / 'old_proxy_masks.npz') as f:
        masks = {k: f[k] for k in f.files}
    c = station_causes(masks, origin, xy, yaw)
    offsets = np.arange(-1.5, 1.5001, .05)
    feas = lateral_feasibility(old_map['layers'][..., 2], old_map['layers'][..., 5], origin, xy, yaw, offsets)
    width = feasible_width(feas, offsets)
    lane = clear_lane(masks, origin, xy, yaw)
    shift_ok = feas[:, np.abs(offsets) <= .75].any(1)
    fail = ~c['feasible']
    n = len(s)
    # Exclusive category per station
    cat = np.full(n, 'pass', object)
    only_u = fail & c['any_unknown'] & ~c['any_steep'] & ~c['any_obstacle']
    only_s = fail & ~c['any_unknown'] & c['any_steep'] & ~c['any_obstacle']
    only_o = fail & ~c['any_unknown'] & ~c['any_steep'] & c['any_obstacle']
    cat[only_u] = 'unknown_only'
    cat[only_s] = 'slope_only'
    cat[only_o] = 'obstacle_only'
    cat[fail & ~(only_u | only_s | only_o)] = 'combined'
    narrow = fail & (lane > 0) & (lane < ROBOT_W)
    blocked_across = fail & (lane == 0)
    summary = dict(
        stations=n, length_m=float(s[-1]),
        feasible_fraction_at_centreline=float(c['feasible'].mean()),
        feasible_fraction_any_offset_within_0p75=float(shift_ok.mean()),
        failing_stations=int(fail.sum()),
        any_criterion_present_in_failing_footprint=dict(
            unknown=int((fail & c['any_unknown']).sum()), slope=int((fail & c['any_steep']).sum()),
            obstacle=int((fail & c['any_obstacle']).sum())),
        exclusive=dict(unknown_only=int(only_u.sum()), slope_only=int(only_s.sum()),
                       obstacle_only=int(only_o.sum()), combined=int((cat == 'combined').sum())),
        counterfactual_feasible_fraction=dict(
            as_is=float(c['feasible'].mean()), drop_unknown=float(c['cf_no_unknown'].mean()),
            drop_slope=float(c['cf_no_slope'].mean()), drop_obstacle=float(c['cf_no_obstacle'].mean()),
            drop_slope_and_unknown=float(c['cf_no_slope_no_unknown'].mean())),
        mean_footprint_fraction_in_failing=dict(
            unknown=float(c['frac_unknown'][fail].mean()), slope=float(c['frac_steep'][fail].mean()),
            obstacle=float(c['frac_obstacle'][fail].mean())),
        narrow_lane_stations=int(narrow.sum()), blocked_across_stations=int(blocked_across.sum()),
        clear_lane_width_m=dict(min=float(lane.min()), p5=float(np.percentile(lane, 5)),
                                median=float(np.median(lane))),
        feasible_width_m=dict(min=float(width.min()), p5=float(np.percentile(width, 5)),
                              median=float(np.median(width))),
        longest_feasible_run_m=float(max([r['length_m'] for r in blocking_runs(fail, s, xy)] or [0])),
    )
    # 25 m bins along the route
    bins = []
    edges = np.arange(0, s[-1] + 25, 25.)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s >= a) & (s < b)
        if not m.any():
            continue
        bins.append(dict(s_from=float(a), s_to=float(b), stations=int(m.sum()),
                         feasible=float(c['feasible'][m].mean()),
                         unknown_present=float(c['any_unknown'][m].mean()),
                         slope_present=float(c['any_steep'][m].mean()),
                         obstacle_present=float(c['any_obstacle'][m].mean()),
                         drop_obstacle_feasible=float(c['cf_no_obstacle'][m].mean()),
                         drop_slope_feasible=float(c['cf_no_slope'][m].mean()),
                         drop_unknown_feasible=float(c['cf_no_unknown'][m].mean())))
    summary['per_25m'] = bins
    validation = validate_against_r9(old_map, origin)
    summary['validation_vs_r9_windows'] = validation
    out = DATA / 'diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'old_proxy_stations.npz', s=s, category=cat.astype(str), width=width, lane=lane,
                        feas=feas, offsets=offsets, **{k: v for k, v in c.items()})
    summary['seconds'] = time.time() - begin
    write_json(out / 'old_proxy_diagnosis.json', summary)
    plot(s, c, cat, width, lane, DATA / 'figures' / 'old_proxy_causes.png')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_25m', 'validation_vs_r9_windows')}, indent=1))
    for v in validation:
        print(v)


def plot(s, c, cat, width, lane, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True, layout='constrained')
    colors = dict(pass_='#2a9d8f', unknown_only='#8d99ae', slope_only='#e9c46a', obstacle_only='#e76f51', combined='#6d597a')
    for k, col in colors.items():
        m = cat == k.rstrip('_')
        ax[0].bar(s[m], np.ones(m.sum()), width=.2, color=col, label=k.rstrip('_'), linewidth=0)
    ax[0].set_yticks([]); ax[0].legend(ncol=5, fontsize=8, loc='upper right'); ax[0].set_title('Old proxy: footprint at centreline, exclusive cause')
    for name, key, col in [('unknown', 'frac_unknown', '#8d99ae'), ('slope>=0.8', 'frac_steep', '#e9c46a'), ('obstacle', 'frac_obstacle', '#e76f51')]:
        ax[1].plot(s, c[key], color=col, lw=.8, label=name)
    ax[1].set_ylabel('fraction of footprint samples'); ax[1].legend(fontsize=8)
    ax[2].plot(s, lane, 'k-', lw=.7, label='clear lane width on cut (m)')
    ax[2].plot(s, width, 'b-', lw=.7, label='feasible width (offsets+0.58 m)')
    ax[2].axhline(.58, color='r', ls='--', lw=.8); ax[2].set_ylim(0, 6.5); ax[2].legend(fontsize=8)
    ax[2].set_xlabel('station s (m)')
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
