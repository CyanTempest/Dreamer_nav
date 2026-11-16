"""Full-route evaluation of planner / dreamer / hybrid on the maps of a manifest.

  python -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method planner \
      --repeats 8 --seed 914901 --output audit_20261116/nav4_dev/c2_planner_s914901.json [--assist] [--plot]

Slot i drives map i % count from the route start (task.route_start) with the reset drawn from --seed only,
so methods with the same seed are paired at reset. One traverse per slot. Without --assist a failure ends
the traverse. With --assist a failure is followed by a restart at the nearest feasible pose at or after the
failure station + assist.advance (env.restart_pose) and the traverse continues; strict completion = no assist,
assisted completion = at most assist.limit (3, competition rule) assists per traverse.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

def overrides(items):
    return {k: json.loads(v) for k, v in (item.split('=', 1) for item in items or [])}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, dt, limit):
    valid = [r for r in rows if r['valid_start']]
    if not valid:
        return dict(traverses=0)
    covered = sum(r['final_station']-r['start_station'] for r in valid)
    per_map = {}
    for r in valid:
        m = per_map.setdefault(r['map'], dict(traverses=0, strict=0, within_limit_assists=0, assists=0))
        m['traverses'] += 1
        m['strict'] += r['success']
        m['within_limit_assists'] += r['arrived'] and r['assists'] <= limit
        m['assists'] += r['assists']
    steps = sum(r['steps'] for r in valid)
    return dict(traverses=len(valid), strict_completions=sum(r['success'] for r in valid),
                strict_completion_rate=sum(r['success'] for r in valid)/len(valid),
                assist_limit=limit, completions_within_limit=sum(r['arrived'] and r['assists'] <= limit for r in valid),
                completions_any_assists=sum(r['arrived'] for r in valid),
                assists=sum(r['assists'] for r in valid), route_m_covered=covered,
                assists_per_100m=100*sum(r['assists'] for r in valid)/max(covered, 1e-9),
                first_failure_codes=dict(sorted(Counter(str(r['failures'][0][0]) if r['failures'] else
                                                        ('truncated' if r['truncated'] else '0') for r in valid).items())),
                all_failure_codes=dict(sorted(Counter(str(f[0]) for r in valid for f in r['failures']).items())),
                mean_final_station=sum(r['final_station'] for r in valid)/len(valid),
                mean_speed=sum(r['distance'] for r in valid)/max(steps*dt, 1e-9),
                takeover_fraction=sum(r['takeover_steps'] for r in valid)/max(sum(r['active_steps'] for r in valid), 1),
                per_map=dict(sorted(per_map.items())))


def plot(path, bank, rows, tracks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cols = min(3, bank.count)
    fig, axes = plt.subplots((bank.count+cols-1)//cols, cols, figsize=(5*cols, 5*((bank.count+cols-1)//cols)),
                             squeeze=False, layout='constrained')
    for m, ax in enumerate(axes.ravel()):
        if m >= bank.count:
            ax.axis('off')
            continue
        route = np.asarray(bank.routes[m, :int(bank.lengths[m])])
        ax.plot(route[:, 0], route[:, 1], 'k-', lw=3, alpha=.25, label='route')
        for r, track in zip(rows, tracks):
            if r['map_index'] != m or not track:
                continue
            xy = np.array(track)
            ax.plot(xy[:, 0], xy[:, 1], '-', lw=.8, color='tab:green' if r['success'] else
                    ('tab:orange' if r['arrived'] else 'tab:red'))
            for code, station, x, y in r['failures']:
                ax.plot(x, y, 'x', color='tab:red', ms=6)
        ax.set_title(bank.names[m])
        ax.set_aspect('equal')
    fig.savefig(path, dpi=120)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True, help='manifest.json or its directory')
    p.add_argument('--maps', nargs='+', help='map names to evaluate (default: every map in the manifest)')
    p.add_argument('--method', choices=['planner', 'dreamer', 'hybrid'], required=True)
    p.add_argument('--checkpoint', help='Dreamer checkpoint (dreamer, hybrid)')
    p.add_argument('--action', choices=['sample', 'mode'], default='sample', help='Dreamer action rule')
    p.add_argument('--repeats', type=int, default=8, help='traverses per map')
    p.add_argument('--seed', type=int, default=914901)
    p.add_argument('--output', required=True, help='result JSON')
    p.add_argument('--assist', action='store_true', help='restart after failures instead of ending the traverse')
    p.add_argument('--max-assists', type=int, help='per traverse (default unlimited within --max-steps)')
    p.add_argument('--max-steps', type=int, help='loop budget (default 2x the per-episode timeout of the longest route)')
    p.add_argument('--env', nargs='*', metavar='KEY=JSON', help='env.json overrides, e.g. sensor.height_std=[0,0]')
    p.add_argument('--planner', nargs='*', metavar='KEY=JSON', help='planner.json overrides')
    p.add_argument('--plot', action='store_true', help='top-view trajectories next to the JSON')
    a = p.parse_args(argv)
    if a.method != 'planner' and not a.checkpoint:
        p.error(f'{a.method} requires --checkpoint')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .bank import TrailBank, manifest_path
    from .env import TrailEnv
    from .planner import Hybrid, Planner

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank = TrailBank(a.manifest, a.maps)
    env = TrailEnv(bank, bank.count*a.repeats, overrides(a.env))
    planner = Planner(env.cfg, overrides(a.planner))
    n, dt, start = env.n, env.cfg['timing']['dt'], env.cfg['task']['route_start']
    state, obs, _ = env.reset(jax.random.PRNGKey(a.seed), start_s=start)
    assert obs['bev'].shape == (n, 64, 64, 6) and obs['bev'].dtype == jnp.uint8 and obs['state'].shape == (n, 13)
    valid = np.asarray(env.feasible(state, state['pos'], state['yaw']))
    leakage, guard, dreamer = None, None, None
    if a.method == 'planner':
        act = planner
    else:
        from .policy import DreamerPolicy
        dreamer = DreamerPolicy(0, str(out.parent/'policy_runtime'), jax_overrides=dict(autotune=1, deterministic=True),
                                deterministic=a.action == 'mode').load(a.checkpoint)
        leakage = dreamer.leakage_check(obs)
        guard = Hybrid(planner) if a.method == 'hybrid' else None

        def act(o):
            action = dreamer(o)
            if guard:
                action = guard(action, o)
            dreamer.executed(action)
            return action
    final = np.asarray(bank.final(state['mids']))
    budget = a.max_steps or 2*int(np.max(np.asarray(state['max_steps'])))
    rows = [dict(slot=i, map=bank.names[int(m)], map_index=int(m), valid_start=bool(valid[i]), start_station=start,
                 final_target=float(final[i]), success=False, arrived=False, truncated=False, assists=0,
                 restarts_without_feasible_pose=0, failures=[], final_station=start, steps=0, distance=0.,
                 active_steps=0, takeover_steps=0) for i, m in enumerate(np.asarray(state['mids']))]
    running = valid.copy()
    tracks = [[] for _ in range(n)]
    begin = time.time()
    for step in range(budget):
        if not running.any():
            break
        action = act(obs)
        if not np.isfinite(np.asarray(action)).all():
            raise ValueError('non-finite action')
        taken = np.asarray(guard.takeover) if guard else np.zeros(n, bool)
        state, obs, _ = env.step(state, action)
        h = jax.device_get({k: state[k] for k in ('done', 'arrived', 'failure', 's', 'steps', 'distance', 'pos')})
        restart = np.full((n,), np.nan, np.float32)
        for i in np.flatnonzero(running):
            r = rows[i]
            r['active_steps'] += 1
            r['takeover_steps'] += int(taken[i])
            tracks[i].append(h['pos'][i].tolist())
            if not h['done'][i]:
                continue
            r['steps'] += int(h['steps'][i])
            r['distance'] += float(h['distance'][i])
            r['final_station'] = float(h['s'][i])
            if h['arrived'][i]:
                r['arrived'], r['success'], running[i] = True, r['assists'] == 0, False
                continue
            r['failures'].append([int(h['failure'][i]), float(h['s'][i]), *map(float, h['pos'][i])])
            if a.assist and (a.max_assists is None or r['assists'] < a.max_assists):
                r['assists'] += 1
                restart[i] = h['s'][i]+env.cfg['assist']['advance']
                tracks[i].append([np.nan, np.nan])
            else:
                running[i] = False
        if np.isfinite(restart).any():
            pose, found = env.restart_pose(state, jnp.asarray(np.nan_to_num(restart)))
            todo = np.isfinite(restart)
            for i in np.flatnonzero(todo & ~np.asarray(found)):
                rows[i]['restarts_without_feasible_pose'] += 1
            state['restart'] = jnp.where(jnp.asarray(todo)[:, None], pose, jnp.nan)
    for i in np.flatnonzero(running):
        rows[i].update(truncated=True, final_station=float(state['s'][i]), steps=rows[i]['steps']+int(state['steps'][i]),
                       distance=rows[i]['distance']+float(state['distance'][i]))
    for r in rows:
        r.update(mean_speed=r['distance']/max(r['steps']*dt, 1e-9), takeover_fraction=r['takeover_steps']/max(r['active_steps'], 1))
    seconds = time.time()-begin
    try:
        peak = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
    except Exception:
        peak = None
    result = dict(
        args=vars(a), method=a.method, action=a.action if dreamer else None, seed=a.seed, episodes=n,
        maps=list(bank.names), manifest_sha256=sha256(manifest_path(a.manifest)),
        checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None, env_config=env.cfg,
        planner_config=planner.cfg, loop_steps=step+1, budget=budget, seconds=seconds, jax_peak_bytes_in_use=peak,
        leakage=leakage, summary=summarize(rows, dt, env.cfg['assist']['limit']), rows=rows)
    out.write_text(json.dumps(result, indent=1)+'\n')
    if a.plot:
        plot(out.with_suffix('.png'), bank, rows, tracks)
    print('RESULT', json.dumps(dict(output=str(out), seconds=round(seconds, 1), **{
        k: v for k, v in result['summary'].items() if k != 'per_map'})), flush=True)


if __name__ == '__main__':
    main()
