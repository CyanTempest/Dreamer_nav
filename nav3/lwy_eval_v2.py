"""Deterministic LWY qualification; G1 failure strictly forbids learning jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def setup():
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def wilson(k, n):
    z = 1.959963984540054; p = k/n; den = 1+z*z/n
    middle = (p+z*z/(2*n))/den
    half = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0, middle-half), min(1, middle+half)]


def protocol_hash(root='maps_lwy_v2'):
    from .provenance import fingerprint
    h = hashlib.sha256(fingerprint().encode())
    for path in ['nav3/lwy_env.py', 'nav3/lwy_env_v2.py', 'nav3/lwy_bank.py', 'nav3/lwy_import.py', 'nav3/lwy_control.py', 'nav3/lwy_eval_v2.py', 'configs/lwy_v2_tracker.json',
                 'nav3/lwy_geometry.py', 'nav3/lwy_control_v2.py', 'configs/lwy_v2_planner.json']:
        h.update(path.encode()); h.update(Path(path).read_bytes())
    records=json.loads((Path(root)/'manifest.json').read_text())['maps']
    qualified=sorted([{k:r[k] for k in ['name','seed','sha256','split','dataset_version']}
                      for r in records if r['split'] in ['test','blind']],key=lambda r:r['name'])
    assert len(qualified)==21
    h.update(json.dumps(qualified,sort_keys=True).encode())
    return h.hexdigest()


def run(split, episodes, tag, planner_name='v2', start_s=2., limit=None, root='maps_lwy_v2', seed_start=940000, tracker_gain=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_env_v2 import LwyEnvV2 as LwyEnv
    from .control import planner
    if planner_name == 'v2':
        from functools import partial
        from .lwy_control_v2 import planning_result
        parameters = json.loads(Path('configs/lwy_v2_planner.json').read_text())
        planner = lambda obs: planning_result(obs, **parameters)[0]
    elif planner_name == 'lwy':
        from .lwy_control import planner
    else:
        assert planner_name == 'legacy'
    path = Path(f'metrics/lwy_{tag}.json'); assert not path.exists(), path
    bank = LwyBank(split, limit=limit, root=root)
    tracker=json.loads(Path('configs/lwy_v2_tracker.json').read_text())
    if tracker_gain is not None:tracker['yaw_tracking_gain']=tracker_gain
    env = LwyEnv(bank, 1, random_obstacles=False,config=tracker)
    n = bank.count*episodes
    mids = np.repeat(np.arange(bank.count), episodes)
    seeds = np.arange(seed_start, seed_start+n)
    # One independent stream per episode; batching cannot silently change noise.
    # The atlas must be a dynamic argument. Capturing a 20-map atlas in a
    # closure embeds gigabytes of constants and exceeds XLA's proto limit.
    reset = jax.jit(jax.vmap(lambda e, k, m, s: e._reset(k, m[None], s[None]), in_axes=(None, 0, 0, 0)))
    step = jax.jit(jax.vmap(lambda e, s, a: e._step(s, a[None]), in_axes=(None, 0, 0)))
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    state, obs, extra = reset(env, keys, jnp.asarray(mids), jnp.full(n, start_s))
    done = np.zeros(n, bool); rows = [None]*n; traces = [[] for _ in range(n)]
    endpoints = [None]*n
    traces_hash = [hashlib.sha256() for _ in range(n)]
    begin = time.monotonic()
    for t in range(env.cfg['max_steps']+2):
        flat = {k: v[:, 0] for k, v in obs.items()}
        action = planner(flat)
        state, obs, extra = step(env, state, action)
        terminal = np.asarray(obs['is_last'][:, 0]) & ~done
        pose = np.concatenate([np.asarray(state['pos'][:, 0]), np.asarray(state['yaw'][:, 0, None]), np.asarray(action)], -1)
        for i in np.flatnonzero(~done):
            traces_hash[i].update(pose[i].tobytes())
            if t % 25 == 0: traces[i].append([float(state['s'][i, 0]), *pose[i].tolist()])
        if terminal.any():
            values = {k: np.asarray(state[k][:, 0]) for k in ['gates', 'failure', 'steps', 's', 'distance', 'ep_return']}
            from .evaluate import FAILURES
            for i in np.flatnonzero(terminal):
                failure = int(values['failure'][i]); gates = int(values['gates'][i])
                endpoints[i] = dict(map=bank.names[mids[i]], noise_seed=int(seeds[i]),
                    xy=pose[i, :2].tolist(), yaw=float(pose[i, 2]), s=float(values['s'][i]), failure=FAILURES[failure])
                rows[i] = dict(map=bank.names[mids[i]], map_seed=bank.records[mids[i]]['seed'],
                    noise_seed=int(seeds[i]), gates=gates, success=gates == 8 and failure == 0,
                    failure=FAILURES[failure], progress_m=float(values['s'][i]),
                    steps=int(values['steps'][i]), mean_speed_mps=float(values['distance'][i])/max(.2, values['steps'][i]*.2),
                    return_=float(values['ep_return'][i]), takeovers=0, trajectory=traces[i],
                    action_pose_sha256=traces_hash[i].hexdigest())
            done |= terminal
        if t % 500 == 0:
            print(json.dumps(dict(split=split, planner=planner_name, step=t, finished=int(done.sum()),
                progress_min=float(np.min(np.asarray(state['s'][:, 0]))), seconds=time.monotonic()-begin)), flush=True)
        if done.all(): break
    assert done.all(), 'No censored episodes may be discarded'
    successes = sum(r['success'] for r in rows)
    result = dict(method=f'planner_{planner_name}', split=split, episodes=n, map_root=root,
        success_rate=successes/n, success_wilson95=wilson(successes, n), mean_gates=float(np.mean([r['gates'] for r in rows])),
        failures={f: sum(r['failure'] == f for r in rows) for f in set(r['failure'] for r in rows)},
        random_obstacles=False, safety_filter=False, start_s_m=start_s, rows=rows, endpoints=endpoints,
        wall_seconds=time.monotonic()-begin, bank_bytes=bank.host_bytes, planner_parameters=parameters if planner_name=='v2' else {},tracker_parameters=tracker,
        runtime=dict(xla_flags=os.environ['XLA_FLAGS'], jax=jax.__version__, device=str(jax.devices()[0]),
                     seed_protocol='one PRNG stream per episode, vmap(single environment)', atlas_arguments='dynamic'),
        environment_config=env.cfg)
    with path.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    return result


def require_g1():
    gate = json.loads(Path('metrics/lwy_v2_g1.json').read_text())
    assert gate['passed'], 'LWY G1 failed: no RL smoke or training is permitted'
    assert gate['environment_sha256'] == protocol_hash(), 'LWY maps or environment changed since G1'
    records=json.loads(Path('maps_lwy_v2/manifest.json').read_text())['maps']
    assert sum(r['split']=='train' for r in records)>=200 and sum(r['split']=='val' for r in records)==20, 'Full training/validation dataset is required'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--split', default='test')
    p.add_argument('--episodes', type=int, default=10); p.add_argument('--tag', required=True)
    p.add_argument('--planner', choices=['legacy', 'lwy', 'v2'], default='v2')
    p.add_argument('--start', type=float, default=2.); p.add_argument('--limit', type=int)
    p.add_argument('--root', default='maps_lwy_v2')
    p.add_argument('--assert-min-progress', type=float)
    p.add_argument('--seed-start', type=int, default=940000)
    p.add_argument('--tracker-gain',type=float)
    a = p.parse_args(); setup(); result = run(a.split, a.episodes, a.tag, a.planner, a.start, a.limit, a.root, a.seed_start,a.tracker_gain)
    if a.assert_min_progress is not None:
        early = [(r['noise_seed'], r['progress_m'], r['failure']) for r in result['rows'] if r['progress_m'] < a.assert_min_progress]
        assert not early, f'Premature failures before {a.assert_min_progress} m: {early}'


if __name__ == '__main__': main()
