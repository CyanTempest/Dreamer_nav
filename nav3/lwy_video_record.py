"""Record one Dreamer episode on an LWY map for kinematic MuJoCo rendering.

Runs on CPU by default so it can share the machine with a GPU training job.
Rendering is kinematic only: this is not an A2 locomotion validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time


def record(checkpoint, split, seed, out, start_s=2., root='maps_lwy_v2', every=3):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_env_v2 import LwyEnvV2
    from .dreamer_adapter import DreamerPolicy
    from .evaluate import FAILURES
    bank = LwyBank(split, root=root)
    tracker = json.loads(Path('configs/lwy_v2_tracker.json').read_text())
    env = LwyEnvV2(bank, 1, random_obstacles=False, config=tracker)
    with open(checkpoint, 'rb') as f: data = pickle.load(f)
    adapter = DreamerPolicy(0, 'logs/lwy_video_agent', jax_overrides=dict(deterministic=True))
    adapter.agent.load(data['agent']); del data
    params = adapter.agent.params
    carry = jax.tree.map(lambda x: x[None], adapter.initial(1))
    infer = jax.jit(jax.vmap(lambda p, c, o, k: adapter._policy(p, c, o, mode='train', seed=k)[1], in_axes=(None, 0, 0, 0)))
    reset = jax.jit(jax.vmap(lambda e, k, m, s: e._reset(k, m[None], s[None]), in_axes=(None, 0, 0, 0)))
    step = jax.jit(jax.vmap(lambda e, s, a: e._step(s, a[None]), in_axes=(None, 0, 0)))
    key = jax.random.PRNGKey(seed+1000000000)[None]
    state, obs, _ = reset(env, jax.random.PRNGKey(seed)[None], jnp.zeros(1, jnp.int32), jnp.full(1, start_s))
    frames = []; begin = time.monotonic()
    for t in range(env.cfg['max_steps']+2):
        pair = jax.vmap(jax.random.split)(key); key, sub = pair[:, 0], pair[:, 1]
        public = {k: obs[k] for k in adapter.obs_space}
        carry, act, _ = infer(params, carry, public, sub)
        action = jnp.where(obs['is_last'][..., None], 0., act['action'])
        carry = (*carry[:3], {'action': action})
        action = action[:, 0]
        if t % every == 0:
            frames.append(dict(pos=np.asarray(state['pos'][0, 0]), yaw=float(state['yaw'][0, 0]),
                s=float(state['s'][0, 0]), gates=int(state['gates'][0, 0]), time=t*env.cfg['dt'],
                action=np.asarray(action[0]), bev=np.asarray(obs['bev'][0, 0]), terminal=False))
        state, obs, _ = step(env, state, action)
        if bool(obs['is_last'][0, 0]):
            frames.append(dict(pos=np.asarray(state['pos'][0, 0]), yaw=float(state['yaw'][0, 0]),
                s=float(state['s'][0, 0]), gates=int(state['gates'][0, 0]), time=(t+1)*env.cfg['dt'],
                action=np.zeros(3, np.float32), bev=np.asarray(obs['bev'][0, 0]), terminal=True))
            break
        if t % 500 == 0:
            print(json.dumps(dict(step=t, s=frames[-1]['s'], gates=frames[-1]['gates'], seconds=round(time.monotonic()-begin, 1))), flush=True)
    assert frames[-1]['terminal'], 'Incomplete rollout'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # Names such as dreamer_3.00h contain dots; never use Path.with_suffix.
    np.savez_compressed(f'{out}.npz', **{k: np.asarray([r[k] for r in frames]) for k in frames[0]})
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        map=bank.names[0], split=split, seed=seed, start_s_m=start_s, random_obstacles=False, safety_filter=False,
        gates=int(state['gates'][0, 0]), failure=FAILURES[int(state['failure'][0, 0])],
        progress_m=float(state['s'][0, 0]), duration_s=float(frames[-1]['time']), frames=len(frames),
        frame_interval_s=every*env.cfg['dt'], device=str(jax.devices()[0]),
        scope='Fast-env inference; MuJoCo rendering only, no leg physics')
    Path(f'{out}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--split', default='test')
    p.add_argument('--seed', type=int, default=940000); p.add_argument('--start', type=float, default=2.)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    record(a.checkpoint, a.split, a.seed, a.out, a.start)
