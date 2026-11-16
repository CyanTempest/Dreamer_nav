"""Audit 2026-11-16: honoured tracker yaw gain and deterministic Dreamer action.

Evaluation logic mirrors r9_official_runner/diagnose_policy.py (same asserts,
speed cap, valid-start exclusion, per-episode rows) without the safety filter,
the .npz dumps and the PNG plot. Existing modules are imported, not copied.

Additions
  --yaw-gain {2,4}      Env = GainScanEnv, a ScanEnv subclass whose _step is
                        LwyEnvV2._step (reads cfg['yaw_tracking_gain']); the
                        gain is written into cfg explicitly.
  --action {sample,mode}
                        sample: stock DreamerPolicy (Agent.policy -> sample()).
                        mode:   mirror of Agent.policy (enc -> dyn.observe -> pol)
                                returning pol(...).pred() == tanh(mean) of the
                                bounded_normal head instead of a sample. The RSSM
                                posterior latent is still sampled (nj.seed()).
  --sanity              Bit-identity check: stock ScanEnv vs GainScanEnv(2) vs
                        GainScanEnv(4) under a fixed action, plus (optional, with
                        --checkpoint) mirrored-sample == stock DreamerPolicy.
"""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import time


def make_env_class():
    import jax
    from scan_env import ScanEnv
    from nav3.lwy_env_v2 import LwyEnvV2

    @jax.tree_util.register_pytree_node_class
    class GainScanEnv(ScanEnv):
        """ScanEnv whose step honours cfg['yaw_tracking_gain'] via LwyEnvV2._step."""
        def __init__(self, bank, n, body_bands=False, yaw_gain=2.):
            super().__init__(bank, n, body_bands)
            self.cfg['yaw_tracking_gain'] = float(yaw_gain)
            self.reset = partial(jax.jit(GainScanEnv._reset), self)
            self.step = partial(jax.jit(GainScanEnv._step), self)
            self.observe = partial(jax.jit(GainScanEnv._observe), self)

        def _step(self, state, action):
            return LwyEnvV2._step(self, state, action)
    return GainScanEnv


def make_mirror(dreamer, deterministic=True):
    """nj.pure mirror of dreamerv3 Agent.policy(); deterministic -> .pred()."""
    import jax
    import ninjax as nj
    from dreamerv3.agent import sample
    model = dreamer.agent.model

    def act_fn(carry, obs):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        kw = dict(training=False, single=True)
        reset = obs['is_first']
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset, **kw)
        dyn_carry, _, feat = model.dyn.observe(dyn_carry, tokens, prevact, reset, **kw)
        # Decoder carry is {} for rssm.Decoder and is not needed for acting; passed through.
        dist = model.pol(model.feat2tensor(feat), bdims=1)
        act = jax.tree.map(lambda d: d.pred(), dist) if deterministic else sample(dist)
        stats = dict(mean=jax.tree.map(lambda d: d.output.mean, dist),
                     stddev=jax.tree.map(lambda d: d.output.stddev, dist))
        return (enc_carry, dyn_carry, dec_carry, act), act, stats
    return jax.jit(nj.pure(act_fn))


class MirrorPolicy:
    """Same carry/key protocol as DreamerPolicy.policy(), mirrored action head."""
    def __init__(self, dreamer, deterministic=True):
        self.dreamer = dreamer
        self.fn = make_mirror(dreamer, deterministic)
        self.last_stats = None

    def __call__(self, obs):
        import jax
        import jax.numpy as jnp
        d = self.dreamer
        obs = {k: obs[k] for k in d.obs_space}
        n = len(obs['is_first'])
        if d.carry is None or d.amnesic:
            d.carry = d.initial(n)
        d.key, key = jax.random.split(d.key)
        _, (carry, act, stats) = self.fn(d.agent.params, d.carry, obs, seed=key)
        action = jnp.where(obs['is_last'][:, None], 0., act['action'])
        d.carry = (*carry[:3], {'action': action})
        self.last_stats = stats
        return action


def bits_equal(x, y):
    import numpy as np
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def sanity(a):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scan_env import ScanBank, ScanEnv
    Gain = make_env_class()
    bank = ScanBank(a.maps)
    n = bank.count * a.repeats
    envs = dict(stock=ScanEnv(bank, n, a.body_bands), g2=Gain(bank, n, a.body_bands, 2.),
                g4=Gain(bank, n, a.body_bands, 4.))
    report = dict(steps=a.sanity_steps, n=n, seed=a.seed,
                  cfg_gain={k: e.cfg.get('yaw_tracking_gain') for k, e in envs.items()})
    # Fixed (time-constant) action; per-slot heading offsets so local_y != 0.
    action = jnp.stack([jnp.zeros(n), jnp.linspace(-.6, .6, n), jnp.zeros(n)], -1).astype(jnp.float32)
    report['action'] = np.asarray(action).tolist()
    traj = {}
    for name, env in envs.items():
        state, obs, _ = env.reset(jax.random.PRNGKey(a.seed))
        seq = [(jax.device_get(state), jax.device_get(obs))]
        for _ in range(a.sanity_steps):
            state, obs, _ = env.step(state, action)
            seq.append((jax.device_get(state), jax.device_get(obs)))
        traj[name] = seq

    def compare(p, q):
        mismatched = []
        for t, ((sa, oa), (sb, ob)) in enumerate(zip(traj[p], traj[q])):
            for group, da, db in (('state', sa, sb), ('obs', oa, ob)):
                for k in sorted(da):
                    if not bits_equal(da[k], db[k]):
                        mismatched.append(f'{group}.{k}@{t}')
        return mismatched
    m2 = compare('stock', 'g2')
    m4 = compare('stock', 'g4')
    last_s, last_4 = traj['stock'][-1][0], traj['g4'][-1][0]
    first4 = next((t for t, (s, _) in enumerate(traj['g4'])
                   if not bits_equal(s['yaw'], traj['stock'][t][0]['yaw'])), None)
    report.update(
        stock_vs_gain2_bit_identical=not m2, stock_vs_gain2_mismatches=m2[:50],
        stock_vs_gain4_bit_identical=not m4, stock_vs_gain4_mismatch_count=len(m4),
        stock_vs_gain4_first_yaw_diff_step=first4,
        gain4_final_max_abs_yaw_diff=float(np.max(np.abs(np.angle(np.exp(1j*(last_4['yaw']-last_s['yaw'])))))),
        gain4_final_max_pos_diff_m=float(np.max(np.linalg.norm(last_4['pos']-last_s['pos'], axis=-1))),
        gain4_final_slots_with_pos_diff=int(np.sum(np.any(last_4['pos'] != last_s['pos'], -1))),
        failures_in_window={k: int(sum(int(np.sum(s['failure'] > 0)) for s, _ in v)) for k, v in traj.items()})
    print('ENV_SANITY', json.dumps({k: v for k, v in report.items() if k != 'action'}), flush=True)

    if a.checkpoint:
        # Mirrored Agent.policy with sample() must equal the stock DreamerPolicy bit-for-bit.
        from nav3.dreamer_adapter import DreamerPolicy
        env = envs['stock']
        state, obs, _ = env.reset(jax.random.PRNGKey(a.seed))
        d = DreamerPolicy(0, str(Path(a.output).parent/'policy_runtime'), jax_overrides=dict(autotune=1, deterministic=True))
        with Path(a.checkpoint).open('rb') as stream: d.agent.load(pickle.load(stream)['agent'])
        mirror_sample = MirrorPolicy(d, deterministic=False)
        mirror_mode = MirrorPolicy(d, deterministic=True)
        equal, gaps, stds, outside = [], [], [], []
        for _ in range(a.sanity_steps):
            # Same carry and key before each of the three calls; env driven by the stock action.
            c0, k0 = d.carry, d.key
            act_ref = d(obs)
            c_ref, k_ref = d.carry, d.key
            d.carry, d.key = c0, k0
            act_mir = mirror_sample(obs)
            c_mir, k_mir = d.carry, d.key
            same = bits_equal(act_ref, act_mir) and bits_equal(k_ref, k_mir) and all(
                bits_equal(x, y) for x, y in zip(jax.tree.leaves(c_ref), jax.tree.leaves(c_mir)))
            equal.append(bool(same))
            d.carry, d.key = c0, k0
            act_mode = mirror_mode(obs)
            st = mirror_mode.last_stats
            gaps.append(float(jnp.mean(jnp.abs(act_ref - act_mode))))
            stds.append(np.asarray(st['stddev']['action']).mean(0).tolist())
            outside.append(float(jnp.mean(jnp.abs(act_ref) > 1)))
            d.carry, d.key = c_ref, k_ref
            state, obs, _ = env.step(state, act_ref)
        report['dreamer'] = dict(
            mirror_sample_equals_stock_each_step=equal,
            mirror_sample_equals_stock_all=all(equal),
            mean_abs_sample_minus_mode=gaps,
            policy_stddev_per_dim_mean=np.mean(stds, 0).tolist(),
            fraction_sampled_components_outside_unit_box=float(np.mean(outside)))
        print('POLICY_SANITY', json.dumps(report['dreamer']), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')


def summarize(rows):
    valid = [r for r in rows.values() if r['valid_start']]
    ok = [r for r in valid if r.get('gates') == 8 and r.get('failure') == 0]
    codes = Counter('truncated' if r.get('truncated') else str(int(r['failure'])) for r in valid)
    per_map = {}
    for r in valid:
        m = per_map.setdefault(r['map'], dict(valid=0, successes=0))
        m['valid'] += 1
        m['successes'] += int(r.get('gates') == 8 and r.get('failure') == 0)
    return dict(valid_starts=len(valid), successes=len(ok),
                success_rate=len(ok)/len(valid) if valid else None,
                failure_code_counts=dict(sorted(codes.items())),
                mean_s_at_termination=sum(r['s'] for r in valid)/len(valid) if valid else None,
                per_map=dict(sorted(per_map.items())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maps', required=True); p.add_argument('--checkpoint')
    p.add_argument('--output', required=True); p.add_argument('--body-bands', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--method', choices=['dreamer', 'planner'], default='dreamer')
    p.add_argument('--speed-cap', type=float, default=.8)
    p.add_argument('--seed', type=int, default=914233)
    p.add_argument('--yaw-gain', type=float, choices=[2., 4.], required=True)
    p.add_argument('--action', choices=['sample', 'mode'], default='sample')
    p.add_argument('--sanity', action='store_true')
    p.add_argument('--sanity-steps', type=int, default=20)
    a = p.parse_args()
    from nav3.lwy_training_runtime import configure
    runtime = configure('dreamer')
    if a.sanity:
        return sanity(a)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scan_env import ScanBank
    from nav3.env import decode, rotate
    Gain = make_env_class()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    bank = ScanBank(a.maps)
    env = Gain(bank, bank.count*a.repeats, a.body_bands, a.yaw_gain)
    assert env.cfg['yaw_tracking_gain'] == a.yaw_gain
    state, obs, extra = env.reset(jax.random.PRNGKey(a.seed))
    assert obs['bev'].shape == (env.n, 64, 64, 6) and obs['state'].shape == (env.n, 13)
    assert obs['bev'].dtype == jnp.uint8 and np.isfinite(np.asarray(obs['state'])).all()
    outside = env.bank.sample(jnp.arange(env.bank.count), jnp.full((env.bank.count, 2), 1e6))
    assert np.all(np.asarray(outside[..., 2]) > 0)
    fp = state['pos'][:, None]+rotate(env.foot[None], state['yaw'])
    physical, _ = env.truth(state, fp)
    initial_valid = np.asarray(jnp.all((physical[..., 2] <= .025) & (physical[..., 5] >= 0), -1))
    initial_bev = np.asarray(decode(obs['bev']))
    dreamer = None
    if a.method == 'dreamer':
        if not a.checkpoint: p.error('Dreamer requires --checkpoint')
        from nav3.dreamer_adapter import DreamerPolicy
        dreamer = DreamerPolicy(0, str(out.parent/'policy_runtime'), jax_overrides=dict(autotune=1, deterministic=True))
        with Path(a.checkpoint).open('rb') as stream: dreamer.agent.load(pickle.load(stream)['agent'])
        leakage = dreamer.leakage_check(obs)
        policy = dreamer if a.action == 'sample' else MirrorPolicy(dreamer, deterministic=True)
    else:
        if a.action != 'sample': p.error('--action mode is defined for Dreamer only')
        from nav3.lwy_control_v2 import planning_result
        parameters = json.loads(Path('configs/lwy_v2_planner.json').read_text())
        policy = jax.jit(lambda o: planning_result(o, **parameters)[0]); leakage = {'not_applicable': 'nonlearned observation-only planner'}
    rows = {i: dict(valid_start=False, diagnostic='invalid_geometry_start; excluded from policy outcomes') for i in range(env.n) if not initial_valid[i]}
    begin = time.time(); active_steps = np.zeros(env.n, int)
    std_sum = np.zeros(3); std_count = 0
    for step in range(a.steps):
        action = policy(obs)
        action = action.at[:, 2].set(jnp.minimum(action[:, 2], a.speed_cap/.4-1))
        active = np.array([i not in rows for i in range(env.n)])
        action = action.at[:, 2].set(jnp.minimum(action[:, 2], a.speed_cap/.4-1))
        active_steps += active
        if a.method == 'dreamer':
            # RSSM action history must contain the action actually executed.
            dreamer.carry = (*dreamer.carry[:3], {'action': action})
            if a.action == 'mode' and active.any():
                std = np.asarray(policy.last_stats['stddev']['action'])
                std_sum += std[active].sum(0); std_count += int(active.sum())
        if not np.isfinite(np.asarray(action)).all(): raise ValueError('Nonfinite policy action')
        state, obs, _ = env.step(state, action)
        done = np.asarray(obs['is_last'])
        if done.any():
            host = jax.device_get({k: state[k] for k in ['gates', 'failure', 's', 'ep_return', 'steps']})
            for i in np.flatnonzero(done):
                if int(i) not in rows: rows[int(i)] = {**{k: float(v[i]) for k, v in host.items()}, 'valid_start': True}
        if len(rows) == env.n: break
    loop_steps = step+1
    for i in range(env.n):
        if i not in rows: rows[i] = dict(valid_start=True, truncated=True, s=float(state['s'][i]))
    for i, row in rows.items():
        row['map'] = env.bank.names[i % env.bank.count]
        row['active_steps'] = int(active_steps[i])
    seconds = time.time()-begin
    try:
        stats = jax.devices()[0].memory_stats() or {}
        peak = stats.get('peak_bytes_in_use')
    except Exception:
        peak = None
    result = dict(
        scope='Audit 2026-11-16 A/B on official scan holdout patches; mirrors diagnose_policy.py without safety filter',
        args=vars(a), method=a.method, action=a.action if a.method == 'dreamer' else None,
        yaw_gain=a.yaw_gain, env_class='GainScanEnv(ScanEnv) with _step=LwyEnvV2._step',
        env_cfg_yaw_tracking_gain=env.cfg['yaw_tracking_gain'],
        action_rule=('stock DreamerPolicy: sample(Agg(Normal(tanh(mean),stddev)))' if a.action == 'sample'
                     else 'mirror of Agent.policy: pol(...).pred() = tanh(mean); RSSM latent still sampled')
        if a.method == 'dreamer' else 'planning_result(configs/lwy_v2_planner.json)',
        checkpoint=a.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.method == 'dreamer' else None,
        map_manifest_sha256=hashlib.sha256((Path(a.maps)/'manifest.json').read_bytes()).hexdigest(),
        seed=a.seed, body_bands=a.body_bands, speed_cap=a.speed_cap, episodes=env.n,
        valid_starts=int(initial_valid.sum()), loop_steps=loop_steps, seconds=seconds,
        jax_version=jax.__version__, runtime_env=runtime,
        jax_peak_bytes_in_use=peak, leakage=leakage,
        mode_policy_stddev_mean_active=(std_sum/std_count).tolist() if std_count else None,
        initial_unknown_fraction=initial_bev[..., 3].mean(axis=(1, 2)).tolist(),
        summary=summarize(rows), rows=rows)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps(dict(output=str(out), seconds=seconds, loop_steps=loop_steps, **result['summary'])), flush=True)


if __name__ == '__main__':
    main()
