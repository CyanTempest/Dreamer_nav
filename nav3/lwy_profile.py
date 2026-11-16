"""Environment-only matched throughput/memory probe, no learning or replay."""
import argparse
import json
from pathlib import Path
import time
import os


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['legacy', 'lwy'], required=True)
    p.add_argument('--envs', type=int, default=1024); p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    from .lwy_eval import setup
    setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    if a.mode == 'lwy':
        from .lwy_bank import LwyBank
        from .lwy_env import LwyEnv
        bank = LwyBank('test'); env = LwyEnv(bank, a.envs)
    else:
        from .bank import MapBank
        from .env import FastEnv
        # Read existing atlas files directly; never create a protected map cache.
        bank = MapBank.__new__(MapBank); bank.count = 1
        for k in bank.ARRAY_KEYS: setattr(bank, k, jnp.asarray(np.load(Path('maps/atlas_test_1')/f'{k}.npy')))
        env = FastEnv(bank, a.envs)
    start = time.monotonic()
    state, obs, extra = env.reset(jax.random.PRNGKey(9306), jnp.zeros(a.envs, jnp.int32))
    action = jnp.zeros((a.envs, 3))
    for _ in range(5): state, obs, extra = env.step(state, action)
    jax.block_until_ready(state)
    warm = time.monotonic()-start
    start = time.monotonic()
    for _ in range(a.steps): state, obs, extra = env.step(state, action)
    jax.block_until_ready(state)
    elapsed = time.monotonic()-start
    result = dict(mode=a.mode, envs=a.envs, steps=a.steps, warmup_compile_seconds=warm,
        seconds=elapsed, env_steps_per_second=a.envs*a.steps/elapsed,
        memory_stats=jax.devices()[0].memory_stats(), xla_flags=os.environ['XLA_FLAGS'],
        atlas_bytes=sum(getattr(bank, k).size*getattr(bank, k).dtype.itemsize for k in bank.ARRAY_KEYS),
        includes_agent=False, includes_replay=False,
        comparison_limit='legacy mini-trail 200m versus original LWY 1km; identical observation/action sizes and deterministic XLA settings')
    with Path(f'metrics/lwy_profile_{a.mode}_{a.envs}.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
