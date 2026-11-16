"""Check that random-obstacle conditions are real and cover the 1 km route."""
import json
from pathlib import Path
from .lwy_eval import setup


def main():
    setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_env import LwyEnv
    bank = LwyBank('test'); env = LwyEnv(bank, 16, random_obstacles=True)
    state = env._reset(jax.random.PRNGKey(930799), jnp.zeros(16, jnp.int32), observe=False)
    props = np.asarray(state['obstacles'])
    assert np.all(props[..., 2] > 0), 'ob=1 must not become an empty narrow-road scene'
    station, lateral = jax.vmap(lambda xy: bank.project(jnp.zeros(16, jnp.int32), xy), in_axes=1, out_axes=1)(state['obstacles'][..., :2])
    _, _, width = bank.route_at(state['mids'], station)
    free = np.asarray(width)-2*props[..., 3]-.03
    assert free.min() >= .68-1e-5
    assert np.all(np.diff(np.asarray(station), axis=1) > 100)
    result = dict(passed=True, episodes=16, active_obstacles=int(np.sum(props[..., 2] > 0)),
        station_range_m=[float(station.min()), float(station.max())],
        nominal_other_side_passage_min_m=float(free.min()),
        note='Placement unit check only; no success-rate claim with random obstacles')
    with Path('metrics/lwy_random_obstacles.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
