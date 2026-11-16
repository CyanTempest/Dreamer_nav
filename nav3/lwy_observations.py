"""Save actual noisy observations at narrow-road, root, rail and stair poses."""
from pathlib import Path
import json
from .lwy_eval import setup


def main():
    setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_env import LwyEnv
    from .env import decode
    stations = np.array([8., 190., 320., 480., 920., 985.], np.float32)
    bank = LwyBank('test'); env = LwyEnv(bank, len(stations))
    state, obs, extra = env.reset(jax.random.PRNGKey(930699), jnp.zeros(len(stations), jnp.int32), jnp.asarray(stations))
    image = np.asarray(decode(obs['bev']))
    result = dict(stations_m=stations.tolist(), unknown_fraction=image[..., 3].mean((1, 2)).tolist(),
        corridor_cells_at_body_x=image[:, 16, :, 4].sum(-1).tolist(),
        observation_shape=list(obs['bev'].shape[1:]), state_shape=list(obs['state'].shape[1:]), action_shape=[3],
        spec_changed=False, note='Single delayed/noisy frame after reset; not a detector accuracy benchmark')
    with Path('metrics/lwy_observations.json').open('x') as f: json.dump(result, f, indent=2)
    np.savez_compressed('metrics/lwy_observations.npz', stations=stations, bev=image,
        priv_trav=np.asarray(obs['priv_trav']), visible=~np.asarray(extra['occluded']))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
