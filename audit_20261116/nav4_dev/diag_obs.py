"""C3 diagnostic: evaluate with the LEGACY support rule (nav3/env.py:173-175: jnp.roll slope < 0.8,
obstacle < 0.22, ground > -0.55) patched into nav4.sensor, then run nav4.evaluate with the given args.
Usage: diag_obs.py legacy_support <nav4.evaluate args...>"""
import sys
import jax.numpy as jnp
import nav4.sensor as sensor


def legacy_support(height, known, obstacle, cfg):
    dx = (jnp.roll(height, -2, axis=1)-jnp.roll(height, 2, axis=1))/.5
    dy = (jnp.roll(height, -2, axis=2)-jnp.roll(height, 2, axis=2))/.5
    return (jnp.hypot(dx, dy) < .8) & (obstacle < .22) & (height > -.55)


assert sys.argv[1] == 'legacy_support'
sensor.support_mask = legacy_support
from nav4.evaluate import main  # noqa: E402
main(sys.argv[2:])
