"""Exercise new sensor observations through motion, reset and curriculum."""
import json
from pathlib import Path
from nav3.lwy_training_runtime import configure
configure('dreamer')
import jax
import jax.numpy as jnp
import numpy as np
from r8_p2.environment import make_env

env=make_env(8,training=True)
state,obs,_=env.reset(jax.random.PRNGKey(8142))
assert obs['bev'].shape==(8,64,64,6) and obs['bev'].dtype==jnp.uint8
assert np.ptp(np.asarray(state['s']))>100
assert np.all(np.asarray(state['obstacles'][...,2])==0)
start=np.asarray(state['pos'])
for i in range(42):
    state,obs,_=env.step(state,jnp.tile(jnp.array([-.5,0.,0.]),(8,1)))
    assert all(np.isfinite(np.asarray(v)).all() for v in obs.values())
state={**state,'done':jnp.ones(8,bool)}
state,obs,_=env.step(state,jnp.zeros((8,3)))
assert np.asarray(obs['is_first']).all()
evalenv=make_env(2,training=False)
s,_,_=evalenv.reset(jax.random.PRNGKey(8142))
assert np.allclose(np.asarray(s['s']),2.)
result=dict(passed=True,moving_steps=42,reset_pass=True,curriculum_training_only=True,static_obstacles_only=True)
Path('metrics/r8_p2_preflight.json').write_text(json.dumps(result,indent=2));print(result)
