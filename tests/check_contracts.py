"""Real environment contracts, excluding optimizer updates (safe before G1)."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nav3.bank import MapBank
from nav3.env import FastEnv
from nav3.control import planner


bank=MapBank('heldout',1)
env=FastEnv(bank,4)
state,obs,extra=env.reset(jax.random.PRNGKey(2))
assert obs['bev'].shape==(4,64,64,6)
assert obs['state'].shape==(4,13)
assert obs['priv_trav'].shape==(4,64,64,1)
assert np.asarray(obs['is_first']).all()
assert set(np.unique(obs['priv_trav']))<={0,255}
assert float(jnp.mean(obs['priv_trav']>0))>0
start=np.asarray(state['s'])
for _ in range(30):state,obs,extra=env.step(state,planner(obs))
assert np.isfinite(np.asarray(obs['state'])).all()
assert (np.asarray(state['s'])>start+.5).all(),np.asarray(state['s'])
# Collision and unsupported footprint must terminate; a point center alone is insufficient.
bad=dict(state);bad['pos']=state['pos']+jnp.array([0.,8.]);bad['done']=jnp.zeros(4,bool)
bad,fail,_=env.step(bad,jnp.zeros((4,3)))
assert np.asarray(fail['is_last']).all()
assert (np.asarray(bad['failure'])==1).all()
fresh,first,_=env.step(bad,jnp.zeros((4,3)))
assert np.asarray(first['is_first']).all()
assert np.asarray(first['reward']==0).all()
out=dict(contract_passed=True,forward_progress_m=(np.asarray(state['s'])-start).tolist(),
         unsupported_terminates=True,terminal_then_first=True)
Path('metrics/contracts.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
