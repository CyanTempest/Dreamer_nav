"""R7 default/legacy reset parity and unchanged footprint/step regression."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from nav3.lwy_bank import LwyBank
from nav3.lwy_env_r7 import LwyEnvR7
from nav3.lwy_safe_obstacles import SafeLwyEnv
from nav3.lwy_env_v2 import LwyEnvV2

bank=LwyBank('test',root='maps_lwy_v2')
old=SafeLwyEnv(bank,1,True)
legacy=LwyEnvR7(bank,1,True,config={'obstacle_rule':'legacy'})
fixed=LwyEnvR7(bank,1,True)
key=jax.random.PRNGKey(940000);mid=jnp.array([0])
a=old._reset(key,mid,observe=False);b=legacy._reset(key,mid,observe=False)
assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
c=fixed._reset(key,mid,observe=False)
assert np.array_equal(a['obstacles'][...,2:],c['obstacles'][...,2:])
assert not np.array_equal(a['obstacles'][...,:2],c['obstacles'][...,:2])
assert all(np.array_equal(a[k],c[k]) for k in a if k!='obstacles')
assert np.array_equal(fixed.foot,old.foot) and fixed.foot.shape==(1075,2)
assert LwyEnvR7._step is LwyEnvV2._step
from nav3.env import encode
from nav3.lwy_support_filter_r7 import support_risk
y=-4+(jnp.arange(64)+.5)*.125
road=jnp.broadcast_to(jnp.abs(y)[None,None,:]<.55,(1,64,64))
bev=jnp.zeros((1,64,64,6)).at[...,2].set(road).at[...,4].set(road)
obs={'bev':encode(bev),'state':jnp.zeros((1,13))}
assert not bool(support_risk(obs,jnp.array([[0.,0.,0.]]),.08)[0])
assert bool(support_risk(obs,jnp.array([[0.,1.,1.]]),.08)[0])
result=dict(passed=True,legacy_reset_exact=True,only_placement_changed=True,footprint_unchanged=True,
    physics_step_identical=True,support_filter_synthetic_passed=True,default_rule='shoulder',old_rule='legacy')
with Path('metrics/lwy_r7_unit_checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
