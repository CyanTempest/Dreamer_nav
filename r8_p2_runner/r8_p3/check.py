import json
from pathlib import Path
from nav3.lwy_training_runtime import configure
configure('dreamer')
import jax
import jax.numpy as jnp
import numpy as np
from r8_p2.environment import make_env as old_env
from r8_p3.environment import make_env,FINISH_MARGIN
from nav3.env import rotate

a=old_env(4);b=make_env(4)
key=jax.random.PRNGKey(83140)
sa,oa,_=a.reset(key);sb,ob,_=b.reset(key)
assert all(np.array_equal(np.asarray(oa[k]),np.asarray(ob[k])) for k in oa),'Early-route observations changed'
assert np.isclose(FINISH_MARGIN,.85),FINISH_MARGIN
m=jnp.zeros(4,jnp.int32)
s=jnp.array([999.15,999.15,1000.,1000.])
# Evaluate physical original route, without the goal-target clamp.
pos,yaw,_=a.bank.route_at(m,s)
xy=pos[:,:2][:,None]+rotate(b.foot[None],yaw)
ta=a.bank.sample(m,xy);tb=b.bank.sample(m,xy)
assert np.array_equal(np.asarray(ta),np.asarray(tb)),'Physical geometry changed'
assert np.all(np.asarray(tb[:2,:,2].max(-1))<=.025)
assert np.all(np.asarray(tb[2:,:,2].max(-1))>.025)
state,obs,_=b.reset(key,start_s=jnp.full(4,999.10))
state={**state,'queue':jnp.tile(jnp.array([.5,0.,0.]),(4,6,1)),'vel':jnp.tile(jnp.array([.5,0.,0.]),(4,1))}
state,obs,_=b.step(state,jnp.tile(jnp.array([-.8,0.,.25]),(4,1)))
assert np.asarray(obs['is_last']).all()
assert np.all(np.asarray(state['gates'])==8) and np.all(np.asarray(state['failure'])==0)
# An off-road robot remains a failure; reaching the finish cannot cancel it.
state,obs,_=b.reset(key,start_s=jnp.full(4,999.10))
state['pos']=state['pos']+rotate(jnp.tile(jnp.array([0.,2.]),(4,1)),state['yaw'])
state,obs,_=b.step(state,jnp.zeros((4,3)))
assert np.all(np.asarray(state['failure'])>0)
result=dict(passed=True,finish_margin_m=FINISH_MARGIN,finish_station_m=999.15,early_observations_identical=True,physical_geometry_identical=True,original_endpoint_infeasible=True,new_goal_center_feasible=True,offroad_still_fails=True)
Path('metrics/r8_p3_checks.json').write_text(json.dumps(result,indent=2));print(result)
