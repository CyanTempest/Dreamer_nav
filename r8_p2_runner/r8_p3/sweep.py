import json
from pathlib import Path
from nav3.lwy_training_runtime import configure
configure('dreamer')
import jax
import jax.numpy as jnp
import numpy as np
from r8_p2.environment import make_env
from nav3.env import rotate

env=make_env(1)
s=jnp.arange(990.,1000.001,.025);m=jnp.zeros(len(s),jnp.int32)
pos,yaw,width=env.bank.route_at(m,s)
fp=pos[:,:2][:,None]+rotate(env.foot[None],yaw)
t=env.bank.sample(m,fp)
maxedge=np.asarray(t[...,2].max(-1));ob=np.asarray(t[...,5].min(-1))
rows=[dict(s=float(x),maxedge=float(e),obstacle_min=float(o),feasible=bool(e<=.025 and o>=0)) for x,e,o in zip(s,maxedge,ob)]
result=dict(gates=np.asarray(env.bank.gates).tolist(),rows=rows,last_feasible_s=max(r['s'] for r in rows if r['feasible']))
Path('metrics/r8_p3_endpoint_sweep.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
