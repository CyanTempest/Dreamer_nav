"""Deterministic red/green fixture: all heldout maps, start 6 m before log.

Captures the true envelope and observed geometry at first failure for diagnosis.
Never used for training or formal success figures.
"""
import json
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nav3.bank import MapBank
from nav3.env import FastEnv,decode,rotate
from nav3.control import planner,lane_planner,planning_result
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--memory',type=float,default=1.4);parser.add_argument('--planner',choices=['grid','lane'],default='grid')
args=parser.parse_args()
if args.planner=='lane':planner=lane_planner
bank=MapBank('heldout');env=FastEnv(bank,bank.count,config={'memory_seconds':args.memory})
starts=jnp.array([r['features']['log']-6 for r in bank.records])
state,obs,extra=env.reset(jax.random.PRNGKey(9137),jnp.arange(bank.count),starts)
done=np.zeros(bank.count,bool);rows=[];trace=[]
for step in range(400):
    action=planner(obs)
    debug=planning_result(obs)[1] if args.planner=='grid' else {}
    if step%2==0:
        trace.append(dict(step=step,s=np.asarray(state['s']).tolist(),pos=np.asarray(state['pos']).tolist(),action=np.asarray(action).tolist()))
    before=obs
    state,obs,extra=env.step(state,action)
    failure=np.asarray(obs['is_last'])&(~done)
    passed=(np.asarray(state['s'])>np.asarray(starts)+9)&(~done)
    for i in np.flatnonzero(failure|passed):
        footprint=state['pos'][:,None]+rotate(env.foot[None],state['yaw'])
        ftruth,_=env.truth(state,footprint)
        rows.append(dict(map=bank.names[i],s=float(state['s'][i]),failure=int(state['failure'][i]),passed=bool(passed[i]),
            pos=np.asarray(state['pos'][i]).tolist(),yaw=float(state['yaw'][i]),
            footprint_truth=np.asarray(ftruth[i]).tolist(),action=np.asarray(action[i]).tolist(),
            debug={k:np.asarray(v[i]).tolist() for k,v in debug.items()}))
        if i<3:
            b=np.asarray(decode(before['bev'])[i]);truth=np.asarray(before['priv_trav'][i,...,0])
            fig,ax=plt.subplots(1,4,figsize=(12,3),layout='constrained')
            for a,img,title in zip(ax,[b[...,1],b[...,3],b[...,4],truth],['Observed obstacle','Unknown','Route','Privileged traversability']):
                a.imshow(img,origin='lower',extent=(-4,4,-2,6));a.set_title(title)
            fig.savefig(f'figures/diagnose_log_{i}.png',dpi=150);plt.close(fig)
    done |= failure|passed
    if done.all():break
result=dict(passed=sum(r['passed'] for r in rows)>=.95*len(bank.names),rows=rows,trace=trace,
    unresolved=[bank.names[i] for i in np.flatnonzero(~done)],final_s=np.asarray(state['s']).tolist(),config=vars(args))
Path(f'metrics/diagnose_log_{args.planner}_{args.memory}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=result['passed'],rows=rows),indent=2))
assert result['passed'], 'Planner must pass deterministic log fixture on >=95% maps'
