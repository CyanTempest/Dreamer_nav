"""Capture why the planner considered the final action safe, before a failure."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nav3.bank import MapBank
from nav3.env import FastEnv,rotate,decode
from nav3.control import planning_result

parser=argparse.ArgumentParser();parser.add_argument('--memory',type=float,default=8.);args=parser.parse_args()
bank=MapBank('heldout');n=100;env=FastEnv(bank,n,config={'memory_seconds':args.memory})
state,obs,_=env.reset(jax.random.PRNGKey(9137),jnp.repeat(jnp.arange(bank.count),5))
done=np.zeros(n,bool);records=[];lastpos=state['pos'];lastyaw=state['yaw'];directory=Path('metrics/failure_snapshots');directory.mkdir(exist_ok=True)
for step in range(6001):
    action,debug=planning_result(obs)
    old=state;before=obs
    state,obs,_=env.step(state,action)
    terminal=np.asarray(obs['is_last'])&(~done)
    if terminal.any():
        for i in np.flatnonzero(terminal):
            failed=int(state['failure'][i])
            rec=dict(index=int(i),map=bank.names[i//5],s=float(state['s'][i]),gates=int(state['gates'][i]),failure=failed,
                selected_risk=float(debug['selected_risk'][i]),selected_border=float(debug['selected_border'][i]),offset=float(debug['offset'][i]))
            records.append(rec)
            if failed:
                world=state['pos'][:,None]+rotate(env.foot[None],state['yaw'])
                relative=rotate(world-lastpos[:,None],-lastyaw)
                truth,_=env.truth(state,world)
                np.savez_compressed(directory/f'm{args.memory}_{i}.npz',bev=np.asarray(before['bev'][i]),foot=np.asarray(relative[i]),
                    truth=np.asarray(truth[i]),action=np.asarray(action[i]),velocity=np.asarray(old['vel'][i]),
                    age=np.asarray(old['age'][i]),state_pos=np.asarray(state['pos'][i]),state_yaw=np.asarray(state['yaw'][i]))
            print(json.dumps(rec),flush=True)
        done|=terminal
    lastpos,lastyaw=old['pos'],old['yaw']
    if done.all():break
Path(f'metrics/diagnose_full_m{args.memory}.json').write_text(json.dumps(dict(rows=records,success=sum(r['gates']==8 for r in records)),indent=2))
