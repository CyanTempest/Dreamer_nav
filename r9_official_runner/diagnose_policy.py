"""Frozen-policy diagnostics, explicitly separate invalid starts from RL outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--maps',required=True);p.add_argument('--checkpoint')
    p.add_argument('--output',required=True);p.add_argument('--body-bands',action='store_true');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--method',choices=['dreamer','planner','follower'],default='dreamer')
    p.add_argument('--safety-filter',action='store_true')
    p.add_argument('--speed-cap',type=float,default=.8)
    p.add_argument('--seed',type=int,default=914233)
    p.add_argument('--filter-hold-steps',type=int,default=0)
    p.add_argument('--action',choices=['sample','mode'],default='sample')
    a=p.parse_args()
    if a.action=='mode' and a.method!='dreamer':p.error('--action mode is defined for Dreamer only')
    from nav3.lwy_training_runtime import configure
    configure('dreamer')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scan_env import make_env
    from nav3.env import decode,rotate
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    env=make_env(a.maps,repeats=a.repeats,body_bands=a.body_bands)
    state,obs,extra=env.reset(jax.random.PRNGKey(a.seed))
    assert obs['bev'].shape==(env.n,64,64,6) and obs['state'].shape==(env.n,13)
    assert obs['bev'].dtype==jnp.uint8 and np.isfinite(np.asarray(obs['state'])).all()
    # Outside a patch must be unsafe regardless of route metadata.
    outside=env.bank.sample(jnp.arange(env.bank.count),jnp.full((env.bank.count,2),1e6))
    assert np.all(np.asarray(outside[...,2])>0)
    fp=state['pos'][:,None]+rotate(env.foot[None],state['yaw'])
    physical,_=env.truth(state,fp)
    initial_valid=np.asarray(jnp.all((physical[...,2]<=.025)&(physical[...,5]>=0),-1))
    initial_bev=np.asarray(decode(obs['bev']))
    np.savez_compressed(out.with_suffix('.npz'),bev=initial_bev,valid=initial_valid,pos=np.asarray(state['pos']),yaw=np.asarray(state['yaw']))
    if a.method=='dreamer':
        if not a.checkpoint:p.error('Dreamer requires --checkpoint')
        from nav3.dreamer_adapter import DreamerPolicy
        policy=DreamerPolicy(0,str(out.parent/'policy_runtime'),jax_overrides=dict(autotune=1,deterministic=True),deterministic=(a.action=='mode'))
        with Path(a.checkpoint).open('rb') as stream: policy.agent.load(pickle.load(stream)['agent'])
        leakage=policy.leakage_check(obs)
    elif a.method=='planner':
        from nav3.lwy_control_v2 import planning_result
        parameters=json.loads(Path('configs/lwy_v2_planner.json').read_text())
        policy=jax.jit(lambda o:planning_result(o,**parameters)[0]);leakage={'not_applicable':'nonlearned observation-only planner'}
    else:
        from nav3.control import target_action
        policy=jax.jit(lambda o:target_action(o['state'][:,2:4]*4,jnp.full(len(o['state']),.28)))
        leakage={'not_applicable':'nonlearned public-waypoint follower'}
    if a.safety_filter:
        if a.method!='dreamer':p.error('Safety filter is defined for Dreamer only')
        from nav3.lwy_control_v2 import planning_result
        from nav3.control import assess
        from nav3.lwy_support_filter_r7 import support_risk
        parameters=json.loads(Path('configs/lwy_v2_planner.json').read_text())
        @jax.jit
        def filtered(o,action):
            radius=.5+(action[:,0]+1)*1.25;angle=action[:,1]*jnp.pi/3
            target=jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1)
            risk,unknown,corridor=assess(decode(o['bev']),target[:,None])
            intervene=(risk[:,0]>.01)|(unknown[:,0]>.65)|(corridor[:,0]>.15)|support_risk(o,action,.08)
            fallback=planning_result(o,**parameters)[0]
            return fallback,intervene
    rows={i:dict(valid_start=False,diagnostic='invalid_geometry_start; excluded from policy outcomes') for i in range(env.n) if not initial_valid[i]}
    begin=time.time();history=[];interventions=np.zeros(env.n,int);active_steps=np.zeros(env.n,int)
    filter_hold=jnp.zeros(env.n,jnp.int32)
    for step in range(a.steps):
        action=policy(obs)
        # Cap before the optional filter so it assesses the action that would be executed.
        action=action.at[:,2].set(jnp.minimum(action[:,2],a.speed_cap/.4-1))
        active=np.array([i not in rows for i in range(env.n)])
        if a.safety_filter:
            fallback,triggered=filtered(obs,action)
            filter_hold=jnp.where(obs['is_first'],0,filter_hold)
            changed=triggered|(filter_hold>0)
            action=jnp.where(changed[:,None],fallback,action)
            filter_hold=jnp.where(triggered,a.filter_hold_steps,jnp.maximum(0,filter_hold-1))
            interventions+=np.asarray(changed)*active
        action=action.at[:,2].set(jnp.minimum(action[:,2],a.speed_cap/.4-1))
        active_steps+=active
        if a.method=='dreamer':
            # RSSM action history must contain the action actually executed.
            policy.carry=(*policy.carry[:3],{'action':action})
        if not np.isfinite(np.asarray(action)).all():raise ValueError('Nonfinite policy action')
        state,obs,_=env.step(state,action)
        if step%10==0:history.append(np.asarray(state['pos']))
        done=np.asarray(obs['is_last'])
        if done.any():
            host=jax.device_get({k:state[k] for k in ['gates','failure','s','ep_return','steps']})
            for i in np.flatnonzero(done):
                if int(i) not in rows:rows[int(i)]={**{k:float(v[i]) for k,v in host.items()},'valid_start':True}
        if len(rows)==env.n:break
    for i in range(env.n):
        if i not in rows:rows[i]=dict(valid_start=True,truncated=True,s=float(state['s'][i]))
    for i,row in rows.items():
        row['map']=env.bank.names[i%env.bank.count]
        row['filter_interventions']=int(interventions[i]);row['active_steps']=int(active_steps[i])
    qualified=[r['name'] for r in env.bank.records if r['conservative_proxy_qualified']]
    result=dict(scope='Official scan local diagnostic only; provisional surface proxy, scanner route excerpts, not race success rate',
                method=a.method,action=a.action if a.method=='dreamer' else None,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.method=='dreamer' else None,
                map_manifest_sha256=hashlib.sha256((Path(a.maps)/'manifest.json').read_bytes()).hexdigest(),
                seed=a.seed,body_bands=a.body_bands,safety_filter=a.safety_filter,speed_cap=a.speed_cap,
                filter_hold_steps=a.filter_hold_steps,
                filter_intervention_fraction=float(interventions.sum()/max(1,active_steps.sum())),
                episodes=env.n,valid_starts=int(initial_valid.sum()),
                qualified_map_names=qualified,formal_benchmark_ready=False,
                initial_unknown_fraction=initial_bev[...,3].mean(axis=(1,2)).tolist(),
                initial_support_fraction=initial_bev[...,2].mean(axis=(1,2)).tolist(),
                initial_obstacle_fraction=(initial_bev[...,1]>.22).mean(axis=(1,2)).tolist(),
                seconds=time.time()-begin,leakage=leakage,rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_name(out.stem+'_rollout.npz'),positions=np.array(history))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,6,figsize=(15,8),layout='constrained')
    columns=['Ground (m)','Obstacle (m)','Support proxy','Unknown','Route prior','Age']
    for row,i in enumerate(np.linspace(0,env.bank.count-1,3).astype(int)):
        for k,title in enumerate(columns):
            im=axes[row,k].imshow(initial_bev[i,:,:,k].T,origin='lower',extent=(-2,6,-4,4),vmin=-.5 if k==0 else 0,vmax=1 if k!=1 else 2,cmap='viridis')
            axes[row,k].set_title(title if row==0 else '')
            axes[row,k].set_xlabel('Forward (m)')
            if k==0:axes[row,k].set_ylabel(env.bank.names[i]+'\nLeft (m)')
    fig.suptitle('Official scan BEV: '+('body-height bands (adaptation)' if a.body_bands else 'all bands (R8 sensing baseline)'))
    fig.savefig(out.with_suffix('.png'),dpi=140)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
