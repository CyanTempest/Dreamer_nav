"""Shared deterministic protocol for validation and all five final methods."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time
from .lwy_eval import wilson
from .lwy_comparison_runtime import setup,level


def run(a):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_safe_obstacles import SafeLegacyEnv
    from .lwy_env_r7 import LwyEnvR7 as SafeLwyEnv
    from .lwy_control_v2 import planning_result
    from .lwy_policy_v2 import EpisodePolicy
    from .control import assess
    from .env import FastEnv,decode
    from .bank import MapBank
    from .evaluate import FAILURES
    output=Path(f'metrics/lwy_{a.tag}.json');assert not output.exists()
    legacy=a.dataset in ['heldout','mini']
    split={'original':'test','mini':'test'}.get(a.dataset,a.dataset)
    bank=MapBank(split) if legacy else LwyBank(split,root=a.root)
    tracker=json.loads(Path('configs/lwy_v2_tracker.json').read_text())
    tracker['obstacle_rule']=a.obstacle_rule
    env=SafeLegacyEnv(bank,1,a.random_obstacles) if legacy else SafeLwyEnv(bank,1,a.random_obstacles,config=tracker)
    n=bank.count*a.episodes;mids=np.repeat(np.arange(bank.count),a.episodes)
    seeds=np.arange(a.seed_start,a.seed_start+n)
    runtime_level=getattr(a,'diagnostic_autotune',None)
    policy=EpisodePolicy(a.method,seeds,a.checkpoint,autotune=level() if runtime_level is None else runtime_level)
    parameters=json.loads(Path('configs/lwy_v2_planner.json').read_text())
    def planner(obs):return planning_result(obs,**parameters)[0]
    @jax.jit
    def filtered(obs,action):
        radius=.5+(action[:,0]+1)*1.25;angle=action[:,1]*jnp.pi/3
        target=jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1)
        risk,unknown,corridor=assess(decode(obs['bev']),target[:,None])
        intervene=(risk[:,0]>.01)|(corridor[:,0]>.15)|(unknown[:,0]>.65)
        if a.support_margin>0:
            from .lwy_support_filter_r7 import support_risk
            intervene|=support_risk(obs,action,a.support_margin)
        return jnp.where(intervene[:,None],planner(obs),action),intervene
    reset=jax.jit(jax.vmap(lambda e,k,m:e._reset(k,m[None]),in_axes=(None,0,0)))
    step=jax.jit(jax.vmap(lambda e,s,x:e._step(s,x[None]),in_axes=(None,0,0)))
    def reward_parts(e,previous,current):
        center,_=e.truth(current,current['pos'])
        station,lateral=e.bank.project(current['mids'],current['pos'])
        _,_,width=e.bank.route_at(current['mids'],station)
        near=jnp.maximum(0,.5-jnp.minimum(-center[:,2],center[:,5]))
        return jnp.stack([-.04*near,-.03*(lateral/jnp.maximum(width/2,.1))**2,
            e.cfg['progress_reward']*jnp.clip(current['s']-previous['s'],-.5,.5),
            -e.cfg['failure_penalty']*(current['failure']>0)],-1)
    parts=jax.jit(jax.vmap(reward_parts,in_axes=(None,0,0)))
    state,obs,extra=reset(env,jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds]),jnp.asarray(mids))
    done=np.zeros(n,bool);rows=[None]*n;traces=[[] for _ in range(n)]
    recent=[deque(maxlen=125) for _ in range(n)]
    hashes=[hashlib.sha256() for _ in range(n)];takeovers=np.zeros(n,np.int64)
    intersection=np.zeros((n,3));union=np.zeros((n,3));begin=time.monotonic()
    reward_sums=np.zeros((n,4))
    for tick in range(env.cfg['max_steps']+2):
        action=policy(obs,planner,predict=policy.head and tick%20==0)
        if policy.head and tick%20==0:
            pred=np.asarray(policy.last_prediction)>.5;truth=np.asarray(obs['priv_trav'][:,0,...,0])>127
            masks=[np.ones_like(truth),np.asarray(extra['occluded'][:,0]),np.asarray(extra['random_near'][:,0])]
            for k,mask in enumerate(masks):
                intersection[:,k]+=(pred&truth&mask).sum((1,2))*~done
                union[:,k]+=((pred|truth)&mask).sum((1,2))*~done
        if a.filter:
            action,changed=filtered({k:v[:,0] for k,v in obs.items()},action)
            takeovers+=np.asarray(changed)&~done;policy.executed(action)
        previous=state
        state,obs,extra=step(env,state,action)
        reward_sums+=np.asarray(parts(env,previous,state)[:,0])*~done[:,None]
        terminal=np.asarray(obs['is_last'][:,0])&~done
        values={k:np.asarray(state[k][:,0]) for k in ['s','pos','yaw','gates','failure','steps','distance','ep_return','vel']}
        poses=np.concatenate([values['pos'],values['yaw'][:,None],np.asarray(action)],-1)
        assert np.isfinite(poses).all()
        for i in np.flatnonzero(~done):
            hashes[i].update(poses[i].tobytes())
            recent[i].append([tick,float(values['s'][i]),*poses[i].tolist()])
            if tick%25==0:traces[i].append([float(values['s'][i]),*poses[i].tolist()])
        for i in np.flatnonzero(terminal):
            failure=int(values['failure'][i]);gates=int(values['gates'][i]);steps=int(values['steps'][i])
            station=float(values['s'][i]);record=bank.records[mids[i]]
            section=None
            if not legacy:
                spec_path=Path('../../ATEC_Env/config/track_spec.json') if split=='test' else Path(a.root)/record['name']/'spec.json'
                spec=json.loads(spec_path.read_text())
                section=next((dict(index=j,label=z.get('label',z.get('name')),start_m=z['start_m'],end_m=z['end_m'])
                    for j,z in enumerate(spec['track']['zones']) if z['start_m']<=station<=z['end_m']),None)
            rows[i]=dict(map=record['name'],map_seed=record['seed'],noise_seed=int(seeds[i]),gates=gates,
                success=gates==8 and failure==0,failure=FAILURES[failure],progress_m=station,
                steps=steps,mean_speed_mps=float(values['distance'][i])/max(.2,steps*.2),
                return_=float(values['ep_return'][i]),takeovers=int(takeovers[i]),
                trav_intersection=intersection[i].tolist() if policy.head else None,
                trav_union=union[i].tolist() if policy.head else None,section=section,
                reward_component_sums=dict(zip(['near_obstacle','lateral','progress','failure'],reward_sums[i].tolist())),
                terminal_velocity_body=values['vel'][i].tolist(),terminal_xy=values['pos'][i].tolist(),terminal_yaw=float(values['yaw'][i]),last_action=np.asarray(action[i]).tolist(),
                terminal_history_5hz=list(recent[i]) if failure else [],trajectory=traces[i],action_pose_sha256=hashes[i].hexdigest())
        done|=terminal
        if tick%1000==0:print(json.dumps(dict(method=a.method,dataset=a.dataset,step=tick,done=int(done.sum()),n=n,seconds=time.monotonic()-begin)),flush=True)
        if done.all():break
    assert done.all(),'Censored episodes cannot be omitted'
    successes=sum(r['success'] for r in rows)
    result=dict(method=a.method,dataset=a.dataset,episodes=n,successes=successes,
        success_rate=successes/n,success_wilson95=wilson(successes,n),mean_gates=float(np.mean([r['gates'] for r in rows])),
        failures={v:sum(r['failure']==v for r in rows) for v in FAILURES.values()},
        takeover_rate=float(takeovers.sum()/sum(r['steps'] for r in rows)),
        mean_speed_mps=float(np.mean([r['mean_speed_mps'] for r in rows])),
        trav_iou=[float(x/y) if y else None for x,y in zip(intersection.sum(0),union.sum(0))] if policy.head else None,
        obstacle_rule=a.obstacle_rule,support_margin=a.support_margin,random_obstacles=a.random_obstacles,safety_filter=a.filter,rows=rows,wall_seconds=time.monotonic()-begin,
        runtime=dict(xla_flags=os.environ['XLA_FLAGS'],tf_cudnn_deterministic=os.environ['TF_CUDNN_DETERMINISTIC'],
            jax=jax.__version__,seed_protocol='independent episode PRNGs; vmap(single environment and policy)',
            noise_seeds=seeds.tolist(),policy_seeds=(seeds+1000000000).tolist(),atlas_arguments='dynamic',
            policy_batch_width=100,dummy_policy_streams_enter_environment=False),
        environment_config=env.cfg,planner_parameters=parameters,checkpoint=a.checkpoint)
    if a.method!='planner':
        result['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint or f'checkpoints/{a.method}/latest.pkl').read_bytes()).hexdigest()
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','runtime']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True)
    p.add_argument('--dataset',choices=['original','blind','val','heldout','mini'],required=True)
    p.add_argument('--episodes',type=int,required=True,help='Episodes per map')
    p.add_argument('--obstacle-rule',choices=['shoulder','legacy'],default='shoulder')
    p.add_argument('--support-margin',type=float,default=0.)
    p.add_argument('--checkpoint');p.add_argument('--tag',required=True)
    p.add_argument('--root',default='maps_lwy_v2')
    p.add_argument('--random-obstacles',action='store_true');p.add_argument('--filter',action='store_true')
    p.add_argument('--seed-start',type=int,default=940000)
    p.add_argument('--diagnostic-autotune',type=int,choices=[0,1],help='Isolated diagnostic only; does not change the formal evaluation default')
    a=p.parse_args();setup()
    if a.diagnostic_autotune is not None:
        assert 'candidate' in a.tag,'Alternate runtime results must be explicitly labeled candidate diagnostics'
        setup(a.diagnostic_autotune)
    run(a)


if __name__=='__main__':main()
