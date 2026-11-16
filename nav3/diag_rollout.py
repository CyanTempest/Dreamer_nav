"""Read-only checkpoint diagnostics; all products use a diag_ prefix.

Example: python -m nav3.diag_rollout --method dreamer_s0 --tag full_dreamer
Never train or save a policy. The optional root contains counterfactual maps.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from .bank import MapBank
from .env import FastEnv,rotate,wrap
from .control import planner
from .evaluate import FAILURES


def load_policy(name):
    if name=='planner':return planner
    cfg=json.loads(Path(f'metrics/config_{name}.json').read_text())
    with Path(f'checkpoints/{name}/latest.pkl').open('rb') as f:data=pickle.load(f)
    if name.startswith('dreamer'):
        from .dreamer_adapter import DreamerPolicy
        # Agent initialization writes only to a new diagnostic log directory.
        policy=DreamerPolicy(cfg['seed'],'logs/diag_agent',not cfg['no_head'],cfg['amnesic'])
        policy.agent.load(data['agent'])
    else:
        from .ppo import PPOPolicy
        policy=PPOPolicy(cfg['seed']);policy.params=jax.tree.map(jnp.asarray,data['policy'][0])
    return policy


def physics_trace(env,state,action):
    """Replay the four actual 50 ms substeps, including the identical slip key.

    Isolated diagnostic mirror of FastEnv._step; endpoint equality is checked.
    """
    c=env.cfg;action=jnp.clip(action,-1,1)
    _,slipkey,_=jax.random.split(state['key'],3)
    radius=.5+(action[:,0]+1)*1.25;angle=action[:,1]*jnp.pi/3
    target=state['pos']+rotate(jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1),state['yaw'])
    terrain=env.bank.sample(state['mids'],state['pos'][:,None]+rotate(jnp.array([[[-.42,0],[.42,0]]]),state['yaw']),0,linear=True)
    speed=jnp.minimum((action[:,2]+1)*.4,jnp.where(jnp.abs(terrain[:,1]-terrain[:,0])>.12,c['terrain_speed'],c['max_speed']))
    slip=jax.random.normal(slipkey,(4,env.n,2))*c['slip_std']
    def step(carry,noise):
        pos,yaw,vel,queue=carry
        local=rotate(target-pos,-yaw);norm=jnp.maximum(jnp.linalg.norm(local,axis=-1),.15)
        cmd=jnp.concatenate([local/norm[:,None]*jnp.minimum(speed,norm)[:,None],
            jnp.clip(2*speed*local[:,1]/norm**2,-c['max_yaw_rate'],c['max_yaw_rate'])[:,None]],-1)
        queue=jnp.concatenate([cmd[:,None],queue[:,:-1]],1)
        delayed=queue[jnp.arange(env.n),state['delay']]
        lim=jnp.array([c['max_accel'],c['max_accel'],c['max_yaw_accel']])*c['control_dt']
        vel=vel+jnp.clip((delayed*state['gain']-vel)*(c['control_dt']/c['motor_tau']),-lim,lim)
        scale=jnp.minimum(1.,jnp.linalg.norm(vel[:,:2],axis=-1)/.5)
        pos=pos+rotate(vel[:,:2]+noise*scale[:,None],yaw)*c['control_dt'];yaw=wrap(yaw+vel[:,2]*c['control_dt'])
        data,_=env.truth(state,pos[:,None]+rotate(env.foot[None],yaw))
        hit=jnp.any(data[...,5]<0,-1)
        return (pos,yaw,vel,queue),(pos,yaw,vel,cmd,hit)
    _,trace=jax.lax.scan(step,(state['pos'],state['yaw'],state['vel'],state['queue']),slip)
    return trace


def run(args):
    start=time.monotonic();out=Path('metrics')/f'diag_{args.tag}'
    if out.with_suffix('.json').exists():raise FileExistsError(out)
    policy=load_policy(args.method)
    shadow=load_policy('ppo_s0') if args.shadow else None
    for p in [policy,shadow]:
        if hasattr(p,'carry'):p.carry=None;p.key=jax.random.PRNGKey(args.seed)
    bank=MapBank(args.split,root=args.root);n=args.episodes*bank.count
    env=FastEnv(bank,n,False)
    mids=jnp.repeat(jnp.arange(bank.count),args.episodes)
    start_s=jnp.full(n,args.start,jnp.float32);stop_s=np.full(n,args.stop)
    if args.around_log:
        logs=np.array([json.loads((Path(args.root)/name/'metadata.json').read_text())['features']['log'] for name in bank.names])
        logs=np.repeat(logs,args.episodes);start_s=jnp.asarray(logs+args.log_start_offset,dtype=jnp.float32);stop_s=logs+7
    if args.start==2 and not args.around_log:
        state,obs,extra=env.reset(jax.random.PRNGKey(args.seed),mids)
    else:
        state,obs,extra=env.reset(jax.random.PRNGKey(args.seed),mids,start_s)
    def oracle_obstacles(s,o):
        xy=s['pos'][:,None,None]+rotate(env.local[None],s['yaw'])
        values,_=env.truth(s,xy);height=jnp.maximum(values[...,1]-values[...,0],0)
        bev=o['bev'].at[...,1].set((jnp.clip(height/2,0,1)*255).astype(jnp.uint8))
        bev=bev.at[...,3].set(jnp.where(height>.20,0,bev[...,3]))
        return {**o,'bev':bev}
    oracle_fn=jax.jit(oracle_obstacles)
    trace_fn=jax.jit(lambda s,a:physics_trace(env,s,a))
    done=np.zeros(n,bool);rows=[None]*n;frames=[];paths=[[] for _ in range(n)]
    max_error=0.
    for tick in range(6002):
        if args.oracle_obstacles:obs=oracle_fn(state,obs)
        action=policy(obs)
        shadow_action=shadow(obs) if shadow else None
        progress=np.asarray(state['s']);active=(~done)&(progress>=np.asarray(start_s) if args.around_log else progress>=args.record_start)&(progress<=stop_s)
        indices=np.flatnonzero(active)
        capture=len(indices)>0 and args.record
        if capture:
            pred=np.asarray(policy.decoder()) if getattr(policy,'head',False) else None
            sub=[np.asarray(v) for v in trace_fn(state,action)]
            current={k:np.asarray(state[k]) for k in ['pos','yaw','s','vel','drift','steps']}
            ob={k:np.asarray(obs[k]) for k in ['bev','priv_trav','state']}
            aa=np.asarray(action);pa=np.asarray(planner(obs))
        next_state,next_obs,extra=env.step(state,action)
        after={k:np.asarray(next_state[k]) for k in ['pos','yaw','s','failure','gates','steps']}
        terminal=np.asarray(next_obs['is_last'])&(~done)
        reached=(after['s']>=stop_s)&(~done)
        if capture:
            error=float(np.max(np.abs(sub[0][-1,indices]-after['pos'][indices])))
            max_error=max(max_error,error)
            assert error<2e-5,(tick,error)
            for i in indices:
                f=dict(episode=i,tick=tick,pos=current['pos'][i],yaw=current['yaw'][i],s=current['s'][i],
                    vel=current['vel'][i],drift=current['drift'][i],bev=ob['bev'][i],priv_trav=ob['priv_trav'][i],
                    vector=ob['state'][i],action=aa[i],planner_action=pa[i],next_pos=after['pos'][i],next_yaw=after['yaw'][i],
                    sub_pos=sub[0][:,i],sub_yaw=sub[1][:,i],sub_vel=sub[2][:,i],sub_cmd=sub[3][:,i],sub_hit=sub[4][:,i])
                if pred is not None:f['pred']=pred[i].astype(np.float16)
                if shadow is not None:f['ppo_action']=np.asarray(shadow_action)[i]
                frames.append(f)
        for i in np.flatnonzero(~done):
            if tick%5==0 or terminal[i] or reached[i]:paths[i].append([float(after['s'][i]),*after['pos'][i].tolist(),float(after['yaw'][i])])
        for i in np.flatnonzero(terminal|reached):
            rows[i]=dict(episode=int(i),map=bank.names[int(mids[i])],failure=FAILURES[int(after['failure'][i])] if terminal[i] else 'segment_pass',
                progress_m=float(after['s'][i]),pos=after['pos'][i].tolist(),yaw=float(after['yaw'][i]),gates=int(after['gates'][i]),
                steps=int(after['steps'][i]),trajectory=paths[i])
        done|=terminal|reached
        state,obs=next_state,next_obs
        if tick%250==0:print(json.dumps(dict(tag=args.tag,tick=tick,done=int(done.sum()),frames=len(frames),seconds=time.monotonic()-start)),flush=True)
        if done.all():break
        if time.monotonic()-start>1700:raise TimeoutError('Diagnostic GPU experiment exceeded 1700 seconds')
    assert done.all()
    if frames:
        np.savez_compressed(out.with_suffix('.npz'),**{k:np.asarray([f[k] for f in frames]) for k in frames[0]})
    result=dict(arguments=vars(args),episodes=n,rows=rows,wall_seconds=time.monotonic()-start,trace_endpoint_max_error_m=max_error,
        failures={k:sum(r['failure']==k for r in rows) for k in set(r['failure'] for r in rows)})
    if args.method!='planner':result['checkpoint_sha256']=hashlib.sha256(Path(f'checkpoints/{args.method}/latest.pkl').read_bytes()).hexdigest()
    out.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',default='dreamer_s0');p.add_argument('--tag',required=True)
    p.add_argument('--root',default='maps');p.add_argument('--split',default='test');p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=9137);p.add_argument('--start',type=float,default=2.);p.add_argument('--stop',type=float,default=171.)
    p.add_argument('--log-start-offset',type=float,default=-14.);p.add_argument('--around-log',action='store_true');p.add_argument('--oracle-obstacles',action='store_true')
    p.add_argument('--record-start',type=float,default=150.);p.add_argument('--record',action='store_true');p.add_argument('--shadow',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
