"""Fixed-stream short regression with controller and physical-substep evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .lwy_eval import setup


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True)
    p.add_argument('--root',default='maps_lwy'); p.add_argument('--start',type=float,default=970.)
    p.add_argument('--stop',type=float,default=998.); p.add_argument('--candidates',type=int,default=15)
    p.add_argument('--offset-weight',type=float,default=1.);p.add_argument('--lookahead',type=float,default=1.2)
    p.add_argument('--prediction-speed',type=float,default=.5);p.add_argument('--no-drift',action='store_true')
    p.add_argument('--prediction-margin',type=float,default=.04)
    p.add_argument('--yaw-gain',type=float,default=2.);p.add_argument('--tracker-v2',action='store_true')
    p.add_argument('--horizon',type=int,default=70);p.add_argument('--border-weight',type=float,default=300.)
    p.add_argument('--capture-file');p.add_argument('--resume-capture');p.add_argument('--match-full')
    p.add_argument('--record',action='store_true');p.add_argument('--assert-pass',action='store_true')
    p.add_argument('--record-start',type=float,default=0.);p.add_argument('--max-steps',type=int,default=2500)
    a=p.parse_args();setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .lwy_env import LwyEnv
    if a.tracker_v2:
        from .lwy_env_v2 import LwyEnvV2 as LwyEnv
    from .lwy_control_v2 import planning_result
    from .diag_rollout import physics_trace
    if a.tracker_v2:
        from .lwy_env_v2 import physics_trace_v2 as physics_trace
    from .env import rotate
    out=Path(f'metrics/lwy_{a.tag}.json');assert not out.exists()
    bank=LwyBank('blind',root=a.root);n=bank.count;mids=np.arange(n)
    if a.resume_capture:
        capture=dict(np.load(a.resume_capture));mids=capture['episode_indices'];n=len(mids)
    config={'position_walk_std':0.,'yaw_walk_std':0.} if a.no_drift else {}
    if a.tracker_v2:config['yaw_tracking_gain']=a.yaw_gain
    env=LwyEnv(bank,1,False,config=config)
    reset=jax.jit(jax.vmap(lambda e,k,m:e._reset(k,m[None],jnp.array([a.start],jnp.float32)),in_axes=(None,0,0)))
    step=jax.jit(jax.vmap(lambda e,s,x:e._step(s,x[None]),in_axes=(None,0,0)))
    def instrument(e,s,x):
        look=s['pos'][:,None]+rotate(jnp.array([[[0.,0.],[.4,0.]]]),s['yaw'])
        root=jnp.any(e.bank.sample(s['mids'],look,6)==2,-1)
        x=x.at[:,2].set(jnp.where(root,jnp.minimum(x[:,2],e.cfg['terrain_speed']/.4-1),x[:,2]))
        sub=physics_trace(e,s,x)
        positions=sub[0][:,0];yaws=sub[1][:,0]
        points=positions[:,None]+rotate(e.foot[None],yaws)
        raster=e.bank.sample(jnp.repeat(s['mids'],4),points)
        return dict(sub_pos=positions,sub_yaw=yaws,sub_vel=sub[2][:,0],sub_cmd=sub[3][:,0],
            sub_support=.025-raster[...,2].max(-1),sub_obstacle=raster[...,5].min(-1))
    traced=jax.jit(jax.vmap(lambda e,s,x:instrument(e,s,x[None]),in_axes=(None,0,0)))
    keys=jnp.stack([jax.random.PRNGKey(940000+int(i)) for i in mids])
    if a.resume_capture:
        state={k.removeprefix('state__'):jnp.asarray(v) for k,v in capture.items() if k.startswith('state__')}
        obs={k.removeprefix('obs__'):jnp.asarray(v) for k,v in capture.items() if k.startswith('obs__')}
        stop_s=capture['stop_s']
    else:
        state,obs,_=reset(env,keys,jnp.asarray(mids));stop_s=np.full(n,a.stop)
    capture_spec=json.loads(Path(a.capture_file).read_text()) if a.capture_file else {}
    captured={}
    done=np.zeros(n,bool);rows=[None]*n;frames=[];hashes=[hashlib.sha256() for _ in range(n)]
    begin=time.monotonic();trace_error=0.
    for tick in range(a.max_steps):
        if capture_spec:
            stations=np.asarray(state['s'][:,0])
            for i in np.flatnonzero(~done):
                spec=capture_spec.get(str(bank.records[mids[i]]['seed']))
                if spec and i not in captured and stations[i]>=spec['start']:
                    captured[i]=({k:np.asarray(v[i]) for k,v in state.items()},
                        {k:np.asarray(v[i]) for k,v in obs.items()},spec['stop'])
        flat={k:v[:,0] for k,v in obs.items()}
        action,plan=planning_result(flat,a.candidates,a.offset_weight,a.lookahead,a.prediction_speed,a.prediction_margin,a.horizon,a.border_weight)
        recording=a.record and np.any((np.asarray(state['s'][:,0])>=a.record_start)&~done)
        if recording:
            sub={k:np.asarray(v) for k,v in traced(env,state,action).items()}
            pre={k:np.asarray(state[k][:,0]) for k in ['pos','yaw','s','drift','vel','delay','gain']}
        nxt,nobs,_=step(env,state,action)
        after={k:np.asarray(nxt[k][:,0]) for k in ['s','pos','yaw','failure']}
        aa=np.asarray(action)
        for i in np.flatnonzero(~done):hashes[i].update(np.r_[after['pos'][i],after['yaw'][i],aa[i]].tobytes())
        terminal=np.asarray(nobs['is_last'][:,0]);reached=after['s']>=stop_s
        if recording:
            ids=np.flatnonzero((pre['s']>=a.record_start)&~done)
            trace_error=max(trace_error,float(np.max(np.abs(sub['sub_pos'][ids,-1]-after['pos'][ids]))))
            for i in ids:
                frame={k:v[i] for k,v in pre.items()};frame.update({k:v[i] for k,v in sub.items()})
                frame.update(episode=i,tick=tick,action=aa[i],offset=np.asarray(plan['offset'])[i],
                    scores=np.asarray(plan['scores'])[i],risk=np.asarray(plan['selected_risk'])[i],border=np.asarray(plan['selected_border'])[i])
                frames.append(frame)
        for i in np.flatnonzero((terminal|reached)&~done):
            rows[i]=dict(map=bank.names[mids[i]],seed=int(940000+mids[i]),map_seed=bank.records[mids[i]]['seed'],
                segment_pass=bool(reached[i] and not after['failure'][i]),failure=int(after['failure'][i]),
                s=float(after['s'][i]),xy=after['pos'][i].tolist(),yaw=float(after['yaw'][i]),
                steps=tick+1,action_pose_sha256=hashes[i].hexdigest())
        done|=terminal|reached;state,obs=nxt,nobs
        if tick%200==0:print(json.dumps(dict(tick=tick,done=int(done.sum()),seconds=time.monotonic()-begin)),flush=True)
        if done.all():break
    assert done.all()
    if capture_spec:
        assert len(captured)==len(capture_spec),(list(captured),capture_spec)
        ids=sorted(captured);saved=dict(episode_indices=mids[ids],stop_s=np.array([captured[i][2] for i in ids]))
        for index,name in [(0,'state'),(1,'obs')]:
            saved.update({name+'__'+k:np.stack([captured[i][index][k] for i in ids]) for k in captured[ids[0]][index]})
        np.savez_compressed(out.with_name(out.stem+'_capture.npz'),**saved)
    if a.record:
        assert trace_error<2e-5,trace_error
        np.savez_compressed(out.with_suffix('.npz'),**{k:np.asarray([f[k] for f in frames]) for k in frames[0]})
    result=dict(arguments=vars(a),rows=rows,segment_pass=sum(r['segment_pass'] for r in rows),episodes=n,
        wall_seconds=time.monotonic()-begin,physics_trace_endpoint_error_m=trace_error)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if a.match_full:
        old=json.loads(Path(a.match_full).read_text())
        assert [r['action_pose_sha256'] for r in rows]==[r['action_pose_sha256'] for r in old['rows']], 'Capture replay changed full episode behavior'
    if a.assert_pass:assert result['segment_pass']==n, f"Reproduced: {n-result['segment_pass']} premature failures"


if __name__=='__main__':main()
