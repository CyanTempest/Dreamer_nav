"""Paired episodes with fixed map/episode seeds. All report numbers use JSON."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .bank import MapBank
from .env import FastEnv
from .control import planner,safety_filter,direct_safety_filter
from .provenance import fingerprint

FAILURES = {0:'success',1:'support_or_drop',2:'collision',3:'wrong_branch_or_reverse',5:'stuck_or_timeout'}


def evaluate(policy,split='heldout',episodes=5,random_obstacles=False,filtered=False,seed=9137,method='planner',config=None,limit=None,save=True,direct=False,filter_fn=None,
             root='maps',reset_policy=False,plots=True,trace_hash=False,start_s=None,stop_s=None):
    bank = MapBank(split,limit,root=root)
    n = bank.count*episodes
    if reset_policy and hasattr(policy,'carry'):
        policy.carry=None
        policy.key=jax.random.PRNGKey(seed)
    env = FastEnv(bank,n,random_obstacles,direct,config=config)
    mids = jnp.repeat(jnp.arange(bank.count),episodes)
    starts=None if start_s is None else jnp.broadcast_to(jnp.asarray(start_s,jnp.float32),(n,))
    state,obs,extra = env.reset(jax.random.PRNGKey(seed),mids,starts)
    if trace_hash:
        import hashlib
        hashes=[hashlib.sha256() for _ in range(n)]
    done = np.zeros(n,bool)
    rows = [None]*n
    takeover = np.zeros(n,np.int64)
    intersections = np.zeros((n,3));unions = np.zeros((n,3))
    trajectories = [[] for _ in range(n)]
    start = time.monotonic()
    for step in range(env.cfg['max_steps']+2):
        action = policy(obs)
        if trace_hash:assert np.isfinite(np.asarray(action)).all(),'Nonfinite policy action'
        if getattr(policy,'head',False) and step%20==0:
            prediction=np.asarray(policy.decoder()[...,0])
            if trace_hash:assert np.isfinite(prediction).all(),'Nonfinite traversability prediction'
            pred=prediction>.5
            truth=np.asarray(obs['priv_trav'][...,0])>127
            masks=[np.ones_like(truth),np.asarray(extra['occluded']),np.asarray(extra['random_near'])]
            for k,mask in enumerate(masks):
                intersections[:,k]+=np.sum(pred&truth&mask,axis=(1,2))*(~done)
                unions[:,k]+=np.sum((pred|truth)&mask,axis=(1,2))*(~done)
            if plots and step in (100,500,1000) and not done[0]:
                from .plots import observation
                observation(np.asarray(obs['bev'][0]),truth[0],pred[0],
                    Path('figures')/f'prediction_{method}_{split}_ob{int(random_obstacles)}_sf{int(filtered)}_{step}.png')
        if filtered:
            action,changed = (filter_fn or (direct_safety_filter if direct else safety_filter))(obs,action)
            takeover += np.asarray(changed)&(~done)
            # RSSM conditions on the command that actually reaches the tracker.
            # PPO reads previous executed action through the public state vector.
            if hasattr(policy,'agent'):
                policy.carry=(*policy.carry[:3],{'action':action})
        state,obs,extra = env.step(state,action)
        terminal = np.asarray(obs['is_last'])&(~done)
        reached=np.zeros(n,bool) if stop_s is None else (np.asarray(state['s'])>=np.asarray(stop_s))&(~done)
        if trace_hash:
            values=np.concatenate([np.asarray(action),np.asarray(state['pos']),np.asarray(state['yaw'])[:,None]],axis=1)
            for i in np.flatnonzero(~done):hashes[i].update(values[i].tobytes())
        if step%10==0:
            positions = np.asarray(state['pos'])
            for i in np.flatnonzero(~done):
                trajectories[i].append(positions[i].tolist())
        if (terminal|reached).any():
            vals = {k:np.asarray(state[k]) for k in ['gates','ep_return','distance','steps','s','failure']}
            for i in np.flatnonzero(terminal|reached):
                duration = int(vals['steps'][i])*env.cfg['dt']
                rows[i] = dict(map=bank.names[int(mids[i])],map_seed=bank.records[int(mids[i])]['seed'],
                    episode=int(i%episodes),scenario_seed=seed,method=method,split=split,
                    random_obstacles=bool(random_obstacles),safety_filter=bool(filtered),
                    gates=int(vals['gates'][i]),success=bool(vals['gates'][i]==8) if stop_s is None else bool(reached[i] and vals['failure'][i]==0),
                    failure='segment_pass' if reached[i] and not terminal[i] else FAILURES[int(vals['failure'][i])],return_=float(vals['ep_return'][i]),
                    distance_m=float(vals['distance'][i]),duration_s=duration,
                    progress_m=float(vals['s'][i]),steps=int(vals['steps'][i]),
                    takeovers=int(takeover[i]),mean_speed_mps=float(vals['distance'][i])/max(duration,.2),
                    trav_intersection=intersections[i].tolist() if getattr(policy,'head',False) else None,
                    trav_union=unions[i].tolist() if getattr(policy,'head',False) else None,
                    trajectory=trajectories[i])
                if trace_hash:rows[i]['action_pose_sha256']=hashes[i].hexdigest()
            done |= terminal|reached
        if step%250==0:
            print(json.dumps(dict(method=method,step=step,finished=int(done.sum()),total=n,seconds=time.monotonic()-start)),flush=True)
        if done.all():
            break
    assert done.all(), 'Evaluation must not silently discard censored episodes'
    successes = sum(r['success'] for r in rows)
    result = dict(method=method,split=split,episodes=n,random_obstacles=bool(random_obstacles),filtered=bool(filtered),
        mean_gates=float(np.mean([r['gates'] for r in rows])),success_rate=successes/n,
        failures={name:sum(r['failure']==name for r in rows) for name in FAILURES.values()},
        wall_seconds=time.monotonic()-start,rows=rows,env_config=env.cfg,environment_fingerprint=fingerprint())
    if getattr(policy,'head',False):
        result['trav_iou']=[float(i/u) if u>0 else None for i,u in zip(intersections.sum(0),unions.sum(0))]
    if stop_s is not None:result['failures']['segment_pass']=sum(r['failure']=='segment_pass' for r in rows)
    if reset_policy or trace_hash:
        result['protocol']=dict(root=str(root),seed=seed,policy_seed=seed if reset_policy else None,
            batch_size=n,map_order=bank.names,map_sha256=[r['sha256'] for r in bank.records],
            start_s=np.asarray(starts).tolist() if starts is not None else None,
            stop_s=np.asarray(stop_s).tolist() if stop_s is not None else None,trace_hash=trace_hash)
    if save:
        path = Path('metrics')/f'eval_{method}_{split}_ob{int(random_obstacles)}_sf{int(filtered)}.json'
        path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split',default='heldout',choices=['heldout','test','train'])
    ap.add_argument('--episodes',type=int,default=5)
    ap.add_argument('--random-obstacles',action='store_true')
    ap.add_argument('--filter',action='store_true')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--g1',action='store_true')
    a = ap.parse_args()
    result = evaluate(planner,a.split,a.episodes,a.random_obstacles,a.filter,limit=a.limit)
    if a.g1:
        assert a.split=='heldout' and a.episodes>=5 and not a.random_obstacles and not a.filter and a.limit is None
        gate = dict(gate='G1',passed=result['success_rate']>=.95,threshold=.95,
            success_rate=result['success_rate'],mean_gates=result['mean_gates'],episodes=result['episodes'],
            source='eval_planner_heldout_ob0_sf0.json',environment_fingerprint=result['environment_fingerprint'])
        Path('metrics/g1.json').write_text(json.dumps(gate,indent=2))


if __name__=='__main__':
    main()
