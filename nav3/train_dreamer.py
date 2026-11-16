"""Batched environment driver, official Replay and Agent updates.

Transition convention matches embodied.Driver: o_t, r_t, a_t, recurrent entries.
Updates = floor(new_online_transitions * train_ratio / (16 * 64)).
Pre-fill is separately counted and incurs no hidden training debt.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import time
import jax
import jax.numpy as jnp
import numpy as np
from .bank import MapBank
from .env import FastEnv
from .control import planner
from .dreamer_adapter import DreamerPolicy,embodied
from .provenance import require_g1,fingerprint
from .memory import MemoryMonitor,release_host_memory,save_device_tree,load_device_tree,atomic_json
from .fix_training import add_arguments,training_env,require_fix_g1,ExposureMonitor


def append(path,data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data,allow_nan=False)+'\n')


def gate1():
    require_g1()


def fixed_probe(policy,bank,seed,steps=250,snapshot=None):
    """Fixed TRAIN-map validation; held-out evaluation is separate and never used in updates."""
    n=16
    env=FastEnv(bank,n,True)
    state,obs,extra=env.reset(jax.random.PRNGKey(seed),jnp.arange(n)%bank.count)
    policy.carry=None
    totals=np.zeros(n);active=np.ones(n,bool);gates=np.zeros(n)
    intersection=np.zeros(3);union=np.zeros(3)
    for t in range(steps):
        action,_=policy.policy(obs,predict=policy.head and t%10==0)
        if policy.head and t%10==0:
            pred=np.asarray(policy.last_prediction[...,0])>.5
            target=np.asarray(obs['priv_trav'][...,0])>127
            masks=[np.ones_like(target),np.asarray(extra['occluded']),np.asarray(extra['random_near'])]
            for i,mask in enumerate(masks):
                intersection[i]+=np.sum(pred&target&mask)
                union[i]+=np.sum((pred|target)&mask)
            if snapshot and t==100:
                from .plots import observation
                observation(np.asarray(obs['bev'][0]),target[0],pred[0],Path('figures')/f'{snapshot}.png')
        state,obs,extra=env.step(state,action)
        totals+=np.asarray(obs['reward'])*active
        gates=np.maximum(gates,np.asarray(state['gates'])*active)
        active &= ~np.asarray(obs['is_last'])
    result=dict(probe_return=float(totals.mean()),probe_gates=float(gates.mean()),
        probe_steps=steps,probe_survival=float(active.mean()),
        trav_iou=(intersection/np.maximum(union,1)).tolist() if policy.head else None)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--name',default='dreamer_s0')
    ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--seconds',type=int,default=7200)
    ap.add_argument('--prefill',type=int,default=65536);ap.add_argument('--random-prefill',action='store_true')
    ap.add_argument('--envs',type=int,default=32);ap.add_argument('--ratio',type=float,default=32)
    ap.add_argument('--no-head',action='store_true');ap.add_argument('--amnesic',action='store_true');ap.add_argument('--direct',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--probe-every',type=int,default=25000)
    ap.add_argument('--chunksize',type=int,default=64)
    ap.add_argument('--fresh-replay',action='store_true')
    add_arguments(ap)
    a=ap.parse_args();gate1()
    if a.fix_round:
        require_fix_g1()
        assert a.name=='dreamer_s0_ft' and a.map_root=='maps_mix_v2'
    assert 0<a.seconds<=10500,'Each job must fit below 3 hours including checkpointing'
    start=time.monotonic();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    out=Path('checkpoints')/a.name;out.mkdir(parents=True,exist_ok=True)
    log=Path('metrics')/f'train_{a.name}.jsonl'
    memory=MemoryMonitor(a.name)
    bank=MapBank('train',root=a.map_root)
    exposure=ExposureMonitor(bank,a.name) if a.fix_round else None
    policy=DreamerPolicy(a.seed,out,not a.no_head,a.amnesic)
    policy.cfg=policy.cfg.update({'replay.chunksize':a.chunksize})
    policy.cfg.save(str(out/'config.yaml'))
    env=training_env(bank,a.envs,a.direct,a.margin_weight,a.margin_distance)
    envstate=obs=extra=None
    replay=embodied.Replay(length=65,capacity=1000000,directory=str(Path('replay')/a.name),chunksize=a.chunksize,online=True,seed=a.seed,save_wait=True,
                          compress=True)
    carry_train=policy.agent.init_train(16)
    steps=updates=prefill=0;elapsed_before=0.
    checkpoint=out/'latest.pkl'
    if a.resume and checkpoint.exists():
        with checkpoint.open('rb') as f:data=pickle.load(f)
        policy.agent.load(data['agent'])
        steps,updates,prefill=data['steps'],data['updates'],data['prefill']
        old_envs=data.get('envs') or len(data['env'][1]['is_first'])
        if old_envs==a.envs and not a.reset_environment:
            envstate,obs,extra=(load_device_tree(data['env_snapshot']) if 'env_snapshot' in data
                               else jax.tree.map(jnp.asarray,data['env']))
            policy.carry=jax.tree.map(jnp.asarray,data['policy_carry'])
        else:
            # Keep learned weights, optimizer, replay and counters, but begin fresh
            # episodes so recurrent state and worker batches have the new shape.
            envstate,obs,extra=env.reset(jax.random.PRNGKey(a.seed+steps))
            policy.carry=None
            append(log,dict(event='resize_envs',old_envs=old_envs,new_envs=a.envs,
                            steps=steps,episodes_reset=True))
        policy.key=jnp.asarray(data['policy_key'])
        carry_train=jax.tree.map(jnp.asarray,data['train_carry']);elapsed_before=data['wall_seconds']
        replay_state=data.get('replay_state') if old_envs==a.envs and not a.reset_environment else None
        if exposure:exposure.restore(data.get('fix_exposure'))
        del data
        jax.block_until_ready((envstate,obs,extra,carry_train))
        release_host_memory()
        memory.check('before_replay_load',steps,force=True)
        if not a.fresh_replay:replay.load(replay_state)
        else:
            carry_train=policy.agent.init_train(16)
            append(log,dict(event='fresh_replay_for_new_reward',steps=steps,optimizer_preserved=True))
        del replay_state
        release_host_memory()
        append(log,dict(event='resume',steps=steps,updates=updates,wall_seconds=elapsed_before,
                        loaded_replay_items=len(replay),loaded_replay_steps=sum(c.length for c in replay.chunks.values()),
                        exact_replay_restore=old_envs==a.envs and bool(replay.current)))
        memory.check('after_replay_load',steps,replay,force=True)
    else:
        envstate,obs,extra=env.reset(jax.random.PRNGKey(a.seed))
        # Noninterference is a per-example contract, independent of collector count.
        leak=policy.leakage_check({k:v[:2] for k,v in obs.items()})
        (Path('metrics')/f'audit_actor_{a.name}.json').write_text(json.dumps(leak,indent=2))
    runconfig=dict(**vars(a),map_names=bank.names,map_seeds=[r['seed'] for r in bank.records],
        batch_size=16,batch_length=64,replay_capacity=1000000,replay_chunksize=a.chunksize,model_preset='size12m',environment_fingerprint=fingerprint(),
        parameter_count=sum(x.size for k,x in policy.agent.params.items() if not k.startswith('opt/')),
        actual_algorithm='official DreamerV3 Agent; upstream update unchanged')
    runconfig['replay_storage']='lossless LZ4 RAM blocks for large entries; original dtype and values'
    (Path('metrics')/f'config_{a.name}.json').write_text(json.dumps(runconfig,indent=2))
    def save():
        memory.check('before_save',steps,replay,force=True)
        replay_state=replay.save(detailed=True)
        data=dict(agent=policy.agent.save(),steps=steps,updates=updates,prefill=prefill,
            env_snapshot=save_device_tree((envstate,obs,extra),out),envs=a.envs,
            replay_state=replay_state,policy_carry=jax.device_get(policy.carry),
            policy_key=np.asarray(policy.key),train_carry=jax.device_get(carry_train),
            wall_seconds=elapsed_before+time.monotonic()-start)
        if exposure:data['fix_exposure']=exposure.snapshot()
        temporary=checkpoint.with_suffix('.tmp')
        with temporary.open('wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
        os.replace(temporary,checkpoint)
        atomic_json(out/'progress.json',{k:data[k] for k in ['steps','updates','prefill','wall_seconds']})
        del data,replay_state
        release_host_memory()
        memory.check('after_save',steps,replay,force=True)
    def insert(obs,action,outs,prefill_phase=False):
        if exposure:exposure.observe(envstate,prefill_phase)
        tran=jax.device_get({**{k:obs[k] for k in policy.obs_space},'action':action,**outs})
        resets=np.flatnonzero(tran['is_first'])
        if len(resets):
            mids=np.asarray(envstate['mids'])
            for i in resets:
                record=bank.records[int(mids[i])]
                append(Path('metrics')/f'trainmaps_{a.name}.jsonl',dict(worker=int(i),online_steps=steps,prefill_steps=prefill,
                    map=record['name'],seed=record['seed'],mini=record['mini'],split=record['split']))
        for i in range(a.envs):replay.add({k:v[i] for k,v in tran.items()},i)
    # Only full trajectories from training maps are used to seed model replay.
    # Every worker needs a complete replay sequence before the first update.
    # At thousands of workers a fixed 65536 transitions is too short per worker.
    prefill_target=prefill+max(a.prefill,a.envs*66) if a.fresh_replay else (max(a.prefill,a.envs*66) if steps==0 else prefill)
    while prefill<prefill_target and not stop[0]:
        if a.random_prefill:
            key=jax.random.fold_in(jax.random.PRNGKey(a.seed+800),prefill)
            action=jax.random.uniform(key,(a.envs,3),minval=-1,maxval=1)
        else:
            action=planner(obs)
            if a.direct:
                angle=action[:,1]*jnp.pi/3
                speed=(action[:,2]+1)*.4
                radius=.5+(action[:,0]+1)*1.25
                action=jnp.clip(jnp.stack([speed*jnp.cos(angle)/.8,speed*jnp.sin(angle)/.6,2*speed*jnp.sin(angle)/radius/.75],-1),-1,1)
        action,outs=policy.policy(obs,action_override=action)
        insert(obs,action,outs,prefill_phase=True)
        envstate,obs,extra=env.step(envstate,action);prefill+=a.envs
        if prefill%8192==0:print(json.dumps(dict(prefill=prefill,seconds=time.monotonic()-start)),flush=True)
    if steps==0:
        saved_carry,saved_key=policy.carry,policy.key
        probe=fixed_probe(policy,bank,19001,snapshot=f'probe_{a.name}_0')
        policy.carry,policy.key=saved_carry,saved_key
        append(log,dict(event='probe',steps=0,updates=0,wall_seconds=elapsed_before+time.monotonic()-start,**probe))
    def batches():
        while True:
            batch=replay.sample(16)
            batch['consec']=np.zeros(batch['is_first'].shape,np.int32)
            yield batch
    stream=iter(policy.agent.stream(batches))
    last_log=last_save=time.monotonic();next_probe=(steps//a.probe_every+1)*a.probe_every
    next_milestone=(steps//250000+1)*250000
    episodes=[];mets={}
    while steps<a.steps and time.monotonic()-start<a.seconds and not stop[0] and not memory.exceeded:
        action,outs=policy.policy(obs)
        insert(obs,action,outs)
        envstate,obs,extra=env.step(envstate,action)
        steps+=a.envs
        terminal=np.asarray(obs['is_last'])
        if terminal.any():
            rs=np.asarray(envstate['ep_return']);gs=np.asarray(envstate['gates'])
            episodes.extend((float(rs[i]),int(gs[i])) for i in np.flatnonzero(terminal))
        target_updates=int(steps*a.ratio/(16*64))
        while updates<target_updates and not stop[0] and (not a.fix_round or time.monotonic()-start<a.seconds):
            if memory.check('train',steps,replay):break
            carry_train,routs,mets=policy.agent.train(carry_train,next(stream))
            for key,value in mets.items():
                if (key.startswith('loss/') or key in ('opt/loss','opt/grad_norm')) and not np.isfinite(np.asarray(value)).all():
                    raise FloatingPointError(f'Nonfinite Dreamer metric: {key}')
            if 'replay' in routs:replay.update(routs['replay'])
            updates+=1
        if memory.exceeded:break
        if time.monotonic()-last_log>60:
            scalars={k:float(np.asarray(v)) for k,v in mets.items() if np.asarray(v).ndim==0 and np.isfinite(np.asarray(v))}
            row=dict(event='train',steps=steps,updates=updates,prefill=prefill,wall_seconds=elapsed_before+time.monotonic()-start,
                achieved_ratio=updates*1024/max(steps,1),episodes=len(episodes),
                mean_return=float(np.mean([e[0] for e in episodes])) if episodes else None,
                mean_gates=float(np.mean([e[1] for e in episodes])) if episodes else None,metrics=scalars)
            append(log,row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
            episodes=[];last_log=time.monotonic()
        if not a.fix_round and steps>=next_probe:
            saved_carry,saved_key=policy.carry,policy.key
            probe=fixed_probe(policy,bank,19001,snapshot=f'probe_{a.name}_{steps}')
            policy.carry,policy.key=saved_carry,saved_key
            append(log,dict(event='probe',steps=steps,updates=updates,wall_seconds=elapsed_before+time.monotonic()-start,**probe))
            next_probe+=a.probe_every
        if not a.fix_round and steps>=next_milestone:
            from .efficiency import measure
            measure(policy,a.name,steps,elapsed_before+time.monotonic()-start,a.direct)
            next_milestone+=250000
        if time.monotonic()-last_save>900:
            save();last_save=time.monotonic()
    save()
    append(log,dict(event='end',steps=steps,updates=updates,prefill=prefill,wall_seconds=elapsed_before+time.monotonic()-start))
    print(json.dumps(dict(done=a.name,steps=steps,updates=updates,seconds=time.monotonic()-start)),flush=True)
    if memory.exceeded:raise SystemExit(75)


if __name__=='__main__':main()
