import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
if '--lwy-round' in sys.argv:
    from .lwy_training_runtime import configure
    configure('ppo')
import jax
import jax.numpy as jnp
import numpy as np
from .bank import MapBank
from .env import FastEnv
from .ppo import PPOPolicy,act,advantages,update
from .provenance import require_g1,fingerprint
from .memory import MemoryMonitor,release_host_memory,atomic_json
from .fix_training import add_arguments,training_env,require_fix_g1,ExposureMonitor


def append(path,data):
    with Path(path).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')


def probe(policy,bank):
    # Identical fixed 250-step horizon and 16 TRAIN-map poses as Dreamer probe.
    n=16;env=FastEnv(bank,n,True)
    state,obs,_=env.reset(jax.random.PRNGKey(19001),jnp.arange(n)%bank.count)
    policy.carry=None;totals=np.zeros(n);active=np.ones(n,bool);gates=np.zeros(n)
    for _ in range(250):
        state,obs,_=env.step(state,policy(obs))
        totals+=np.asarray(obs['reward'])*active
        gates=np.maximum(gates,np.asarray(state['gates'])*active)
        active &= ~np.asarray(obs['is_last'])
    return dict(probe_return=float(totals.mean()),probe_gates=float(gates.mean()),probe_survival=float(active.mean()),probe_steps=250)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--name',default='ppo_s0')
    ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--seconds',type=int,default=7200)
    ap.add_argument('--envs',type=int,default=128);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--lwy-round',action='store_true')
    add_arguments(ap)
    a=ap.parse_args();assert 0<a.seconds<=10500
    require_g1()
    if a.lwy_round:
        from .lwy_eval_v2 import require_g1 as require_lwy_g1
        from .lwy_ppo_update import update as updater
        require_lwy_g1()
        assert a.name in ['ppo_lwy','ppo_lwy_runtime2'] and a.margin_weight==0
    else:updater=update
    if a.fix_round:
        require_fix_g1()
        assert a.name=='ppo_s0_ft' and a.map_root=='maps_mix_v2'
    start=time.monotonic();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    out=Path('checkpoints')/a.name;out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'latest.pkl';log=Path('metrics')/f'train_{a.name}.jsonl'
    memory=MemoryMonitor(a.name)
    if a.lwy_round:
        from .lwy_mixed import load_banks,MixedEnv,EpisodeMixMonitor
        new,old=load_banks()
        env=MixedEnv(new,old,a.envs,tracker_config=json.loads(Path('configs/lwy_v2_tracker.json').read_text()))
        bank=env.bank;mix=EpisodeMixMonitor(new.count)
    else:
        bank=MapBank('train',root=a.map_root);env=training_env(bank,a.envs,weight=a.margin_weight,margin=a.margin_distance)
    exposure=ExposureMonitor(bank,a.name) if a.fix_round else None
    state,obs,extra=env.reset(jax.random.PRNGKey(a.seed));policy=PPOPolicy(a.seed)
    steps=updates=0;elapsed_before=0.;rng=np.random.default_rng(a.seed+42)
    if a.resume and checkpoint.exists():
        with checkpoint.open('rb') as f:data=pickle.load(f)
        policy.params,policy.optimizer,policy.carry,policy.key=jax.tree.map(jnp.asarray,data['policy'])
        old_envs=len(data['env'][1]['is_first'])
        source_checkpoint=a.lwy_round and not data.get('lwy_round',False)
        if old_envs==a.envs and not a.reset_environment and not source_checkpoint:
            state,obs,extra=jax.tree.map(jnp.asarray,data['env'])
        else:
            policy.carry=None
            append(log,dict(event='resize_envs',old_envs=old_envs,new_envs=a.envs,
                            steps=data['steps'],episodes_reset=True))
        steps,updates,elapsed_before=data['steps'],data['updates'],data['wall_seconds']
        if source_checkpoint:
            append(log,dict(event='initialize_lwy_from_original',source_steps=steps,source_updates=updates,
                            source_wall_seconds=elapsed_before,optimizer_preserved=True))
            steps=updates=0;elapsed_before=0.
        if a.lwy_round and data.get('episode_mix'):mix.load(data['episode_mix'])
        if 'minibatch_rng' in data:rng.bit_generator.state=data['minibatch_rng']
        if exposure:exposure.restore(data.get('fix_exposure'))
        append(log,dict(event='resume',steps=steps,updates=updates,wall_seconds=elapsed_before))
        del data
        release_host_memory()
    config=dict(**vars(a),algorithm='clipped recurrent PPO',sequence_length=32,rollout_length=128,epochs=4,
        minibatch_sequences=16,clip=.2,discount=.997,gae_lambda=.95,learning_rate=.0003,gru_units=256,
        actor_privileged=False,critic_privileged=False,map_names=bank.names,map_seeds=[r['seed'] for r in bank.records],
        parameter_count=sum(x.size for x in policy.params.values()),environment_fingerprint=fingerprint())
    if a.lwy_round:config['training_runtime']=configure('ppo')
    (Path('metrics')/f'config_{a.name}.json').write_text(json.dumps(config,indent=2))
    def save():
        data=dict(policy=jax.device_get((policy.params,policy.optimizer,policy.carry,policy.key)),env=jax.device_get((state,obs,extra)),
            steps=steps,updates=updates,minibatch_rng=rng.bit_generator.state,
            wall_seconds=elapsed_before+time.monotonic()-start)
        if exposure:data['fix_exposure']=exposure.snapshot()
        if a.lwy_round:data.update(lwy_round=True,episode_mix=mix.save())
        tmp=checkpoint.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
        os.replace(tmp,checkpoint)
        atomic_json(out/'progress.json',{k:data[k] for k in ['steps','updates','wall_seconds']})
    def do_probe():
        carry,key=policy.carry,policy.key
        result=probe(policy,bank)
        policy.carry,policy.key=carry,key
        append(log,dict(event='probe',steps=steps,updates=updates,wall_seconds=elapsed_before+time.monotonic()-start,**result))
        print(json.dumps(result),flush=True)
    if steps==0 and not a.lwy_round:do_probe()
    last_save=time.monotonic();next_probe=(steps//25000+1)*25000
    next_milestone=(steps//250000+1)*250000
    while steps<a.steps and time.monotonic()-start<a.seconds and not stop[0] and not memory.exceeded:
        rollout=[];episodes=[]
        for t in range(128):
            if a.lwy_round:mix.observe(state,obs)
            if exposure:exposure.observe(state)
            first=np.flatnonzero(np.asarray(obs['is_first']))
            if len(first):
                mids=np.asarray(state['mids'])
                for i in first:
                    record=bank.records[int(mids[i])]
                    append(Path('metrics')/f'trainmaps_{a.name}.jsonl',dict(worker=int(i),online_steps=steps+t*a.envs,
                        map=record['name'],seed=record['seed'],mini=record['mini'],split=record['split']))
            action,info=policy.policy(obs)
            nstate,nobs,nextra=env.step(state,action)
            # Bootstrap from the actual terminal observation, before reset.
            _,_,_,_,nv=act(policy.params,policy.carry,nobs,jax.random.PRNGKey(0))
            rollout.append(dict(bev=obs['bev'],state=obs['state'],first=obs['is_first'],valid=~obs['is_last'],
                reward=nobs['reward'],last=nobs['is_last'],terminal=nobs['is_terminal'],next_value=nv,**info))
            terminal=np.asarray(nobs['is_last'])
            if terminal.any():
                rs,gs=np.asarray(nstate['ep_return']),np.asarray(nstate['gates'])
                episodes.extend((float(rs[i]),int(gs[i])) for i in np.flatnonzero(terminal))
            state,obs,extra=nstate,nobs,nextra
        data=jax.tree.map(lambda *x:jnp.stack(x),*rollout)
        jax.block_until_ready(data);del rollout
        adv,returns=advantages(data['reward'],data['value'],data['next_value'],data['last'],data['terminal'],data['valid'])
        data.update(advantage=adv,**{'return':returns})
        # Sequence minibatches retain per-segment GRU state and actual reset markers.
        sequences=jax.tree.map(lambda x:x.reshape((4,32,a.envs,*x.shape[2:])).swapaxes(1,2).reshape((-1,32,*x.shape[2:])),data)
        jax.block_until_ready(sequences);del data,adv,returns
        metrics={}
        for epoch in range(4):
            order=rng.permutation(4*a.envs)
            for i in range(0,len(order),16):
                batch=jax.tree.map(lambda x:x[order[i:i+16]].swapaxes(0,1),sequences)
                policy.params,policy.optimizer,metrics=updater(policy.params,policy.optimizer,batch)
                if a.lwy_round:
                    raw_grad=float(metrics['grad_norm'])
                    if not np.isfinite(raw_grad) or raw_grad>1e12:
                        append(log,dict(event='numerical_health_abort',updates=updates,raw_gradient_norm=raw_grad))
                        raise FloatingPointError('PPO raw gradient exceeds 1e12 or is nonfinite; stop before another update')
                updates+=1
        jax.block_until_ready(policy.params);del sequences,batch
        steps+=128*a.envs
        memory.check('train',steps,force=True)
        row=dict(event='train',steps=steps,updates=updates,wall_seconds=elapsed_before+time.monotonic()-start,
            episodes=len(episodes),mean_return=float(np.mean([e[0] for e in episodes])) if episodes else None,
            mean_gates=float(np.mean([e[1] for e in episodes])) if episodes else None,
            metrics={k:float(v) for k,v in metrics.items()})
        if a.lwy_round:
            assert all(np.isfinite(v) for v in row['metrics'].values()), 'Nonfinite PPO training metrics'
        if a.lwy_round:row['episode_mix']=mix.save()
        append(log,row);print(json.dumps(row),flush=True)
        if not (a.fix_round or a.lwy_round) and steps>=next_probe:do_probe();next_probe+=25000
        if not (a.fix_round or a.lwy_round) and steps>=next_milestone:
            from .efficiency import measure
            measure(policy,a.name,steps,elapsed_before+time.monotonic()-start)
            next_milestone+=250000
        if time.monotonic()-last_save>900:save();last_save=time.monotonic()
    if not (a.fix_round or a.lwy_round):do_probe()
    save()
    append(log,dict(event='end',steps=steps,updates=updates,wall_seconds=elapsed_before+time.monotonic()-start))
    if memory.exceeded:raise SystemExit(75)


if __name__=='__main__':main()
