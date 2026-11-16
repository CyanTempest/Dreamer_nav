"""R8 single-scene pilot adapted from lwy_train_r7; fresh replay on new BEV.

The official loop owns callbacks, Replay sampling, Ratio, Agent updates and
checkpoint scheduling. Only Driver and Checkpoint construction are injected;
the checked-out upstream training-loop source is never edited or copied.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import signal
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch


class SegmentComplete(Exception):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--envs',type=int,required=True)
    p.add_argument('--name',default='dreamer_lwy');p.add_argument('--seconds',type=int,default=1800);p.add_argument('--old-replay-items',type=int,default=100000)
    p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();r7=json.loads(Path(a.config).read_text());assert 0<a.seconds<=9000
    from nav3.lwy_eval_v2 import require_g1
    from nav3.lwy_training_runtime import configure
    require_g1()
    training_runtime=configure('dreamer')
    memory_config={}
    if Path('configs/lwy_v2_memory.json').exists():
        memory_config=json.loads(Path('configs/lwy_v2_memory.json').read_text())
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(memory_config['jax_pool_fraction'])
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']=memory_config.get('jax_allocator','default')
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.70'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import elements
    from nav3.dreamer_adapter import DreamerPolicy,embodied
    from nav3.lwy_mixed_r7 import EpisodeMixMonitor
    from r8_p2.environment import make_env
    from nav3.memory import release_host_memory,atomic_json,MemoryMonitor
    from nav3.lwy_checkpoint_r7 import save_tree,load_tree
    assert a.name.startswith('dreamer_lwy_r8_pilot')
    from nav3 import dreamer_adapter
    base_config=dreamer_adapter.make_config
    dreamer_adapter.make_config=lambda *args,**kwargs:base_config(*args,**kwargs).update({'agent.opt.lr':r7['learning_rate']})
    out=Path('checkpoints')/a.name;out.mkdir(parents=True,exist_ok=True)
    context=SimpleNamespace(begin=time.monotonic(),elapsed=0.,stop=False,cp=None,ratio=None,driver=None,
        updates=0,old_ids=set(),sample_old=0,sample_total=0,last_report_old=0,last_report_total=0,
        train_carry=None,pending_carry=None,last_train_metrics={})
    signal.signal(signal.SIGTERM,lambda *_:setattr(context,'stop',True))
    signal.signal(signal.SIGINT,lambda *_:setattr(context,'stop',True))
    monitor=MemoryMonitor(a.name);mix=None
    log=Path(f'metrics/train_{a.name}.jsonl')
    def append(row):
        with log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    env=make_env(a.envs,training=True)
    mix=EpisodeMixMonitor(env.bank.count)
    policy=DreamerPolicy(0,out,jax_overrides=dict(autotune=1,deterministic=True))
    resumed=(out/'official_ckpt/ckpt/latest').exists()
    if not resumed:
        with Path(a.source).open('rb') as f: source=pickle.load(f)
        policy.agent.load(source['agent'])
        del source
        release_host_memory()
        append(dict(event='initialize_r8_pilot',source=a.source,fresh_replay=True,
            optimizer_preserved=True,scope='canonical LWY only, static obstacles',
            test_split_repurposed_for_training=True,not_a_held_out_test=True))
    replay=embodied.Replay(length=65,capacity=200000,directory=f'replay/{a.name}',chunksize=64,
        online=r7.get("online",True),seed=0,save_wait=True,compress=True)
    sampled_online=[0,0]
    original_sample=replay._sample
    def measured_sample(mode):
        seq,online=original_sample(mode)
        if mode=='train':
            sampled_online[0]+=int(online);sampled_online[1]+=1
        return seq,online
    replay._sample=measured_sample
    class Agent:
        def __getattr__(self,key):return getattr(policy.agent,key)
        def init_policy(self,n):return policy.initial(n)
        def policy(self,carry,obs,mode='train'):
            policy.key,key=jax.random.split(policy.key)
            _,(carry,act,outs)=policy._policy(policy.agent.params,carry,obs,mode=mode,seed=key)
            action=jnp.where(obs['is_last'][:,None],0.,act['action'])
            carry=(*carry[:3],{'action':action})
            outs={k:v.astype(policy.agent.spaces[k].dtype) for k,v in outs.items() if k!='finite'}
            return carry,{'action':action},outs
        def train(self,carry,batch):
            if context.pending_carry is not None:
                carry=context.pending_carry;context.pending_carry=None
            carry,outs,metrics=policy.agent.train(carry,batch)
            for key,value in metrics.items():
                if (key.startswith('loss/') or key in ['opt/loss','opt/grad_norm']) and not np.isfinite(np.asarray(value)).all():
                    raise FloatingPointError(f'Nonfinite Dreamer metric {key}')
            if float(metrics.get('opt/grad_norm',0))>1e12:
                append(dict(event='numerical_health_abort',updates=context.updates,
                    metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).ndim==0}))
                raise FloatingPointError('Dreamer raw gradient exceeds 1e12; stop and diagnose before more learning')
            context.train_carry=carry;context.updates+=1
            if context.updates==1:
                context.begin=time.monotonic()
                append(dict(event='training_clock_started',unix=time.time(),budget_seconds=a.seconds))
            context.last_train_metrics={k:float(np.asarray(v)) for k,v in metrics.items()
                if np.asarray(v).ndim==0 and np.isfinite(np.asarray(v))}
            if context.updates<=8 or context.updates%32==0:
                append(dict(event='update_trace',steps=int(logger.step),updates=context.updates,
                    wall_seconds=context.elapsed+time.monotonic()-context.begin,
                    metrics=context.last_train_metrics))
            return carry,outs,metrics
        def save(self):return policy.agent.save()
        def load(self,data):policy.agent.load(data)

    class Replay:
        def __len__(self):
            # Allow new 65-step sequences to form before optimizer updates.
            # The official Ratio is first called after this warmup, so there
            # is no accumulated update backlog and old replay stays available.
            return len(replay) if int(logger.step)>=66*a.envs else 0
        def __getattr__(self,key):return getattr(replay,key)
        def save(self):return replay.save(detailed=True)
        def load(self,data):replay.load(data)

    class Logger:
        def __init__(self):self.step=elements.Counter();self.values={}
        def add(self,values,prefix=None):
            for key,value in values.items():
                arr=np.asarray(value)
                if arr.ndim==0 and np.issubdtype(arr.dtype,np.number) and np.isfinite(arr):
                    self.values[f'{prefix}/{key}' if prefix else key]=float(arr)
        def write(self):
            for key,value in policy.agent.params.items():
                if key.startswith('retnorm/'):
                    self.values['train/'+key]=float(np.asarray(value))
            row=dict(event='train',steps=int(self.step),updates=context.updates,
                wall_seconds=context.elapsed+time.monotonic()-context.begin,metrics=self.values,
                episode_mix=mix.save(),
                replay_samples_in_interval=context.sample_total-context.last_report_total,
                replay_old_sample_fraction=((context.sample_old-context.last_report_old)/(context.sample_total-context.last_report_total)
                    if context.sample_total>context.last_report_total else None),
                replay_old_cumulative_fraction=context.sample_old/max(1,context.sample_total),
                replay_items=len(replay),online_priority_sample_fraction=sampled_online[0]/max(1,sampled_online[1]))
            append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
            self.values={}
            context.last_report_old=context.sample_old;context.last_report_total=context.sample_total
            if monitor.check('train',int(self.step),replay,force=True):context.stop=True
        def close(self):
            if context.last_train_metrics and not any(k.startswith('train/opt/') for k in self.values):
                self.add(context.last_train_metrics,prefix='train')
            self.write()
    logger=Logger()

    class Driver:
        def __init__(self,*_,**__):
            self.callbacks=[];self.carry=None;self.restored=False;self.acts=None
            self.state,self.obs,self.extra=env.reset(jax.random.PRNGKey(610777))
            context.driver=self
        def on_step(self,fn):self.callbacks.append(fn)
        def reset(self,init_policy=None):
            if not self.restored or self.carry is None:self.carry=init_policy(a.envs)
            self.restored=False
        def __call__(self,fn,steps=0,episodes=0):
            if context.stop or (context.updates>0 and time.monotonic()-context.begin>=a.seconds):raise SegmentComplete()
            if self.acts is not None:self.state,self.obs,self.extra=env.step(self.state,self.acts)
            mix.observe(self.state,self.obs)
            self.carry,acts,outs=fn(self.carry,self.obs)
            self.acts=acts['action']
            tran=jax.device_get({**self.obs,**acts,**outs,
                'log/gates':self.state['gates'],'log/reset_fraction':jnp.broadcast_to(jnp.mean(self.obs['is_first']),a.envs)})
            for i in range(a.envs):
                row={k:v[i] for k,v in tran.items()}
                for callback in self.callbacks:callback(row,i)
        def save(self):
            return dict(tree=save_tree((self.state,self.obs,self.extra,self.carry,self.acts),out),envs=a.envs)
        def load(self,data):
            self.state,self.obs,self.extra,self.carry,self.acts=load_tree(data['tree'],data['envs'],a.envs)
            if data['envs']!=a.envs:
                assert data['envs']>a.envs
                tree=(self.state,self.obs,self.extra,self.carry,self.acts)
                tree=jax.tree.map(lambda x:x[:a.envs] if x.ndim and x.shape[0]==data['envs'] else x,tree)
                self.state,self.obs,self.extra,self.carry,self.acts=tree
                append(dict(event='resize_envs',old=data['envs'],new=a.envs,reset=False))
            self.restored=True

    class Runtime:
        def save(self):
            return dict(wall_seconds=context.elapsed+time.monotonic()-context.begin,updates=context.updates,
                ratio_prev=context.ratio._prev,policy_key=np.asarray(policy.key),
                train_carry=jax.device_get(context.train_carry),episode_mix=mix.save(),
                old_ids=list(context.old_ids),sample_old=context.sample_old,sample_total=context.sample_total,
                last_report_old=context.last_report_old,last_report_total=context.last_report_total)
        def load(self,data):
            context.elapsed=data['wall_seconds'];context.updates=data['updates'];context.ratio._prev=data['ratio_prev']
            policy.key=jnp.asarray(data['policy_key']);mix.load(data['episode_mix'])
            context.pending_carry=jax.tree.map(jnp.asarray,data['train_carry']) if data['train_carry'] is not None else None
            context.train_carry=context.pending_carry;context.old_ids=set(data['old_ids'])
            context.sample_old=data['sample_old'];context.sample_total=data['sample_total']
            context.last_report_old=data.get('last_report_old',context.sample_old)
            context.last_report_total=data.get('last_report_total',context.sample_total)
    original_cp=elements.Checkpoint;original_ratio=elements.when.Ratio
    def ratio_factory(value):
        context.ratio=original_ratio(value);return context.ratio
    class Checkpoint(original_cp):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('keep',2)
            super().__init__(*args,**kwargs);self.driver=context.driver;self.runtime=Runtime();context.cp=self
    def stream(buffer,mode):
        while True:
            batch=buffer.sample(16)
            ids=batch['stepid'][...,:16].reshape(-1,16)
            context.sample_old+=sum(row.tobytes() in context.old_ids for row in ids)
            context.sample_total+=len(ids)
            batch['consec']=np.zeros(batch['is_first'].shape,np.int32)
            yield batch
    args=SimpleNamespace(logdir=str(out/'official_ckpt'),usage=dict(psutil=True,nvsmi=False),
        batch_size=16,batch_length=64,train_ratio=32,log_every=60,report_every=0,save_every=600,
        envs=a.envs,debug=True,from_checkpoint='',from_checkpoint_regex='.*',steps=10**15,
        consec_report=1,report_batches=1)
    configuration=dict(effective_runtime={key:os.environ.get(key) for key in training_runtime},r7=r7,method=a.name,seed=0,no_head=False,amnesic=False,direct=False,
        actual_algorithm='embodied.run.train.train (unmodified source), injected GPU Driver',
        envs=a.envs,batch_size=16,batch_length=64,model_preset='size12m',train_ratio=32,
        replay_capacity=200000,retained_old_items=0,lwy_episode_probability=1.,
        scope="canonical LWY pilot, static obstacles, fresh replay; not held-out evaluation",
        warmup_collector_steps=66,training_runtime=training_runtime,
        learning_rate=float(policy.cfg.agent.opt.lr),optimizer=dict(policy.cfg.agent.opt),memory_configuration=memory_config,
        imagination_loss=dict(policy.cfg.agent.imag_loss),return_normalization=dict(policy.cfg.agent.retnorm),
        map_seeds=[r['seed'] for r in env.bank.records],map_names=env.bank.names)
    atomic_json(f'metrics/config_{a.name}.json',configuration)
    training=importlib.import_module('embodied.run.train')
    finished=False
    try:
        with patch.object(embodied,'Driver',Driver),patch.object(elements,'Checkpoint',Checkpoint),patch.object(elements.when,'Ratio',ratio_factory):
            training.train(lambda:Agent(),lambda:Replay(),lambda _:None,stream,lambda:logger,args)
        finished=True
    except SegmentComplete:finished=True
    finally:
        if finished and context.cp is not None:
            context.cp.save();logger.close()
            payload=dict(agent=policy.agent.save(),lwy_round=True,steps=int(logger.step),updates=context.updates,
                wall_seconds=context.elapsed+time.monotonic()-context.begin,envs=a.envs,
                official_checkpoint=str(context.cp.latest()),episode_mix=mix.save())
            temp=out/'latest.tmp'
            with temp.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
            temp.replace(out/'latest.pkl')
            atomic_json(out/'progress.json',{k:payload[k] for k in ['steps','updates','wall_seconds','episode_mix']})
    if monitor.exceeded:raise SystemExit(75)
    assert context.updates>=8, 'Too few optimizer updates to qualify as a successful pilot'


if __name__=='__main__':main()
