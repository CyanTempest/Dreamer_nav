"""Measure real mixed-map collector, original Agent and optimizer allocations."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['dreamer','ppo'],required=True)
    p.add_argument('--envs',type=int,required=True);p.add_argument('--output');p.add_argument('--live-output',default='metrics/lwy_r7_memory_live.json');a=p.parse_args()
    from .lwy_eval_v2 import require_g1
    from .lwy_training_runtime import configure
    require_g1()
    assert json.loads(Path('metrics/lwy_r7_gate.json').read_text())['passed']
    training_runtime=configure(a.method)
    pool=Path('configs/lwy_v2_memory.json')
    if pool.exists():
        config=json.loads(pool.read_text())
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(config['jax_pool_fraction'])
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']=config.get('ppo_allocator','default') if a.method=='ppo' else config.get('jax_allocator','default')
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.70'
    output=Path(a.output or f'metrics/lwy_r7_memory_{a.method}_{a.envs}.json');assert not output.exists()
    samples=[];stop=threading.Event();stage=['loading'];begin=time.monotonic()
    def monitor():
        last=0.
        while not stop.is_set():
            used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
            samples.append(dict(seconds=time.monotonic()-begin,stage=stage[0],used_mib=used,total_mib=total))
            if time.monotonic()-last>=5:
                Path(a.live_output).write_text(json.dumps(dict(method=a.method,envs=a.envs,
                    **samples[-1],peak_used_mib=max(r['used_mib'] for r in samples))))
                last=time.monotonic()
            stop.wait(.25)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    result=dict(method=a.method,envs=a.envs,finite_loss=False,completed=False,training_runtime=training_runtime,
        jax_pool_fraction=float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION','.95')),
        jax_allocator=os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR','default'))
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .lwy_mixed_r7 import load_banks,MixedEnv
        from .memory import release_host_memory
        lwy,legacy=load_banks()
        env=MixedEnv(lwy,legacy,a.envs,tracker_config=json.loads(Path('configs/lwy_v2_tracker.json').read_text()))
        result['map_counts']=[lwy.count,legacy.count]
        checkpoint_path=(Path('checkpoints/snapshots/ppo_lwy_runtime2_3.50h/latest.pkl')
                         if a.method=='ppo' else Path('checkpoints/dreamer_s0/latest.pkl'))
        with checkpoint_path.open('rb') as f:checkpoint=pickle.load(f)
        if a.method=='dreamer':
            from .dreamer_adapter import DreamerPolicy,embodied
            policy=DreamerPolicy(logdir=f'logs/lwy_r7_memory_dreamer_{a.envs}',jax_overrides=dict(autotune=1,deterministic=True))
            policy.agent.load(checkpoint['agent']);del checkpoint;release_host_memory()
            state,obs,_=env.reset(jax.random.PRNGKey(743));stage[0]='collector'
            replay=embodied.Replay(length=65,capacity=4096,chunksize=64,online=True,seed=0,compress=True)
            for _ in range(66):
                action,outs=policy.policy(obs)
                tran=jax.device_get({**obs,'action':action,**outs})
                for i in range(16):replay.add({k:v[i] for k,v in tran.items()},i)
                state,obs,_=env.step(state,action)
            stage[0]='optimizer'
            def batches():
                while True:
                    batch=replay.sample(16);batch['consec']=np.zeros(batch['is_first'].shape,np.int32);yield batch
            stream=iter(policy.agent.stream(batches()));carry=policy.agent.init_train(16)
            for _ in range(8):
                carry,_,metrics=policy.agent.train(carry,next(stream))
                assert all(np.isfinite(np.asarray(v)).all() for k,v in metrics.items() if k.startswith('loss/') or k=='opt/loss')
            stage[0]='mixed_training'
            # Repeated fixed-shape updates expose live optimizer allocations;
            # their count does not need to match the subsequent training ratio.
            for _ in range(3):
                action,_=policy.policy(obs);state,obs,_=env.step(state,action)
                for _ in range(2):carry,_,metrics=policy.agent.train(carry,next(stream))
            jax.block_until_ready((obs,carry));result['loss_keys']=[k for k in metrics if 'loss' in k]
        else:
            from .ppo import PPOPolicy,act,advantages,update
            from .lwy_ppo_update_r7 import update
            policy=PPOPolicy();policy.params,policy.optimizer=jax.tree.map(jnp.asarray,checkpoint['policy'][:2])
            del checkpoint;release_host_memory()
            state,obs,_=env.reset(jax.random.PRNGKey(743));stage[0]='rollout128';rollout=[]
            for _ in range(128):
                action,info=policy.policy(obs);nxt,nobs,_=env.step(state,action)
                _,_,_,_,value=act(policy.params,policy.carry,nobs,jax.random.PRNGKey(0))
                rollout.append(dict(bev=obs['bev'],state=obs['state'],first=obs['is_first'],valid=~obs['is_last'],
                    reward=nobs['reward'],last=nobs['is_last'],terminal=nobs['is_terminal'],next_value=value,**info))
                state,obs=nxt,nobs
            data=jax.tree.map(lambda *x:jnp.stack(x),*rollout);jax.block_until_ready(data);del rollout
            adv,returns=advantages(data['reward'],data['value'],data['next_value'],data['last'],data['terminal'],data['valid'])
            data.update(advantage=adv,**{'return':returns})
            sequences=jax.tree.map(lambda x:x.reshape((4,32,a.envs,*x.shape[2:])).swapaxes(1,2).reshape((-1,32,*x.shape[2:])),data)
            jax.block_until_ready(sequences);del data,adv,returns
            stage[0]='optimizer'
            for i in range(8):
                batch=jax.tree.map(lambda x:x[i*16:(i+1)*16].swapaxes(0,1),sequences)
                policy.params,policy.optimizer,metrics=update(policy.params,policy.optimizer,batch)
                assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
            jax.block_until_ready(policy.params);result['loss_keys']=list(metrics)
        result.update(completed=True,finite_loss=True)
    except Exception as exc:
        result['error']=repr(exc)
        raise
    finally:
        stop.set();worker.join()
        result.update(wall_seconds=time.monotonic()-begin,samples=samples,
            peak_used_mib=max((r['used_mib'] for r in samples),default=0),
            peak_fraction=max((r['used_mib']/r['total_mib'] for r in samples),default=0))
        with output.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)


if __name__=='__main__':main()
