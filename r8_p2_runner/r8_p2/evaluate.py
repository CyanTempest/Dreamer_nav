"""Paired development evaluation on the training scene, never a held-out test."""
import argparse,json,pickle,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=16)
    a=p.parse_args()
    from nav3.lwy_training_runtime import configure
    configure('dreamer')
    import jax
    import numpy as np
    from r8_p2.environment import make_env
    from nav3.dreamer_adapter import DreamerPolicy
    env=make_env(a.episodes,training=False)
    policy=DreamerPolicy(0,'checkpoints/r8_p2_eval',jax_overrides=dict(autotune=1,deterministic=True))
    with Path(a.checkpoint).open('rb') as f: policy.agent.load(pickle.load(f)['agent'])
    state,obs,_=env.reset(jax.random.PRNGKey(814233))
    leakage=policy.leakage_check(obs)
    rows={};begin=time.time()
    for step in range(18002):
        action=policy(obs);state,obs,_=env.step(state,action)
        done=np.asarray(obs['is_last'])
        if done.any():
            host=jax.device_get({k:state[k] for k in ['gates','failure','s','ep_return','steps']})
            for i in np.flatnonzero(done):
                if int(i) not in rows:
                    rows[int(i)]={k:float(v[i]) for k,v in host.items()}
        if len(rows)==a.episodes:break
    assert len(rows)==a.episodes,(len(rows),a.episodes)
    result=dict(scope='same canonical scene used for training; development evaluation only',checkpoint=a.checkpoint,episodes=a.episodes,seed=814233,
        success_rate=sum(r['gates']==8 and r['failure']==0 for r in rows.values())/a.episodes,
        mean_gates=float(np.mean([r['gates'] for r in rows.values()])),seconds=time.time()-begin,leakage=leakage,rows=rows)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
