"""Paired development evaluation on the training scene, never a held-out test."""
import argparse,json,pickle,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=16)
    p.add_argument('--seed',type=int,default=814233)
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
    state,obs,_=env.reset(jax.random.PRNGKey(a.seed))
    leakage=policy.leakage_check(obs)
    rows={};begin=time.time();trace=[]
    from nav3.env import rotate
    for step in range(18002):
        action=policy(obs);state,obs,_=env.step(state,action)
        trace.append(np.column_stack([np.asarray(state[k]) for k in ['pos','yaw','s','gates','failure']]))
        done=np.asarray(obs['is_last'])
        if done.any():
            host=jax.device_get({k:state[k] for k in ['gates','failure','s','ep_return','steps','yaw']})
            for i in np.flatnonzero(done):
                if int(i) not in rows:
                    rows[int(i)]={k:float(v[i]) for k,v in host.items()}
                    xy=state['pos'][:,None]+rotate(env.foot[None],state['yaw'])
                    t,_=env.truth(state,xy)
                    rows[int(i)]['pos']=np.asarray(state['pos'][i]).tolist()
                    rows[int(i)]['foot_support_max']=float(t[i,:,2].max())
                    rows[int(i)]['foot_obstacle_min']=float(t[i,:,5].min())
        if len(rows)==a.episodes:break
    assert len(rows)==a.episodes,(len(rows),a.episodes)
    result=dict(scope='same canonical scene used for training; development evaluation only',checkpoint=a.checkpoint,episodes=a.episodes,seed=a.seed,
        success_rate=sum(r['gates']==8 and r['failure']==0 for r in rows.values())/a.episodes,
        mean_gates=float(np.mean([r['gates'] for r in rows.values()])),seconds=time.time()-begin,leakage=leakage,rows=rows)
    np.savez_compressed(Path(a.output).with_suffix('.npz'),trace=np.stack(trace))
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
