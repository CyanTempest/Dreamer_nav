"""Check whether the training-kernel issue affects fixed-100 policy inference."""
import argparse,json,os,pickle
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--autotune',type=int,required=True);p.add_argument('--output');a=p.parse_args()
    os.environ['XLA_FLAGS']=f'--xla_gpu_autotune_level={a.autotune} --xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .dreamer_adapter import DreamerPolicy
    policy=DreamerPolicy(jax_overrides=dict(autotune=a.autotune,deterministic=True))
    with Path('checkpoints/dreamer_s0/latest.pkl').open('rb') as f:data=pickle.load(f)
    policy.agent.load(data['agent']);del data
    batch=dict(np.load('diagnostics/lwy_v2_health_old_batch.npz'))
    carry=jax.tree.map(lambda x:jnp.repeat(x[None],100,0),policy.initial(1))
    keys=jnp.stack([jax.random.PRNGKey(1000940000+i) for i in range(100)])
    infer=jax.jit(jax.vmap(lambda params,c,o,k:policy._policy(params,c,o,mode='train',seed=k)[1],in_axes=(None,0,0,0)))
    decode=jax.jit(jax.vmap(lambda params,c,f,k:policy._decode(params,c,f,seed=k)[1],in_axes=(None,0,0,0)))
    actions=[]
    for t in range(20):
        obs={k:jnp.asarray(batch[k][np.arange(100)%16,t,None]) for k in policy.obs_space}
        pair=jax.vmap(jax.random.split)(keys);keys,key=pair[:,0],pair[:,1]
        carry,act,_=infer(policy.agent.params,carry,obs,key)
        action=jnp.where(obs['is_last'][...,None],0.,act['action'])
        carry=(*carry[:3],{'action':action});actions.append(np.asarray(action))
    pred=np.asarray(decode(policy.agent.params,carry,obs['is_first'],key),dtype=np.float32)
    path=Path(a.output or f'diagnostics/lwy_v2_forward_autotune{a.autotune}.npz');assert not path.exists()
    np.savez_compressed(path,actions=np.array(actions),prediction=pred)
    print(json.dumps(dict(autotune=a.autotune,steps=20,streams=100,finite=bool(np.isfinite(pred).all()))),flush=True)


if __name__=='__main__':main()
