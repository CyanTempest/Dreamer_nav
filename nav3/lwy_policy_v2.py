"""Single-environment vmap policy inference with independent episode PRNGs."""
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np


class EpisodePolicy:
    def __init__(self,method,seeds,checkpoint=None,autotune=None):
        if autotune is None:
            from .lwy_comparison_runtime import level
            autotune=level()
        self.method=method;self.head=method.startswith('dreamer')
        self.actual_n=len(seeds)
        # Keep convolution/GEMM shapes fixed across 40-episode validation,
        # 100-episode final evaluation and 1-episode video recording. Dummy
        # policy streams never enter the physical environment or metrics.
        assert self.actual_n<=100
        seeds=np.r_[seeds,2000000000+np.arange(100-self.actual_n)]
        self.keys=jnp.stack([jax.random.PRNGKey(int(s)+1000000000) for s in seeds])
        self.n=len(seeds);self.carry=None;self.last_prediction=None
        if method=='planner':return
        path=Path(checkpoint or f'checkpoints/{method}/latest.pkl')
        with path.open('rb') as f:data=pickle.load(f)
        if self.head:
            from .dreamer_adapter import DreamerPolicy
            self.adapter=DreamerPolicy(0,'logs/lwy_v2_eval_agent',jax_overrides=dict(autotune=autotune,deterministic=True))
            self.adapter.agent.load(data['agent'])
            self.params=self.adapter.agent.params
            single=self.adapter.initial(1)
            self.carry=jax.tree.map(lambda x:jnp.repeat(x[None],self.n,axis=0),single)
            self.infer=jax.jit(jax.vmap(lambda params,c,o,k:self.adapter._policy(params,c,o,mode='train',seed=k)[1],in_axes=(None,0,0,0)))
            self.decode=jax.jit(jax.vmap(lambda params,c,f,k:self.adapter._decode(params,c,f,seed=k)[1],in_axes=(None,0,0,0)))
        else:
            from .ppo import act
            self.params=jax.tree.map(jnp.asarray,data['policy'][0])
            self.carry=jnp.zeros((self.n,1,256))
            self.infer=jax.jit(jax.vmap(act,in_axes=(None,0,0,0)))
        del data
        from .memory import release_host_memory
        release_host_memory()

    def __call__(self,obs,planner,predict=False):
        if self.method=='planner':return planner({k:v[:,0] for k,v in obs.items()})
        obs={k:jnp.pad(v,[(0,self.n-self.actual_n)]+[(0,0)]*(v.ndim-1)) for k,v in obs.items()}
        pair=jax.vmap(lambda key:jax.random.split(key))(self.keys)
        self.keys,key=pair[:,0],pair[:,1]
        if self.head:
            public={k:obs[k] for k in self.adapter.obs_space}
            self.carry,act,_=self.infer(self.params,self.carry,public,key)
            action=jnp.where(obs['is_last'][...,None],0.,act['action'])
            self.carry=(*self.carry[:3],{'action':action})
            if predict:self.last_prediction=self.decode(self.params,self.carry,obs['is_first'],key)[:self.actual_n,0,...,0]
        else:
            self.carry,action,_,_,_=self.infer(self.params,self.carry,obs,key)
        return action[:self.actual_n,0]

    def executed(self,action):
        if self.head:
            padded=jnp.pad(action,[(0,self.n-self.actual_n),(0,0)])
            self.carry=(*self.carry[:3],{'action':padded[:,None]})
