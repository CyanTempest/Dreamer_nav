"""Official Dreamer Agent, with a device-batched policy adapter and decoder readout.

The update, optimizer, replay and distributions are upstream implementations.
Patched vs upstream e01491f: agent.py (priv_ keys excluded from the encoder), and replay storage in
embodied/core chunk.py, replay.py plus new compress.py (LZ4 in-RAM or mmap chunks, replay-v1 save/restore).
"""
from pathlib import Path
import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
import elements
import ninjax as nj
import ruamel.yaml as yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dreamerv3'))
from dreamerv3.agent import Agent
import embodied


def make_config(seed,logdir,head=True):
    cfg=yaml.YAML(typ='safe').load((ROOT/'dreamerv3/dreamerv3/configs.yaml').read_text())
    config=elements.Config(cfg['defaults']).update(cfg['size12m'])
    config=config.update(seed=seed,logdir=str(logdir),**{
        'replay.size':1000000,'replay.chunksize':64,
        'jax.prealloc':False,'jax.platform':'cuda','agent.report':False})
    return config


class DreamerPolicy:
    def __init__(self,seed=0,logdir='checkpoints/dreamer',head=True,amnesic=False,jax_overrides=None,deterministic=False):
        cfg=make_config(seed,logdir,head)
        self.cfg=cfg;self.head=head;self.amnesic=amnesic
        obs_space=dict(bev=elements.Space(np.uint8,(64,64,6)),state=elements.Space(np.float32,(13,)),
            reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        if head:
            obs_space['priv_trav']=elements.Space(np.uint8,(64,64,1))
        act_space=dict(action=elements.Space(np.float32,(3,),-1,1))
        jaxcfg=dict(cfg.jax)
        # The batched driver creates JAX arrays and reads metrics on the host.
        # Upstream's implicit-transfer guard assumes its own host-only driver.
        jaxcfg.update(precompile=False,profiler=False,transfer_guard=False)
        if os.environ.get('NAV3_REPRO')=='1':jaxcfg.update(autotune=1,deterministic=True)
        if jax_overrides:jaxcfg.update(jax_overrides)
        self.agent=Agent(obs_space,act_space,elements.Config(**cfg.agent,
            logdir=str(logdir),seed=seed,jax=jaxcfg,batch_size=16,batch_length=64,
            replay_context=1,report_length=32,replica=0,replicas=1))
        self.obs_space=obs_space
        self.key=jax.random.PRNGKey(seed+1234)
        self.carry=None
        self._initial=jax.jit(nj.pure(self.agent.model.init_policy),static_argnums=(1,))
        def mode_policy(carry,obs,mode='train'):
            # Agent.policy with act=pol(...).pred() (tanh(mean) of bounded_normal) instead of sample().
            # That sample() is its last nj.seed() draw, so the RSSM latent draws are unchanged.
            m=self.agent.model;enc_carry,dyn_carry,dec_carry,prevact=carry
            kw=dict(training=False,single=True);reset=obs['is_first']
            enc_carry,enc_entry,tokens=m.enc(enc_carry,obs,reset,**kw)
            dyn_carry,dyn_entry,feat=m.dyn.observe(dyn_carry,tokens,prevact,reset,**kw)
            dec_entry={}
            if dec_carry:dec_carry,dec_entry,_=m.dec(dec_carry,feat,reset,**kw)
            act=jax.tree.map(lambda d:d.pred(),m.pol(m.feat2tensor(feat),bdims=1))
            outs={'finite':elements.tree.flatdict(jax.tree.map(lambda x:jnp.isfinite(x).all(range(1,x.ndim)),
                dict(obs=obs,carry=carry,tokens=tokens,feat=feat,act=act)))}
            if m.config.replay_context:outs.update(elements.tree.flatdict(dict(enc=enc_entry,dyn=dyn_entry,dec=dec_entry)))
            return (enc_carry,dyn_carry,dec_carry,act),act,outs
        self._policy=jax.jit(nj.pure(mode_policy if deterministic else self.agent.model.policy),static_argnames=('mode',))
        def readout(carry,first):
            _,_,outs=self.agent.model.dec(carry[2],carry[1],first,training=False,single=True)
            return outs['priv_trav'].pred()
        self._decode=jax.jit(nj.pure(readout))
        self.last_prediction=None

    def initial(self,n):
        _,carry=self._initial(self.agent.params,n,seed=jax.random.PRNGKey(0))
        return carry

    def policy(self,obs,action_override=None,predict=False):
        obs={k:obs[k] for k in self.obs_space}
        n=len(obs['is_first'])
        if self.carry is None or self.amnesic:
            self.carry=self.initial(n)
        self.key,key=jax.random.split(self.key)
        # Upstream Agent.policy ignores mode and always samples; deterministic=True uses pred() instead.
        _,(carry,act,outs)=self._policy(self.agent.params,self.carry,obs,mode='train',seed=key)
        action=act['action'] if action_override is None else action_override
        action=jnp.where(obs['is_last'][:,None],0.,action)
        self.carry=(*carry[:3],{'action':action})
        outs={k:v.astype(self.agent.spaces[k].dtype) for k,v in outs.items() if k!='finite'}
        if predict and self.head:
            _,self.last_prediction=self._decode(self.agent.params,self.carry,obs['is_first'],seed=key)
        return action,outs

    def __call__(self,obs):
        action,_=self.policy(obs)
        return action

    def decoder(self):
        _,pred=self._decode(self.agent.params,self.carry,jnp.zeros(len(self.carry[1]['deter']),bool),seed=self.key)
        return pred

    def leakage_check(self,obs):
        if not self.head:
            return {}
        carry=self.initial(len(obs['is_first']))
        params=self.agent.params
        obs={k:obs[k] for k in self.obs_space}
        _,a=self._policy(params,carry,obs,mode='train',seed=jax.random.PRNGKey(991))
        _,b=self._policy(params,carry,{**obs,'priv_trav':255-obs['priv_trav']},mode='train',seed=jax.random.PRNGKey(991))
        error=max(float(jnp.max(jnp.abs(x-y))) for x,y in zip(jax.tree.leaves(a[0]),jax.tree.leaves(b[0])))
        assert error==0, error
        return dict(privileged_actor_carry_max_error=error,encoder_keys=list(self.agent.model.enc.obs_space),decoder_keys=list(self.agent.model.dec.obs_space))
