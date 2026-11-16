"""Episode-weighted mixed collector, preserving each source's physical checker.

Each episode chooses LWY with probability .75. A persistent worker is not tied
to a map family: that would underweight the longer LWY episodes. Both banks are
dynamic JAX arguments; legacy atlas caches are required to exist, never created.
"""
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .bank import MapBank
from .lwy_safe_obstacles import SafeLegacyEnv,SafeLwyEnv
from .lwy_bank import LwyBank


def load_banks():
    manifest=json.loads(Path('maps/manifest.json').read_text())
    records=[r for r in manifest['maps'] if r['split']=='train']
    cache=Path(f'maps/atlas_train_{len(records)}')
    done=json.loads((cache/'done.json').read_text())
    assert done['signature']==[r['sha256'] for r in records], 'Refuse to create or update protected legacy atlas'
    return LwyBank('train',root='maps_lwy_v2'),MapBank('train')


class MixedEnv:
    def __init__(self,lwy,legacy,num_envs,probability=.75,random_obstacles=True,tracker_config=None):
        assert 0<probability<1
        self.n=num_envs;self.probability=probability
        self.lwy=SafeLwyEnv(lwy,1,random_obstacles,config=tracker_config)
        self.legacy=SafeLegacyEnv(legacy,1,random_obstacles)
        self.bank=SimpleNamespace(count=lwy.count+legacy.count,names=lwy.names+legacy.names,
            records=[dict(r,source_family='lwy') for r in lwy.records]+[dict(r,source_family='legacy') for r in legacy.records])
        self.lwy_count=lwy.count
        self.cfg={**self.lwy.cfg,'lwy_episode_probability':probability}
        self._reset_many=jax.jit(jax.vmap(partial(_reset_one,probability=probability),in_axes=(None,None,0,0)))
        self._step_many=jax.jit(partial(_step_many,probability=probability))

    def reset(self,key,mids=None):
        keys=jax.vmap(lambda i:jax.random.fold_in(key,i))(jnp.arange(self.n))
        mids=jnp.full(self.n,-1,jnp.int32) if mids is None else mids
        state,obs,extra=self._reset_many(self.lwy,self.legacy,keys,mids)
        return self._flatten_state(state),self._flatten(obs),self._flatten(extra)

    def step(self,state,action):
        nested={k:(v if k=='key' else v[:,None]) for k,v in state.items()}
        state,obs,extra=self._step_many(self.lwy,self.legacy,nested,action)
        return self._flatten_state(state),self._flatten(obs),self._flatten(extra)

    @staticmethod
    def _flatten(x):return {k:v[:,0] for k,v in x.items()}

    @staticmethod
    def _flatten_state(x):return {k:(v if k=='key' else v[:,0]) for k,v in x.items()}


def _reset_one(lwy,legacy,key,mid,probability):
    choose,which,episode=jax.random.split(key,3)
    source=jax.random.bernoulli(choose,probability)
    count=jnp.where(source,lwy.bank.count,legacy.bank.count)
    sampled=jax.random.randint(which,(),0,count)+jnp.where(source,0,lwy.bank.count)
    mid=jnp.where(mid<0,sampled,mid)
    new=lwy._reset(episode,jnp.clip(mid,0,lwy.bank.count-1)[None])
    old=legacy._reset(episode,jnp.clip(mid-lwy.bank.count,0,legacy.bank.count-1)[None])
    state,obs,extra=jax.tree.map(lambda a,b:jnp.where(mid<lwy.bank.count,a,b),new,old)
    state['mids']=mid[None]
    return state,obs,extra


def _advance_one(lwy,legacy,state,action):
    mid=state['mids'][0]
    new=lwy._step({**state,'mids':jnp.clip(state['mids'],0,lwy.bank.count-1)},action[None])
    old=legacy._step({**state,'mids':jnp.clip(state['mids']-lwy.bank.count,0,legacy.bank.count-1)},action[None])
    nxt,obs,extra=jax.tree.map(lambda a,b:jnp.where(mid<lwy.bank.count,a,b),new,old)
    nxt['mids']=state['mids']
    return nxt,obs,extra


def _step_many(lwy,legacy,state,action,probability):
    advanced=jax.vmap(_advance_one,in_axes=(None,None,0,0))(lwy,legacy,state,action)
    resetting=state['done'][:,0]
    def fresh(_):
        keys=jax.vmap(lambda key:jax.random.fold_in(key,81173))(state['key'])
        reset=jax.vmap(partial(_reset_one,probability=probability),in_axes=(None,None,0,0))(
            lwy,legacy,keys,jnp.full(len(keys),-1,jnp.int32))
        choose=lambda a,b:jnp.where(resetting.reshape((len(keys),)+(1,)*(a.ndim-1)),b,a)
        return jax.tree.map(choose,advanced,reset)
    # This condition is scalar and outside vmap. Branches never select or
    # broadcast the multi-gigabyte map atlases along the environment axis.
    return jax.lax.cond(jnp.any(resetting),fresh,lambda _:advanced,None)


class EpisodeMixMonitor:
    def __init__(self,lwy_count):
        self.boundary=lwy_count;self.counts=np.zeros(2,np.int64)
        self.map_counts={}

    def observe(self,state,obs):
        mids=np.asarray(state['mids'])[np.asarray(obs['is_first'])]
        self.counts+=np.array([(mids<self.boundary).sum(),(mids>=self.boundary).sum()])
        for mid in mids:self.map_counts[int(mid)]=self.map_counts.get(int(mid),0)+1

    def save(self):
        return dict(lwy_episodes=int(self.counts[0]),legacy_episodes=int(self.counts[1]),
            lwy_fraction=float(self.counts[0]/max(1,self.counts.sum())),map_episode_counts=self.map_counts)

    def load(self,data):
        self.counts=np.array([data['lwy_episodes'],data['legacy_episodes']],np.int64)
        self.map_counts={int(k):v for k,v in data['map_episode_counts'].items()}
