"""Select from jointly checked placement catalogs at episode reset only."""
from functools import partial
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .env import FastEnv
from .lwy_env_v2 import LwyEnvV2


def catalog(bank,family):
    records=[]
    for r in bank.records:
        directory=Path('maps_lwy_v2_obstacles')/family/r['name']
        audit=json.loads((directory/'audit.json').read_text())
        assert audit['map_sha256']==r['sha256'] and audit['minimum_witness_margin_m']>=0
        records.append(np.load(directory/'catalog.npz')['obstacles'])
    return jnp.asarray(np.stack(records))


class PlacementMixin:
    def _reset(self,key,mids=None,start_s=None,observe=True):
        if not self.random_obstacles:return super()._reset(key,mids,start_s,observe)
        state=super()._reset(key,mids,start_s,observe=False)
        choice=jax.random.randint(jax.random.fold_in(key,774201),(self.n,),0,self.placements.shape[1])
        state['obstacles']=self.placements[state['mids'],choice]
        if not observe:return state
        state,obs,extra=self._observe(state)
        state['obs_queue']=jnp.repeat(obs['bev'][:,None],2,axis=1)
        return state,obs,extra


@jax.tree_util.register_pytree_node_class
class SafeLwyEnv(PlacementMixin,LwyEnvV2):
    ARRAY_KEYS=LwyEnvV2.ARRAY_KEYS+('placements',)
    def __init__(self,bank,*args,**kwargs):
        super().__init__(bank,*args,**kwargs)
        self.placements=catalog(bank,'lwy') if self.random_obstacles else jnp.zeros((bank.count,1,3,6))
        self.reset=partial(jax.jit(SafeLwyEnv._reset),self)


@jax.tree_util.register_pytree_node_class
class SafeLegacyEnv(PlacementMixin,FastEnv):
    ARRAY_KEYS=FastEnv.ARRAY_KEYS+('placements',)
    def __init__(self,bank,*args,**kwargs):
        super().__init__(bank,*args,**kwargs)
        self.placements=catalog(bank,'legacy') if self.random_obstacles else jnp.zeros((bank.count,1,3,6))
        self.reset=partial(jax.jit(SafeLegacyEnv._reset),self)
