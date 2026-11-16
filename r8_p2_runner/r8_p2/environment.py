"""Canonical-map pilot. Deliberately excludes multi-map and dynamic obstacles."""
from functools import partial
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from nav3.r8_p1_online import SimpleEnv, Atlas, fitted_config
from nav3.lwy_bank import LwyBank

@jax.tree_util.register_pytree_node_class
class PilotEnv(SimpleEnv):
    def __init__(self,bank,n,training=False):
        cfg={**fitted_config(),**json.loads(Path('configs/lwy_v2_tracker.json').read_text()),'r8_random_starts':training}
        super().__init__(bank,n,atlas=Atlas(),config=cfg)
        self.reset=partial(jax.jit(PilotEnv._reset),self)
        self.step=partial(jax.jit(PilotEnv._step),self)
        self.observe=partial(jax.jit(PilotEnv._observe),self)

    def _reset(self,key,mids=None,start_s=None,observe=True):
        if start_s is None and self.cfg['r8_random_starts']:
            k1,k2=jax.random.split(jax.random.fold_in(key,8142))
            start_s=jnp.where(jax.random.uniform(k1,(self.n,))<.25,2.,jax.random.uniform(k2,(self.n,),minval=2.,maxval=900.))
        return super()._reset(key,mids,start_s,observe)

def make_env(n,training=False):
    bank=LwyBank('test',limit=1,root='maps_lwy')
    assert bank.names==['test_862428'],bank.names
    return PilotEnv(bank,n,training)
