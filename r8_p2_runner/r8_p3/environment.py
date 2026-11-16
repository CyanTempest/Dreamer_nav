"""Versioned finite-map arrival task; original geometry and fall checks preserved."""
from functools import partial
import math
import jax
import jax.numpy as jnp
from nav3.lwy_bank import LwyBank
from nav3.r8_p1_online import SimpleEnv
from r8_p2.environment import PilotEnv

# Envelope circumscribed radius + raster/footprint conservatism + one policy step.
# Chosen from physical/configuration bounds, not failure positions or success rates.
RASTER_PAD=.05277048508933442
FINISH_MARGIN=math.ceil((math.hypot(1.03/2,.58/2)+RASTER_PAD+.8*.2)/.05)*.05

@jax.tree_util.register_pytree_node_class
class GoalBank(LwyBank):
    def route_at(self,mids,s):
        mi=mids.reshape((len(mids),)+(1,)*(s.ndim-1))
        return super().route_at(mids,jnp.minimum(s,self.gates[mi,-1]))

@jax.tree_util.register_pytree_node_class
class FinishEnv(PilotEnv):
    def __init__(self,bank,n,training=False):
        super().__init__(bank,n,training)
        self.reset=partial(jax.jit(FinishEnv._reset),self)
        self.step=partial(jax.jit(FinishEnv._step),self)
        self.observe=partial(jax.jit(FinishEnv._observe),self)

    def _reset(self,key,mids=None,start_s=None,observe=True):
        if start_s is None and self.cfg['r8_random_starts']:
            choose,uniform,tail=jax.random.split(jax.random.fold_in(key,8314),3)
            u=jax.random.uniform(choose,(self.n,))
            start_s=jnp.where(u<.25,2.,jnp.where(u<.80,
                jax.random.uniform(uniform,(self.n,),minval=2.,maxval=900.),
                jax.random.uniform(tail,(self.n,),minval=970.,maxval=995.)))
        return super()._reset(key,mids,start_s,observe)

def make_env(n,training=False,clamp_goals=True):
    bank=(GoalBank if clamp_goals else LwyBank)('test',limit=1,root='maps_lwy')
    assert bank.names==['test_862428']
    length=bank.arcs[0,bank.lengths[0]-1]
    bank.gates=bank.gates.at[:,-1].set(length-FINISH_MARGIN)
    return FinishEnv(bank,n,training)
