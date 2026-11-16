"""Round-seven placement switch; historical v2 physics stays byte-identical."""
from functools import partial
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .lwy_safe_obstacles import SafeLwyEnv


@jax.tree_util.register_pytree_node_class
class LwyEnvR7(SafeLwyEnv):
    def __init__(self,bank,*args,**kwargs):
        super().__init__(bank,*args,**kwargs)
        rule=self.cfg.get('obstacle_rule','shoulder')
        assert rule in ['shoulder','legacy']
        if self.random_obstacles and rule=='shoulder':
            rows=[]
            for r in bank.records:
                directory=Path('maps_lwy_r7_obstacles/lwy')/r['name']
                audit=json.loads((directory/'audit.json').read_text())
                assert audit['map_sha256']==r['sha256'] and audit['minimum_witness_margin_m']>=0
                rows.append(np.load(directory/'catalog.npz')['obstacles'])
            self.placements=jnp.asarray(np.stack(rows))
        self.reset=partial(jax.jit(LwyEnvR7._reset),self)
