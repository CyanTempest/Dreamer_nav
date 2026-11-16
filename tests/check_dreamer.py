"""Verify the actual size12m graph and privileged-key noninterference before RL."""
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from nav3.dreamer_adapter import DreamerPolicy

policy=DreamerPolicy(logdir='checkpoints/contract_model')
obs={k:jnp.zeros((2,*s.shape),s.dtype) for k,s in policy.obs_space.items()}
obs['is_first']=jnp.ones(2,bool)
check=policy.leakage_check(obs)
action,outs=policy.policy(obs,predict=True)
assert action.shape==(2,3)
assert policy.last_prediction.shape==(2,64,64,1)
assert np.isfinite(np.asarray(action)).all()
assert 'priv_trav' not in check['encoder_keys'] and 'priv_trav' in check['decoder_keys']
check.update(size='size12m',deter=policy.agent.model.dyn.deter,
             parameter_count=sum(x.size for k,x in policy.agent.params.items() if not k.startswith('opt/')),
             decoder_prediction_shape=list(policy.last_prediction.shape),
             output_keys=list(outs),passed=True)
Path('metrics/dreamer_contract.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
