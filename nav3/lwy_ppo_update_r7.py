"""Unchanged PPO optimizer update with gradient and scale telemetry for LWY."""
import jax
import jax.numpy as jnp
import optax
from .ppo import loss
OPT=optax.chain(optax.clip_by_global_norm(.5),optax.adam(3e-5,eps=1e-5))


@jax.jit
def update(params,optimizer,batch):
    (_,metrics),grad=jax.value_and_grad(loss,has_aux=True)(params,batch)
    delta,optimizer=OPT.update(grad,optimizer,params)
    metrics={**metrics,'grad_norm':optax.global_norm(grad),
        'policy_std_mean':jnp.mean(jnp.exp(jnp.clip(params['logstd'],-3,1))),
        'return_mean':jnp.mean(batch['return']),'return_std':jnp.std(batch['return'])}
    return optax.apply_updates(params,delta),optimizer,metrics
