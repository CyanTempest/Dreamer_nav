"""S5: CPU throughput of TrailEnv + DreamerPolicy in the training input variant, and XLA memory analysis
(compile-time buffer sizes) of env step, policy step and train step for a larger env count.

  JAX_PLATFORMS=cpu $NAV3_PYTHON audit_20261116/train_dev/s5_throughput.py 64 40 256 > audit_20261116/train_dev/s5.json
"""
import json
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.policy import DreamerPolicy

cfg = json.load(open('nav4/config/train.json'))
envs, steps, mem_envs = (int(x) for x in sys.argv[1:4])
bank = TrailBank(cfg['manifest'], cfg['maps'])
env = TrailEnv(bank, envs, cfg['env'])
key = jax.random.PRNGKey(0)
state, obs, extra = env.reset(key)
policy = DreamerPolicy(0, tempfile.mkdtemp(prefix='s5_', dir='audit_20261116/train_dev'), obs=obs,
                       jax_overrides=dict(platform='cpu', autotune=1, deterministic=True))
keys = list(policy.obs_space)


def act(carry, obs, k):
    _, (carry, a, outs) = policy._policy(policy.agent.params, carry, {x: obs[x] for x in keys}, mode='train', seed=k)
    action = jnp.where(obs['is_last'][:, None], 0., a['action'])
    return (*carry[:3], {'action': action}), action, outs


random = jax.jit(lambda k: jax.random.uniform(k, (envs, 3), minval=-1, maxval=1))
carry = policy.initial(envs)
t = time.time()
for i in range(3):                                               # compile + warm up
    carry, action, outs = act(carry, obs, jax.random.fold_in(key, i))
    state, obs, extra = env.step(state, random(jax.random.fold_in(key, 50+i)))
jax.block_until_ready((obs, carry))
warmup = time.time()-t


def timed(fn, n):
    t = time.time()
    out = None
    for i in range(n):
        out = fn(i)
    jax.block_until_ready(out)
    return (time.time()-t)/n


s = [state]
env_only = timed(lambda i: s.__setitem__(0, env.step(s[0], random(jax.random.fold_in(key, 100+i)))[0]) or s[0]['pos'], steps)
c = [carry]
policy_only = timed(lambda i: c.__setitem__(0, act(c[0], obs, jax.random.fold_in(key, 200+i))[0]) or c[0][1]['deter'], steps)
loop = dict(state=state, obs=obs, carry=carry, action=action)


def driver(i):
    # As the trainer's Driver: env step with the previous action, policy, host copy of the transition.
    loop['state'], loop['obs'], _ = env.step(loop['state'], loop['action'])
    loop['carry'], loop['action'], outs = act(loop['carry'], loop['obs'], jax.random.fold_in(key, 300+i))
    jax.device_get(({k: loop['obs'][k] for k in keys}, outs, loop['action']))
    return loop['action']


combined = timed(driver, steps)


def analysis(compiled):
    m = compiled.memory_analysis()
    return {f'{k}_mib': getattr(m, f'{k}_size_in_bytes', 0)/2**20 for k in ('temp', 'argument', 'output', 'alias')}


memory = {}
for n in sorted({envs, mem_envs}):
    e = TrailEnv(bank, n, cfg['env'])
    shapes = jax.eval_shape(lambda k: TrailEnv._reset(e, k), key)
    step = jax.jit(TrailEnv._step).lower(e, shapes[0], jax.ShapeDtypeStruct((n, 3), jnp.float32)).compile()
    reset = jax.jit(TrailEnv._reset).lower(e, key).compile()
    obs_shapes = {k: shapes[1][k] for k in keys}
    pol = policy._policy.lower(policy.agent.params, policy.initial(n), obs_shapes, mode='train', seed=key).compile()
    memory[n] = dict(env_step=analysis(step), env_reset=analysis(reset), policy_step=analysis(pol),
                     env_state_mib=sum(np.prod(x.shape)*x.dtype.itemsize for x in jax.tree.leaves(shapes[0]))/2**20)
from embodied.jax import internal  # noqa: E402  (importable once nav4.policy put dreamerv3/ on sys.path)
agent = policy.agent
B, T, C = agent.config.batch_size, agent.config.batch_length, agent.config.replay_context
data = agent._zeros(agent.spaces, (B, T+C))
allo = {k: v for k, v in agent.params.items() if k in agent.policy_keys}
dona = {k: v for k, v in agent.params.items() if k not in agent.policy_keys}
t = time.time()
train = agent._train.lower(dona, allo, agent._seeds(0, agent.train_mirrored), agent.init_train(B),
                           internal.device_put(data, agent.train_sharded)).compile()
train_compile = time.time()-t
print(json.dumps(dict(
    envs=envs, steps=steps, device=str(jax.devices()[0]), warmup_s=warmup,
    env_step_s=env_only, policy_step_s=policy_only, driver_step_s=combined,
    env_steps_per_s=dict(env_only=envs/env_only, policy_only=envs/policy_only, env_plus_policy=envs/combined),
    obs_keys={k: [str(v.dtype), list(v.shape)] for k, v in obs.items()},
    bank_mib=sum(getattr(bank, k).nbytes for k in bank.KEYS)/2**20,
    agent_state_mib=sum(v.nbytes for v in agent.params.values())/2**20,
    agent_state_tensors=len(agent.params),
    train_step=dict(batch=[B, T+C], compile_s=train_compile, **analysis(train)),
    memory_by_envs=memory), indent=1))
