"""DreamerV3 policy wrapper (copy of nav3/dreamer_adapter.py incl. `deterministic`) and checkpoint loading.

dreamerv3/ is used unmodified; priv_* keys are excluded from its encoder (decoder target only).
deterministic=True acts with pol(...).pred() = tanh(mean) instead of a sample; RSSM latents are still sampled.
The observation space is the legacy one (bev, state, priv_trav when head) unless built from an env observation
dict or a saved spec (`spaces`); nav4.train_dreamer checkpoints carry that spec in payload['nav4']['obs_space'].
"""
import os
import pickle
import sys
from pathlib import Path

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'dreamerv3'))
from dreamerv3.agent import Agent  # noqa: E402


def spaces(obs, keys=None):
    """elements.Space per key from a batched env observation dict (leading env axis) or a saved spec
    {key: [dtype name, shape]} (`spec`). Keys are sorted (the order of jitted env outputs), since the decoder
    splits its image channels in key order; `keys` selects and orders a subset explicitly."""
    out = {}
    for k in (sorted(obs) if keys is None else keys):
        v = obs[k]
        dtype, shape = (v[0], tuple(v[1])) if isinstance(v, (list, tuple)) else (v.dtype, tuple(v.shape[1:]))
        out[k] = elements.Space(np.dtype(dtype), shape)
    return out


def spec(obs_space):
    """JSON/pickle-safe {key: [dtype name, shape]} of an obs space (inverse of `spaces`)."""
    return {k: [np.dtype(s.dtype).name, list(s.shape)] for k, s in obs_space.items()}


def read_checkpoint(path):
    """Checkpoint payload dict (latest.pkl): 'agent' plus, for nav4.train_dreamer checkpoints, 'nav4'."""
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def make_config(seed, logdir):
    cfg = yaml.YAML(typ='safe').load((ROOT/'dreamerv3/dreamerv3/configs.yaml').read_text())
    config = elements.Config(cfg['defaults']).update(cfg['size12m'])
    return config.update(seed=seed, logdir=str(logdir), **{
        'replay.size': 1000000, 'replay.chunksize': 64, 'jax.prealloc': False, 'jax.platform': 'cuda',
        'agent.report': False})


class DreamerPolicy:
    def __init__(self, seed=0, logdir='checkpoints/dreamer', head=True, amnesic=False, jax_overrides=None,
                 deterministic=False, obs=None, keys=None):
        """obs/keys: build the observation space with `spaces(obs, keys)` (env observation dict or saved spec)
        instead of the legacy one; head is then whether it contains priv_trav."""
        cfg = make_config(seed, logdir)
        if obs is None:
            obs_space = dict(bev=elements.Space(np.uint8, (64, 64, 6)), state=elements.Space(np.float32, (13,)),
                             reward=elements.Space(np.float32), is_first=elements.Space(bool),
                             is_last=elements.Space(bool), is_terminal=elements.Space(bool))
            if head:
                obs_space['priv_trav'] = elements.Space(np.uint8, (64, 64, 1))
        else:
            obs_space = spaces(obs, keys)
            head = 'priv_trav' in obs_space
        self.cfg, self.head, self.amnesic = cfg, head, amnesic
        act_space = dict(action=elements.Space(np.float32, (3,), -1, 1))
        jaxcfg = dict(cfg.jax)
        # The batched driver creates JAX arrays and reads metrics on the host.
        jaxcfg.update(precompile=False, profiler=False, transfer_guard=False)
        if os.environ.get('NAV3_REPRO') == '1':
            jaxcfg.update(autotune=1, deterministic=True)
        if jax_overrides:
            jaxcfg.update(jax_overrides)
        self.agent = Agent(obs_space, act_space, elements.Config(
            **cfg.agent, logdir=str(logdir), seed=seed, jax=jaxcfg, batch_size=16, batch_length=64,
            replay_context=1, report_length=32, replica=0, replicas=1))
        self.obs_space = obs_space
        self.key = jax.random.PRNGKey(seed+1234)
        self.carry = None
        self._initial = jax.jit(nj.pure(self.agent.model.init_policy), static_argnums=(1,))

        def mode_policy(carry, obs, mode='train'):
            # Agent.policy with act=pol(...).pred() instead of sample(); sample() is its last nj.seed() draw,
            # so the RSSM latent draws are unchanged.
            m = self.agent.model
            enc_carry, dyn_carry, dec_carry, prevact = carry
            kw = dict(training=False, single=True)
            reset = obs['is_first']
            enc_carry, enc_entry, tokens = m.enc(enc_carry, obs, reset, **kw)
            dyn_carry, dyn_entry, feat = m.dyn.observe(dyn_carry, tokens, prevact, reset, **kw)
            dec_entry = {}
            if dec_carry:
                dec_carry, dec_entry, _ = m.dec(dec_carry, feat, reset, **kw)
            act = jax.tree.map(lambda d: d.pred(), m.pol(m.feat2tensor(feat), bdims=1))
            outs = {'finite': elements.tree.flatdict(jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)), dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))}
            if m.config.replay_context:
                outs.update(elements.tree.flatdict(dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
            return (enc_carry, dyn_carry, dec_carry, act), act, outs
        self._mode = jax.jit(nj.pure(mode_policy), static_argnames=('mode',))
        self._policy = self._mode if deterministic else jax.jit(nj.pure(self.agent.model.policy), static_argnames=('mode',))

        def readout(carry, first):
            _, _, outs = self.agent.model.dec(carry[2], carry[1], first, training=False, single=True)
            return outs['priv_trav'].pred()
        self._decode = jax.jit(nj.pure(readout))
        self.last_prediction = None

    def load(self, path):
        """Load agent parameters from a checkpoint path or an already read payload (`read_checkpoint`)."""
        payload = path if isinstance(path, dict) else read_checkpoint(path)
        self.agent.load(payload['agent'])
        return self

    def initial(self, n):
        _, carry = self._initial(self.agent.params, n, seed=jax.random.PRNGKey(0))
        return carry

    def policy(self, obs, action_override=None, predict=False):
        obs = {k: obs[k] for k in self.obs_space}
        n = len(obs['is_first'])
        if self.carry is None or self.amnesic:
            self.carry = self.initial(n)
        self.key, key = jax.random.split(self.key)
        _, (carry, act, outs) = self._policy(self.agent.params, self.carry, obs, mode='train', seed=key)
        action = act['action'] if action_override is None else action_override
        action = jnp.where(obs['is_last'][:, None], 0., action)
        self.carry = (*carry[:3], {'action': action})
        outs = {k: v.astype(self.agent.spaces[k].dtype) for k, v in outs.items() if k != 'finite'}
        if predict and self.head:
            _, self.last_prediction = self._decode(self.agent.params, self.carry, obs['is_first'], seed=key)
        return action, outs

    def __call__(self, obs):
        return self.policy(obs)[0]

    def executed(self, action):
        """Record the action actually executed (after takeover) as the RSSM's previous action."""
        self.carry = (*self.carry[:3], {'action': action})

    def decoder(self):
        _, pred = self._decode(self.agent.params, self.carry, jnp.zeros(len(self.carry[1]['deter']), bool), seed=self.key)
        return pred

    def leakage_check(self, obs):
        """Perturbing every priv_* key (uint8 255-x, bool not x, float x+1) must not change the actor carry or
        action: they are decoder-only targets."""
        priv = [k for k in self.obs_space if k.startswith('priv_')]
        if not priv:
            return {}
        carry = self.initial(len(obs['is_first']))
        obs = {k: obs[k] for k in self.obs_space}

        def perturb(x):
            return 255-x if x.dtype == jnp.uint8 else (~x if x.dtype == jnp.bool_ else x+1)
        changed = {k: perturb(obs[k]) for k in priv}
        assert all(bool(jnp.any(changed[k] != obs[k])) for k in priv), 'perturbation left a priv key unchanged'
        _, a = self._policy(self.agent.params, carry, obs, mode='train', seed=jax.random.PRNGKey(991))
        _, b = self._policy(self.agent.params, carry, {**obs, **changed}, mode='train', seed=jax.random.PRNGKey(991))
        error = max(float(jnp.max(jnp.abs(x-y))) for x, y in zip(jax.tree.leaves(a[0]), jax.tree.leaves(b[0])))
        assert error == 0, error
        return dict(privileged_actor_carry_max_error=error, perturbed_keys=priv,
                    encoder_keys=list(self.agent.model.enc.obs_space), decoder_keys=list(self.agent.model.dec.obs_space))
