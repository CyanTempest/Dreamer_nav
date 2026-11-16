"""nav4.train_dreamer pieces that run without a GPU: GPU guard, XLA flags, obs space from the env, leakage check."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

from nav4 import train_dreamer
from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import load

TRAIN_ENV = json.loads((Path(train_dreamer.ROOT)/'nav4/config/train.json').read_text())['env']


def fake_smi(free, total=24564, apps=''):
    def query(args):
        return f'{free}, {total}\n' if args[0].startswith('--query-gpu') else apps
    return query


class Guard(unittest.TestCase):
    def test_available_counts_this_process(self):
        info = train_dreamer.gpu_memory(fake_smi(2000, apps='4242, 9000\n999, 22000\n'), pid=4242)
        self.assertEqual(info, dict(available_mib=11000, free_mib=2000, own_mib=9000, total_mib=24564))
        self.assertTrue(train_dreamer.guard_ok(info, 10000))
        self.assertFalse(train_dreamer.guard_ok(train_dreamer.gpu_memory(fake_smi(9999), pid=1), 10000))

    def test_unreadable_nvidia_smi_is_unavailable(self):
        def broken(args):
            raise FileNotFoundError('nvidia-smi')
        info = train_dreamer.gpu_memory(broken)
        self.assertIn('error', info)
        self.assertFalse(train_dreamer.guard_ok(info, 0))

    def test_refuses_to_start_with_exit_code_3(self):
        with tempfile.TemporaryDirectory() as logdir, patch.object(train_dreamer, 'nvidia_smi', fake_smi(500)):
            self.assertEqual(train_dreamer.main(['--logdir', logdir, '--seconds', '10']), train_dreamer.EXIT_GPU)
            rows = [json.loads(x) for x in (Path(logdir)/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'gpu_guard_refused')
            self.assertFalse((Path(logdir)/'ckpt').exists())

    def test_deterministic_xla_flags(self):
        flags = train_dreamer.deterministic_xla('--xla_gpu_autotune_level=0 --foo=1')
        self.assertEqual(flags.split(), ['--foo=1', '--xla_gpu_autotune_level=1', '--xla_gpu_deterministic_ops=true'])


class PolicySpace(unittest.TestCase):
    def test_obs_space_from_the_training_env_and_leakage_check(self):
        from nav4.policy import DreamerPolicy, spaces, spec
        env = TrailEnv(load({'m': synthetic.build('flat')}), 2, TRAIN_ENV)
        _, obs, _ = env.reset(jax.random.PRNGKey(0))
        with tempfile.TemporaryDirectory() as logdir:
            policy = DreamerPolicy(0, logdir, obs=obs, jax_overrides=dict(platform='cpu'))
        self.assertEqual(list(policy.obs_space), sorted(obs))          # canonical key order
        self.assertEqual(list(policy.agent.model.enc.obs_space), ['bev', 'state'])
        self.assertEqual(sorted(policy.agent.model.dec.obs_space),
                         ['bev', 'priv_progress', 'priv_route', 'priv_trav', 'state'])
        saved = spec(policy.obs_space)
        self.assertEqual(saved['priv_route'], ['uint8', [64, 64, 1]])
        self.assertEqual(saved['priv_progress'], ['float32', [4]])
        self.assertEqual(spec(spaces(json.loads(json.dumps(saved)))), saved)       # JSON round trip
        leak = policy.leakage_check(obs)
        self.assertEqual(leak['privileged_actor_carry_max_error'], 0.)
        self.assertEqual(leak['perturbed_keys'], ['priv_progress', 'priv_route', 'priv_trav'])   # jit sorts keys
        action = policy(obs)
        self.assertEqual(action.shape, (2, 3))
        self.assertTrue(np.isfinite(np.asarray(action)).all())


if __name__ == '__main__':
    unittest.main()
