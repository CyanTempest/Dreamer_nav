"""The evaluation entry point constructs the agent before it knows the saved training shape."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import jax

from nav4 import evaluate
from nav4.env import TrailEnv
from nav4.bank import TrailBank
from nav4.policy import DreamerPolicy, make_config, spec
from nav4.tests import synthetic


class ModelConfig(unittest.TestCase):
    def test_legacy_defaults_and_invalid_presets(self):
        cfg = make_config(0, '.')
        self.assertEqual((cfg.batch_size, cfg.batch_length), (16, 64))
        with self.assertRaises(ValueError):
            make_config(0, '.', 'defaults')
        with self.assertRaises(ValueError):
            make_config(0, '.', 'size12m', 0)

    def test_evaluate_reads_model_and_batch_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = synthetic.write(tmp, {'flat': synthetic.build('flat')})
            env = TrailEnv(TrailBank(manifest), 1)
            _, obs, _ = env.reset(jax.random.PRNGKey(0))
            train = dict(model='size25m', batch_size=32, batch_length=32)
            policy = DreamerPolicy(0, root, jax_overrides=dict(platform='cpu'), obs=obs, **train)
            self.assertEqual(policy.agent.config.batch_size, 32)
            self.assertEqual(policy.agent.config.batch_length, 32)
            ckpt = root/'test.pkl'
            with ckpt.open('wb') as f:
                pickle.dump(dict(agent=policy.agent.save(), nav4=dict(
                    train_config=train, env_overrides={}, obs_space=spec(policy.obs_space))), f)
            del policy
            result = root/'result.json'
            evaluate.main(['--manifest', str(manifest), '--method', 'dreamer', '--action', 'mode',
                           '--checkpoint', str(ckpt), '--repeats', '1', '--max-steps', '1', '--output', str(result)])
            row = json.loads(result.read_text())
            self.assertEqual(row['loop_steps'], 1)
            self.assertEqual(row['leakage']['privileged_actor_carry_max_error'], 0)

