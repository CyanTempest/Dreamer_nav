"""Use the explicitly approved deterministic runtime for every LWY method."""
import json
import os
from pathlib import Path


def configure(method, environment=None):
    env = os.environ if environment is None else environment
    config = json.loads(Path('configs/lwy_v2_memory.json').read_text())
    env['XLA_PYTHON_CLIENT_ALLOCATOR'] = config.get('ppo_allocator', 'default') if method == 'ppo' else config.get('jax_allocator', 'default')
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(config['jax_pool_fraction'])
    from .lwy_comparison_runtime import setup
    setup(environment=env)
    return {key: env.get(key) for key in ['XLA_FLAGS', 'TF_CUDNN_DETERMINISTIC',
        'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_MEM_FRACTION']}
