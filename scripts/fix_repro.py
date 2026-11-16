"""Fresh-process legacy evaluator reproduction; never overwrites old metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--name', default='ppo_s0')
p.add_argument('--warm', action='store_true')
p.add_argument('--deterministic', action='store_true')
p.add_argument('--autotune-only', action='store_true')
p.add_argument('--ops-only', action='store_true')
p.add_argument('--compare')
a = p.parse_args()
out = Path(f'metrics/fix_repro_{a.tag}.json')
assert not out.exists(), out
if a.deterministic:
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
elif a.autotune_only:
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
elif a.ops_only:
    os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import jax
from nav3.diag_rollout import load_policy
from nav3.evaluate import evaluate

policy = load_policy(a.name)
if a.warm:
    for obstacles, filtered in [(False, False), (False, True), (True, False), (True, True)]:
        policy.carry = None
        policy.key = jax.random.PRNGKey(9137)
        evaluate(policy, 'heldout', 5, obstacles, filtered, method='fix_repro_warm', save=False)
policy.carry = None
policy.key = jax.random.PRNGKey(9137)
result = evaluate(policy, 'test', 30, False, False, method=a.name, save=False)
result['runtime'] = dict(jax=jax.__version__, flags=os.environ.get('XLA_FLAGS'),
    checkpoint_sha256=hashlib.sha256(Path(f'checkpoints/{a.name}/latest.pkl').read_bytes()).hexdigest(),
    policy_seed=9137, environment_seed=9137, batch_size=30, warm=a.warm)
with out.open('x') as f:
    json.dump(result, f, indent=2)
if a.compare:
    old = json.loads(Path(a.compare).read_text())
    changed = [i for i, (x, y) in enumerate(zip(old['rows'], result['rows'])) if x != y]
    print('REPRO', json.dumps(dict(reference_success=old['success_rate'], success=result['success_rate'], changed_episodes=changed)), flush=True)
    assert not changed, f'Per-episode evaluation mismatch: {len(changed)}/30'
