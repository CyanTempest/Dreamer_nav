"""Uninstrumented original evaluator repeat, kept separate from old results."""
import json
from pathlib import Path
import jax
from nav3.diag_rollout import load_policy
from nav3.evaluate import evaluate
p=Path('metrics/diag_ppo_original_evaluator_repeat.json')
if p.exists():raise FileExistsError(p)
policy=load_policy('ppo_s0');policy.carry=None;policy.key=jax.random.PRNGKey(9137)
result=evaluate(policy,'test',30,False,False,method='diag_ppo_repeat',save=False)
p.write_text(json.dumps(result,indent=2))
