"""Full-route learning-efficiency measurements on the identical heldout set."""
import json
from pathlib import Path
from .evaluate import evaluate


def measure(policy,name,steps,wall_seconds,direct=False):
    carry,key=policy.carry,policy.key
    policy.carry=None
    result=evaluate(policy,'heldout',5,False,False,method=f'{name}_at_{steps}',direct=direct)
    policy.carry,policy.key=carry,key
    record=dict(method=name,steps=int(steps),wall_seconds=float(wall_seconds),mean_gates=result['mean_gates'],
        success_rate=result['success_rate'],episodes=result['episodes'],
        source=f'eval_{name}_at_{steps}_heldout_ob0_sf0.json',environment_fingerprint=result['environment_fingerprint'])
    with (Path('metrics')/f'efficiency_{name}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return record
