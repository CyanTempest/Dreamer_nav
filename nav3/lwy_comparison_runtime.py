"""Explicit evaluation protocol; alternate runtime adoption requires evidence."""
import json
import os
from pathlib import Path


def level():
    path=Path('configs/lwy_v2_evaluation.json')
    value=json.loads(path.read_text())['autotune'] if path.exists() else 0
    assert value in [0,1]
    if value==1:
        decision=json.loads(Path('metrics/lwy_v2_evaluation_runtime_decision.json').read_text())
        assert decision['approved_by_user'] and decision['autotune']==1
        evidence=json.loads(Path('metrics/lwy_v2_runtime_candidate_protocol.json').read_text())
        assert evidence['passed'] and evidence['qualification']['original']['successes']==10
        assert evidence['qualification']['blind']['success_rate']>=.95
        from .lwy_eval_v2 import protocol_hash
        assert evidence['environment_sha256']==protocol_hash(), 'Maps or physical parameters changed since the candidate qualification'
    return value


def flags(value=None):
    value=level() if value is None else value
    return f'--xla_gpu_autotune_level={value} --xla_gpu_deterministic_ops=true'


def suffix():return '_rt1' if level()==1 else ''


def setup(value=None,environment=None):
    env=os.environ if environment is None else environment
    env['XLA_FLAGS']=flags(value);env['TF_CUDNN_DETERMINISTIC']='1'
