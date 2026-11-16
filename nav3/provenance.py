import hashlib
import json
from pathlib import Path


def fingerprint():
    h=hashlib.sha256()
    for path in ['nav3/env.py','nav3/bank.py','nav3/control.py','configs/env.json','configs/calibration.yaml','metrics/map_manifest.json']:
        h.update(path.encode());h.update(Path(path).read_bytes())
    return h.hexdigest()


def require_g1():
    gate=json.loads(Path('metrics/g1.json').read_text())
    assert gate['passed'] and gate['episodes']>=100,'G1 must pass before RL'
    assert gate.get('environment_fingerprint')==fingerprint(),'Environment changed since G1; rerun G1'
