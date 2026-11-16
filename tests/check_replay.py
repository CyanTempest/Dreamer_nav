"""Test temporal/worker alignment across chunk boundaries and disk restoration."""
import json
from pathlib import Path
import tempfile
import numpy as np
import embodied

with tempfile.TemporaryDirectory(prefix='replay_contract_',dir='tmp') as folder:
    replay=embodied.Replay(length=65,capacity=1000000,directory=folder,chunksize=32,save_wait=True,seed=12)
    for t in range(180):
        for worker in range(2):
            replay.add(dict(state=np.array([t,worker],np.float32),action=np.float32(t/200),
                reward=np.float32((t-1)/200),is_first=np.bool_(t%97==0),is_last=np.bool_(t%97==96),
                is_terminal=np.bool_(t%97==96)),worker)
    def check(buffer):
        batch=buffer.sample(16)
        t=batch['state'][...,0];worker=batch['state'][...,1]
        assert np.all(np.diff(t,axis=1)==1),'Temporal discontinuity at chunk boundary'
        assert np.all(np.diff(worker,axis=1)==0),'Workers mixed in one sequence'
        np.testing.assert_allclose(batch['action'],t/200)
        np.testing.assert_allclose(batch['reward'],(t-1)/200)
        assert batch['is_first'][:,0].all()
    check(replay);replay.save()
    restored=embodied.Replay(length=65,capacity=1000000,directory=folder,chunksize=32,save_wait=True,seed=13)
    restored.load();check(restored)
result=dict(passed=True,sequence_length=65,workers=2,chunk_length=32,disk_restore=True,transition_alignment='o_t,r_t,a_t')
Path('metrics/replay_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
