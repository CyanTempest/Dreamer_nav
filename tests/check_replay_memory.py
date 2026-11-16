"""Regression: restored tails are freed; detailed snapshots preserve streams."""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import tempfile
import numpy as np
import elements
from embodied import Replay


def add(replay, start, stop):
    for t in range(start, stop):
        for worker in range(32):
            replay.add(dict(state=np.array([t, worker], np.float32)), worker)


def aligned(replay):
    for _ in range(8):
        state = replay.sample(16)['state']
        assert (np.diff(state[..., 0], axis=1) == 1).all()
        assert (np.diff(state[..., 1], axis=1) == 0).all()


with tempfile.TemporaryDirectory(dir='tmp') as folder:
    original = Replay(65, 4096, folder, chunksize=64, save_wait=True)
    add(original, 0, 256)
    snapshot = original.save(detailed=True)
    restored = Replay(65, 4096, folder, chunksize=16, save_wait=True)
    restored.load(snapshot)
    assert len(restored) == len(original)
    assert restored.current == original.current
    assert dict(restored.streams) == dict(original.streams)
    assert restored.refs == original.refs
    aligned(restored)
    add(restored, 256, 768)
    aligned(restored)
    restored.save()
    legacy = Replay(65, 4096, folder, chunksize=16, save_wait=True)
    legacy.load()
    aligned(legacy)
    # File mapping is a lossless storage backend, including in-place recurrent
    # state updates. Compare actual samples against the unchanged RAM backend.
    mmap = Replay(65, 512, folder, chunksize=64, seed=17, save_wait=True,
                  mmap_directory=Path(folder)/'buffers')
    ram = Replay(65, 512, chunksize=64, seed=17)
    for t in range(160):
        for worker in range(4):
            step=dict(state=np.array([t,worker],np.float32),
                      deter=np.arange(128,dtype=np.float32)*np.float32(t/71))
            ram.add(step,worker);mmap.add(step,worker)
    for _ in range(32):
        one,two=ram.sample(16),mmap.sample(16)
        np.testing.assert_array_equal(one['state'],two['state'])
        np.testing.assert_array_equal(one['deter'],two['deter'])
        for replay,batch in [(ram,one),(mmap,two)]:
            replay.update(dict(stepid=batch['stepid'],deter=batch['deter']+np.float32(.125)))
    snapshot=mmap.save(detailed=True)
    remap=Replay(65,512,folder,chunksize=64,save_wait=True,
                 mmap_directory=Path(folder)/'buffers')
    remap.load(snapshot);aligned(remap)
    for u,c in mmap.chunks.items():
        if c.data:
            for key in c.data:
                np.testing.assert_array_equal(c.data[key][:c.length],remap.chunks[u].data[key])
    del remap,mmap,ram
    ram=Replay(65,512,chunksize=32,seed=51)
    packed=Replay(65,512,folder,chunksize=32,seed=51,compress=True,save_wait=True)
    rng=np.random.default_rng(7)
    for t in range(160):
        for worker in range(4):
            step=dict(state=np.array([t,worker],np.float32),
                      deter=np.arange(2048,dtype=np.float32)*np.float32(t/71),
                      bev=rng.integers(0,256,(32,32,2),dtype=np.uint8))
            ram.add(step,worker);packed.add(step,worker)
    for _ in range(16):
        one,two=ram.sample(16),packed.sample(16)
        for key in ['state','deter','bev']:
            np.testing.assert_array_equal(one[key],two[key])
        for replay,batch in [(ram,one),(packed,two)]:
            replay.update(dict(stepid=batch['stepid'],deter=batch['deter'].astype(np.float64)+.123456789))
    snapshot=packed.save(detailed=True)
    restored=Replay(65,512,folder,chunksize=32,compress=True,save_wait=True)
    restored.load(snapshot);aligned(restored)
    for u,c in packed.chunks.items():
        if c.length:
            for key,value in c.slice(0,c.length).items():
                np.testing.assert_array_equal(value,restored.chunks[u].slice(0,c.length)[key])
    legacy_snapshot={**snapshot,
        'items':[(elements.UUID(u),i) for u,i in snapshot['items']],
        'current':{k:(elements.UUID(u),i) for k,(u,i) in snapshot['current'].items()},
        'streams':{k:[(elements.UUID(u),i) for u,i in v] for k,v in snapshot['streams'].items()}}
    for version,state in [('portable',snapshot),('legacy_uuid_hash',legacy_snapshot)]:
        snapshot_path=Path(folder)/(version+'.pkl')
        snapshot_path.write_bytes(pickle.dumps(state))
        subprocess.run([sys.executable,'-c',
        "import pickle,sys,numpy as np; from embodied import Replay; "
        "r=Replay(65,512,sys.argv[1],chunksize=32,compress=True); "
        "r.load(pickle.load(open(sys.argv[2],'rb'))); "
        "x=r.sample(16)['state']; assert (np.diff(x[...,0],axis=1)==1).all(); "
        "assert (np.diff(x[...,1],axis=1)==0).all(); print('cross-process restore passed')",
            str(folder),str(snapshot_path)],check=True,env={**os.environ,'PYTHONHASHSEED':'12345'})
    add(legacy, 0, 512)
    assert not any(v == 0 for v in legacy.refs.values()), 'Unreachable tails retained'
    # C items + W*(L-1) stream entries + two chunk boundary paddings.
    bound = 4096 + 32 * (64 + 2 * 16)
    slots = sum(c.size for c in legacy.chunks.values())
    assert slots <= bound, (slots, bound)
    aligned(legacy)
result = dict(passed=True, resident_slots=slots, bound=bound,
              legacy_restore=True, exact_stream_restore=True, temporal_alignment=True,
              mmap_bitwise_equivalent=True,lz4_bitwise_equivalent=True,
              cross_process_restore=True)
Path('metrics/replay_memory_contract.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
