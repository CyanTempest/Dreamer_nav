"""Frozen old/new replay batches and disposable gradient checks; no saved updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys


def freeze():
    import numpy as np
    import elements
    sys.path.insert(0,str(Path('dreamerv3').resolve()))
    import embodied
    with Path('checkpoints/dreamer_s0/latest.pkl').open('rb') as f:original=pickle.load(f)
    old=original['replay_state'];old_ids={elements.UUID(r[0]).value for r in old['chunks']}
    root=Path('checkpoints/dreamer_lwy/official_ckpt/ckpt')
    with (root/(root/'latest').read_text().strip()/'replay.pkl').open('rb') as f:new=pickle.load(f)
    metadata={}
    for name,data,directory in [('old',old,'replay/dreamer_s0'),('new',new,'replay/dreamer_lwy')]:
        raw=lambda x:x if isinstance(x,bytes) else elements.UUID(x).value
        items=[v for v in data['items'] if name=='old' or raw(v[0]) not in old_ids][-16:]
        records={elements.UUID(r[0]).value:r for r in data['chunks']};needed=set()
        for uid,index in items:
            uid=raw(uid);remaining=65
            while remaining>0:
                record=records[uid];needed.add(uid)
                remaining-=int(record[1].removesuffix('.npz').split('-')[-1])-index
                index=0;uid=elements.UUID(record[3]).value
        subset={**data,'items':items,'chunks':[r for r in data['chunks'] if elements.UUID(r[0]).value in needed],
            'current':{},'streams':{},'queue':[],'lengths':{}}
        replay=embodied.Replay(length=65,capacity=32,online=False,seed=0,compress=True,
            directory=f'diagnostics/lwy_v2_batch_store_{name}')
        replay.load(subset,directory=directory)
        batch=replay.sample(16);batch['consec']=np.zeros(batch['is_first'].shape,np.int32)
        path=Path(f'diagnostics/lwy_v2_health_{name}_batch.npz');assert not path.exists()
        np.savez_compressed(path,**batch)
        metadata[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            first_fraction=float(batch['is_first'][:,1:].mean()),terminal_fraction=float(batch['is_terminal'].mean()),
            reward_mean=float(batch['reward'].mean()),
            shapes={k:list(v.shape) for k,v in batch.items()},finite=all(np.isfinite(v).all() for v in batch.values()))
    metadata['online_queue_at_smoke_end']=len(new['queue'])
    metadata['online_queue_rule']='One priority sequence per 65 collected steps per worker; remaining samples are uniform'
    Path('metrics/lwy_v2_health_batches.json').write_text(json.dumps(metadata,indent=2))


def check(weights,batch_name,steps,autotune):
    from .lwy_eval_v2 import setup
    setup();os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['XLA_FLAGS']=f'--xla_gpu_autotune_level={autotune} --xla_gpu_deterministic_ops=true'
    import numpy as np
    from .dreamer_adapter import DreamerPolicy
    source=f'checkpoints/{"dreamer_s0" if weights=="healthy" else "dreamer_lwy"}/latest.pkl'
    with Path(source).open('rb') as f:data=pickle.load(f)
    policy=DreamerPolicy(0,'logs/lwy_v2_gradient_diagnosis',jax_overrides=dict(autotune=autotune,deterministic=True))
    policy.agent.load(data['agent']);del data
    batch=dict(np.load(f'diagnostics/lwy_v2_health_{batch_name}_batch.npz'))
    def batches():
        while True:yield batch
    stream=iter(policy.agent.stream(batches()));carry=policy.agent.init_train(16);rows=[]
    for i in range(steps):
        carry,_,metrics=policy.agent.train(carry,next(stream))
        row={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).ndim==0}
        assert all(np.isfinite(v) for k,v in row.items() if 'loss' in k or 'norm' in k or 'rms' in k)
        rows.append(row)
        print(json.dumps(dict(step=i,**{k:v for k,v in row.items() if k.startswith('opt/') or k in ['ent/action','ret']})),flush=True)
    suffix=f'_autotune{autotune}' if autotune else ''
    path=Path(f'metrics/lwy_v2_grad_{weights}_{batch_name}{suffix}.json');assert not path.exists()
    path.write_text(json.dumps(dict(weights=weights,batch=batch_name,source=source,steps=steps,rows=rows,
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),checkpoint_written=False,autotune=autotune,
        protocol='Disposable original/current Agent; fixed batch, seed and settings; short update probe only'),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true')
    p.add_argument('--weights',choices=['healthy','smoke']);p.add_argument('--batch',choices=['old','new'])
    p.add_argument('--steps',type=int,default=8);p.add_argument('--autotune',type=int,default=0);a=p.parse_args()
    if a.freeze:freeze()
    else:check(a.weights,a.batch,a.steps,a.autotune)


if __name__=='__main__':main()
