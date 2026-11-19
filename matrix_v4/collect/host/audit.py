"""Replay qualification and decode a recorded wire frame for each driveable production shard."""
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
base=Path(os.environ['MATRIX_COLLECTION_DIR']).resolve()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from codec import unpack_wire
from control import attitude
from sensors import decode

def audit():
    hist=defaultdict(lambda:defaultdict(list))
    output=defaultdict(lambda:dict(reasons=Counter(),stand=0,attempt=0,reset_ok=0,reset_attempt=0,
                                   steps=0,qualified=0,shards=0,bytes=0,shapes=Counter()))
    settings=set()
    sample=None
    finished={p.parent.name for p in (base/'results').glob('*/*/done.json')}
    for path in sorted((base/'data').glob('prod*.npz')):
        if path.stem.rsplit('-s',1)[0] not in finished:
            continue
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['meta']))
            card=m['gpu']
            o=output[card]
            o['shards']+=1
            o['bytes']+=path.stat().st_size
            o['reasons'][m['termination']]+=1
            settings.add(json.dumps(m['config'],sort_keys=True))
            if 'topic_counts' in m:
                ok=m['stand_accepted']
                o['attempt']+=1
                o['stand']+=int(ok)
                if not path.stem.endswith('e000'):
                    o['reset_attempt']+=1
                    o['reset_ok']+=int(ok)
            if not m['steps']:
                continue
            ages=z['age_s']
            pose=z['priv_pose']
            c=m['config']
            tilt=np.array([max(abs(x) for x in attitude(q)[:2]) for q in pose[:,3:]])
            pred=((ages>=0)&(ages<=c['age_s'])).all(1)&(z['priv_clearance']>=c['min_clearance_m'])&(tilt<c['max_tilt_rad'])
            np.testing.assert_array_equal(pred,z['qualified'])
            assert len(ages)==m['steps']
            o['steps']+=len(ages)
            o['qualified']+=int(pred.sum())
            for key,arr in [('front_age',ages[:,0]),('rear_age',ages[:,1]),
                            ('odom_epoch_lag',z['mono']+m['epoch_minus_mono']-z['odom_header']),
                            ('imu_minus_odom',z['imu_header']-z['odom_header']),
                            ('clock_mapping_frac',np.isfinite(z['scan_sim_end']).all(1))]:
                hist[card][key].extend(np.asarray(arr,dtype=float).tolist())
            indices=z['frame_index']
            for k,side in enumerate(('front','rear')):
                ids=np.unique(indices[:,k])
                ids=ids[ids>=0]
                for i in ids:
                    h,w,slots,finite=z[f'{side}_{i:05d}_shape_count']
                    assert slots==h*w and finite<=slots
                    o['shapes'][f'{h}x{w}']+=1
                if not len(ids):
                    continue
                prefix=f'{side}_{ids[0]:05d}_'
                raw=unpack_wire(z[prefix+'cdr_zstd'])
                scan=decode(raw,float(z[prefix+'times'][2]))
                np.testing.assert_array_equal(scan.distance,z[prefix+'mm'])
                np.testing.assert_array_equal(scan.intensity,z[prefix+'intensity'])
            if sample is None or int(pred.sum())>sample['qualified']:
                sample=dict(name=path.name,gpu=card,steps=len(ages),qualified=int(pred.sum()),
                    reason=m['termination'],window=m['window'],size=path.stat().st_size,
                    sha256=path.with_suffix('.sha256').read_text().split()[0])
    assert len(settings)<=1
    for card,metrics in hist.items():
        o=output[card]
        for k,v in metrics.items():
            a=np.array(v)
            if k=='clock_mapping_frac':
                o[k]=float(a.mean())
            else:
                a=a[np.isfinite(a)]
                o[k+'_p10_p50_p90']=np.percentile(a,[10,50,90]).tolist() if len(a) else []
    result=dict(cards=output,sample=sample,config_count=len(settings))
    (base/'results/audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    audit()
