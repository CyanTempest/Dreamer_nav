"""Evidence for the historical discrepancy, without changing any historical JSON."""
import json
from pathlib import Path
import re
import numpy as np


def compare(a,b):
    pairs=list(zip(a['rows'],b['rows']));changed=[];flips=[];first=[];deltas=[]
    for i,(x,y) in enumerate(pairs):
        if {k:v for k,v in x.items() if k!='method'}!={k:v for k,v in y.items() if k!='method'}:changed.append(i)
        if x['success']!=y['success']:flips.append(i)
        xx=np.asarray(x['trajectory']);yy=np.asarray(y['trajectory']);n=min(len(xx),len(yy))
        dist=np.max(np.abs(xx[:n]-yy[:n]),axis=1);deltas.extend(dist.tolist())
        where=np.flatnonzero(dist)
        if len(where):first.append(dict(episode=i,first_record=int(where[0]),first_delta_m=float(dist[where[0]])))
    return dict(changed_episodes=changed,success_flips=flips,first_trajectory_differences=first,
        max_common_trajectory_coordinate_difference_m=max(deltas,default=0.))


def main():
    load=lambda name:json.loads(Path(name).read_text())
    names=['legacy_a','legacy_b','warm','deterministic_a','deterministic_b','deterministic_warm','autotune-only','ops-only']
    data={n:load(f'metrics/fix_repro_{n}.json') for n in names}
    old=load('metrics/eval_ppo_s0_test_ob0_sf0.json')
    result=dict(experiments={n:dict(success=int(round(d['success_rate']*d['episodes'])),episodes=d['episodes'],runtime=d['runtime']) for n,d in data.items()},
        cold_repeat=compare(data['legacy_a'],data['legacy_b']),cold_vs_old=compare(old,data['legacy_a']),
        warm_vs_old=compare(old,data['warm']),cold_vs_warm=compare(data['legacy_a'],data['warm']),
        fixed_runtime_vs_warm=compare(data['deterministic_a'],data['deterministic_warm']),
        configs_equal=all(d['env_config']==old['env_config'] for d in data.values()),
        checkpoint_hashes_equal=len({d['runtime']['checkpoint_sha256'] for d in data.values()})==1,
        batch_sizes={n:d['runtime']['batch_size'] for n,d in data.items()})
    probes={n:load(f'metrics/fix_numeric_{n}.json') for n in ['default','warm','level1']}
    arr={n:np.load(f'metrics/fix_numeric_{n}.npz')['action'] for n in probes}
    result['identical_input_probe']=dict(records=probes,
        input_equal=len({d['obs_sha256'] for d in probes.values()})==1,
        weights_equal=len({d['params_sha256'] for d in probes.values()})==1,
        random_key_equal=all(d['key']==probes['default']['key'] for d in probes.values()),
        action_max_abs_delta={n:float(np.max(abs(arr[n]-arr['default']))) for n in probes},
        algorithm_ids={n:sorted(set(re.findall(r'"algo_id"\s*:\s*"?(-?\d+)',Path(f'logs/fix_numeric_{n}.hlo').read_text()))) for n in probes},
        convolution_algorithms={n:[json.loads(line.split('backend_config=',1)[1])['cudnn_conv_backend_config']['algorithm']
            for line in Path(f'logs/fix_numeric_{n}.hlo').read_text().splitlines() if 'custom_call_target="__cudnn' in line and 'backend_config=' in line] for n in probes})
    Path('metrics/fix_repro_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['experiments','cold_vs_old','cold_vs_warm','warm_vs_old']}),flush=True)


if __name__=='__main__':main()
