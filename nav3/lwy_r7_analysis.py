"""Read-only historical evidence; no causal claims without interventions."""
import json
from pathlib import Path
import numpy as np


def main():
    log=Path('metrics/train_dreamer_lwy_runtime2.jsonl')
    events=[json.loads(s) for s in log.read_text().splitlines()]
    rows=[r for r in events if r['event']=='train' and r.get('wall_seconds',0)>=10800]
    keys=sorted({k for r in rows for k in r.get('metrics',{}) if any(x in k for x in ['loss/','ent/','ret','grad','rand/','scale'])})
    series={k:[[r['wall_seconds']/3600,r['metrics'][k]] for r in rows if k in r.get('metrics',{})] for k in keys}
    failures={}
    for method in ['dreamer_lwy','planner']:
        for dataset in ['original','blind']:
            for ob in ([0,1] if method=='planner' else [0]):
                p=Path(f'metrics/lwy_runtime2_final_{method}_{dataset}_ob{ob}_sf0_rt1.json')
                result=json.loads(p.read_text());items=[];cache={}
                for r in result['rows']:
                    if r['success']:continue
                    if r['map'] not in cache:cache[r['map']]=dict(np.load(Path('maps_lwy_v2')/r['map']/'map.npz'))
                    a=cache[r['map']];pos=np.array(r['terminal_xy']);idx=((a['route'][:,:2]-pos)**2).sum(-1).argmin()
                    yaw=float(a['yaw'][idx]);delta=pos-a['route'][idx,:2];lateral=float(-np.sin(yaw)*delta[0]+np.cos(yaw)*delta[1])
                    act=np.clip(r['last_action'],-1,1);distance=.5+(act[0]+1)*1.25;angle=act[1]*np.pi/3
                    history=np.array(r['trajectory'])[-20:];diff=np.diff(history[:,5]) if len(history)>1 else np.array([])
                    items.append(dict(map=r['map'],noise_seed=r['noise_seed'],s=r['progress_m'],section=r['section'],
                        failure=r['failure'],lateral_m=lateral,road_width_m=float(a['width'][idx]),
                        target_body_m=[float(distance*np.cos(angle)),float(distance*np.sin(angle))],
                        requested_speed_mps=float((act[2]+1)*.4),last_action=r['last_action'],
                        sampled_steer_direction_changes=int(np.sum(diff[:-1]*diff[1:]<0)),
                        trajectory_sampling_seconds=5.,gates=r['gates']))
                failures[f'{method}_{dataset}_ob{ob}']=items
    collapse=json.loads(Path('metrics/lwy_v2_collapse_dreamer_lwy_runtime2_4.00h.json').read_text())
    result=dict(metric_series=series,replay=[dict(hours=r['wall_seconds']/3600,old_fraction=r.get('replay_old_sample_fraction'),items=r.get('replay_items')) for r in rows],
        failures=failures,previous_collapse=collapse['evidence'],
        notes=['Old entropy extraction omitted ent/action; this script reads the original log.',
               'Original ob1 planner mean gates is 1, not 0: failures occur after gate 1.',
               'Finite improving model losses do not establish policy robustness.',
               'Old replay fraction 0 predates 3 h; temporal association is not proof of root cause.',
               'Stored actions are raw Gaussian draws; physical environment clips [-1,1] before decoding.'])
    with Path('metrics/lwy_r7_historical_analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(series={k:[v[0],v[-1]] for k,v in series.items()},failure_counts={k:len(v) for k,v in failures.items()})),flush=True)


if __name__=='__main__':main()
