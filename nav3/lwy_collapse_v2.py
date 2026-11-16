"""Read-only collapse diagnosis from recorded scalar histories and fixed validation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--state',default='metrics/lwy_v2_queue.json');p.add_argument('--run-name');a=p.parse_args()
    run_name=a.run_name or a.method
    queue=json.loads(Path(a.state).read_text());run=queue['training'][a.method]
    rows=[json.loads(line) for line in Path(f'metrics/train_{run_name}.jsonl').read_text().splitlines()]
    training=[r for r in rows if r['event']=='train'];best=run['best'];last=run['rounds'][-1]
    evaluations={key:json.loads(Path(row['evaluation']).read_text()) for key,row in [('best',best),('last',last)]}
    series={}
    for row in training:
        for key,value in row.get('metrics',{}).items():
            if any(token in key for token in ['loss','entropy','std','scale','grad','norm','ret','reset']):
                series.setdefault(key,[]).append([row['wall_seconds'],value])
    evidence={}
    for label,ev in evaluations.items():
        failure=[r for r in ev['rows'] if not r['success']]
        evidence[label]=dict(success_rate=ev['success_rate'],mean_gates=ev['mean_gates'],
            failures=[{k:r.get(k) for k in ['map','noise_seed','progress_m','failure','last_action','reward_component_sums','terminal_xy','terminal_yaw']} for r in failure])
    fresh=[r for r in rows if r['event']=='initialize_from_original']
    replay=[[r['wall_seconds'],r.get('replay_old_sample_fraction') if r.get('updates',0)>0 else None] for r in training]
    initialization=[r for r in rows if r['event'] in ['initialize_from_original','initialize_lwy_from_original','resize_envs','resume']]
    result=dict(method=a.method,trigger=run.get('stop_reason'),read_only=True,training_restarted=False,
        evidence=evidence,metric_series=series,replay_old_fraction=replay,initialization_events=initialization,
        hypotheses={
            'near_obstacle_penalty':dict(environment_coefficient=-.04,new_penalty_introduced=False,
                evidence='Per-failure and episode accumulated near-obstacle, lateral, progress and failure rewards are recorded above. Association alone does not establish causality.'),
            'fresh_replay':dict(applicable=a.method.startswith('dreamer'),initialization=fresh,
                excluded_when_retained_items_positive=bool(fresh and fresh[-1].get('retained_old_items',0)>0),
                evidence='Measured old-sample fraction over the complete run is included; decay can still produce distribution shift.'),
            'synchronized_resets':dict(evidence='Full mixed collector and policy state are checkpointed and restored between segments; initialization events and reset-fraction metrics are included.',
                initial_sync_reset=True,scheduled_sync_resets=False),
            'learning_rate':dict(configuration=json.loads(Path(f'metrics/config_{run_name}.json').read_text()),
                evidence='Unchanged original optimizer and learning-rate configuration; gradient, entropy, value/reward scale histories are included. No causal learning-rate ablation was run.')},
        conclusion='Early-stop executed. Evidence supports comparison of the four hypotheses, but no unique causal diagnosis is asserted without an intervention.',
        gpu_diagnosis_seconds=0)
    output=Path(f'metrics/lwy_v2_collapse_{run_name}_{last["label"]}h.json')
    with output.open('x') as f:json.dump(result,f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[k for k in series if len(series[k])>1][:12]
    if keys:
        fig,axes=plt.subplots(4,3,figsize=(13,10),squeeze=False)
        for ax,key in zip(axes.ravel(),keys):
            values=np.asarray(series[key],float);ax.plot(values[:,0]/3600,values[:,1]);ax.set_title(key,fontsize=8);ax.set_xlabel('training wall h')
        fig.tight_layout();fig.savefig(f'figures/lwy_v2_collapse_{run_name}_{last["label"]}h.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(method=a.method,stopped=True,diagnosis=str(output))),flush=True)


if __name__=='__main__':main()
