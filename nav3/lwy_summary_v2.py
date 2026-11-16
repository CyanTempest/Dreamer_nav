"""Summarize controlled diagnosis without merging old/new evaluation protocols."""
import json
from pathlib import Path
import numpy as np


def main():
    def read(tag):return json.loads(Path(f'metrics/lwy_{tag}.json').read_text())
    short=[]
    for mode in ['baseline_run','dense','lookahead','slow','no_drift','margin','tracker_parity','yaw4','yaw6']:
        d=read('v2_probe_'+mode)
        short.append(dict(mode=mode,successes=d['segment_pass'],episodes=d['episodes'],parameters=d['arguments'],
            endpoint_replay_error_m=d['physics_trace_endpoint_error_m']))
    base=read('v2_full_baseline');full=[]
    for mode in ['baseline','no_drift','yaw4']:
        d=read('v2_full_'+mode)
        full.append(dict(mode=mode,successes=d['segment_pass'],episodes=d['episodes'],
            improved=[x['map_seed'] for x,y in zip(d['rows'],base['rows']) if x['segment_pass'] and not y['segment_pass']],
            regressed=[x['map_seed'] for x,y in zip(d['rows'],base['rows']) if not x['segment_pass'] and y['segment_pass']]))
    old=read('g1_blind_dynamic')
    identical=[x['action_pose_sha256']==y['action_pose_sha256'] for x,y in zip(base['rows'],old['rows']) if not x['segment_pass']]
    assert len(identical)==11 and all(identical)
    a=read('v2_probe_baseline_run');b=read('v2_probe_tracker_parity')
    assert [r['action_pose_sha256'] for r in a['rows']]==[r['action_pose_sha256'] for r in b['rows']]
    geometry=read('v2_geometry_diagnosis')['rows']
    targets={r['seed'] for r in geometry}
    trace=[r for r in read('v2_full_baseline_analysis')['rows'] if r['map_seed'] in targets]
    result=dict(short_regressions=short,full_history_regressions=full,
        previous_internal_failures_exactly_reproduced=11,default_tracker_hash_parity_episodes=20,
        adopted_tracker_change=dict(parameter='yaw_tracking_gain',before=2,after=4,
            physical_checker_changed=False,envelope_changed=False,terrain_speed_changed=False,
            full_old_successes_before=9,full_old_successes_after=12,
            original_gate_shift_gain2_successes=int(read('v2_original_gain2')['success_rate']*10),
            original_gate_shift_gain4_successes=int(read('v2_original_gain4')['success_rate']*10)),
        geometry_rows=geometry,first_bad_substep_rows=trace,
        coordinate_correction=dict(positive_lateral='route left: normal=(-t_y,t_x)',negative_lateral='route right',
            left_count=sum(r['actual_lateral_m']>0 for r in geometry),right_count=sum(r['actual_lateral_m']<0 for r in geometry)),
        hypotheses=dict(tracking='gain-only intervention improves 9/20 to 12/20; residual failures remain',
            localization='zero drift gives 10/20, including regressions; drift alone is not sufficient',
            candidates='15 to 57 samples gives 12/20 in both short runs; not adopted',
            support='first bad substeps replay exactly; original footprint remains inside geometric support, conservative raster limit is reached',
            collision='walls/rails are in the original primitive projections; some variants fail independent 10 cm clearance and are rejected',
            importer_bug='no unsupported raster optimism or mislabeled step surface demonstrated by these probes; no physical checker changes adopted'))
    with Path('metrics/lwy_v2_diagnosis_summary.json').open('x') as f:json.dump(result,f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar([r['mode'] for r in short],[r['successes'] for r in short],color='#447a9c')
    axes[0].tick_params(axis='x',rotation=55);axes[0].set(ylim=(0,20),ylabel='Reached s=998 m / 20',title='Old maps: short regression from 970 m')
    axes[1].bar([r['mode'] for r in full],[r['successes'] for r in full],color=['#888888','#b58650','#438569'])
    axes[1].set(ylim=(0,20),ylabel='Reached s=998 m / 20',title='Old maps: complete observation history')
    fig.tight_layout();fig.savefig('figures/lwy_v2_parameter_regressions.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(previous_failures_reproduced=11,tracker_parity=20,full=full,coordinate_correction=result['coordinate_correction'])),flush=True)


if __name__=='__main__':main()
