"""Summarize controlled runtime diagnosis without changing any model weights."""
import json
from pathlib import Path
import numpy as np


def main():
    def read(path):return json.loads(Path(path).read_text())
    fields=['opt/loss','opt/grad_norm','opt/update_rms','loss/bev','loss/priv_trav',
        'loss/rew','loss/value','ent/action','ret','ret_min','ret_max']
    comparisons={}
    for batch in ['old','new']:
        comparisons[batch]={}
        for level in [0,1]:
            suffix='_autotune1' if level else ''
            path=f'metrics/lwy_v2_grad_healthy_{batch}{suffix}.json'
            data=read(path);rows=[r for r in data['rows'] if r]
            comparisons[batch][str(level)]=dict(artifact=path,source_sha256=data['source_sha256'],
                values={key:dict(first=rows[0][key],last=rows[-1][key],minimum=min(r[key] for r in rows),
                    maximum=max(r[key] for r in rows)) for key in fields},rows=rows)
        assert comparisons[batch]['0']['source_sha256']==comparisons[batch]['1']['source_sha256']
    logs=[json.loads(line) for line in Path('metrics/lwy_v2_autotune0_train_dreamer_lwy.jsonl').read_text().splitlines()]
    training=[r for r in logs if r['event']=='train']
    config=read('metrics/lwy_v2_autotune0_config_dreamer_lwy.json')
    batch_metadata=read('metrics/lwy_v2_health_batches.json')
    validation=read('metrics/lwy_v2_health_val_smoke.json')
    result=dict(cause='Forcing evaluation autotune=0 into Dreamer training changes GPU kernels and produces pathological finite gradients, even on frozen old replay with healthy weights.',
        intervention='Restore original Dreamer training autotune=1; hold weights, frozen batch, seed, learning rate and optimizer fixed.',
        comparisons=comparisons,forward=read('metrics/lwy_v2_forward_runtime_comparison.json'),
        abandoned_run=dict(steps=313344,updates=289,checkpoint='checkpoints/lwy_v2_autotune0_diagnostic/dreamer_lwy',
            replay='replay/lwy_v2_autotune0_diagnostic/dreamer_lwy',snapshot='checkpoints/snapshots/dreamer_lwy_0.10h',
            excluded_from_selection=True),
        training_log=training,batch_metadata=batch_metadata,
        validation_observation=dict(artifact='metrics/lwy_v2_health_val_smoke.json',
            success_rate=validation['success_rate'],mean_gates=validation['mean_gates'],
            failures=validation['failures'],
            rows=[{key:r.get(key) for key in ['map','noise_seed','progress_m','failure','last_action','terminal_xy','terminal_yaw','reward_component_sums']} for r in validation['rows']],
            inference_runtime_affected=True,
            interpretation='The evaluation itself uses the affected level0 inference, so these positions/actions cannot isolate damage to learned weights. They are retained as observations, not as valid model-selection scores.'),
        hypotheses={
            'near_obstacle_penalty':dict(coefficient=-.04,changed=False,
                finding='Old replay alone reproduces the anomaly before any new-map reward or obstacle exposure. This rules out a new near-obstacle penalty as the cause of this numerical failure; it does not rule out later behavioral effects.'),
            'fresh_replay':dict(retained_sequence_starts=100000,fresh=False,
                finding='The online priority queue temporarily favors newly completed 65-step sequences. Its length at the smoke end was zero; it does not permanently exclude old replay. Frozen old replay alone also fails at autotune=0.'),
            'synchronized_resets':dict(initial=True,periodic=False,
                finding='Frozen old batch has no is_first markers after the context frame, yet reproduces the anomaly. Segment checkpoints preserve collector state; initial reset cannot explain the old-batch control.'),
            'learning_rate':dict(value=config['learning_rate'],optimizer_preserved=True,
                finding='The first loss and raw gradient already differ by runtime setting before the first new update. Learning rate is unchanged and cannot explain that pre-update difference; no tuning is justified by this probe.')},
        diagnostic_gpu_allowance_seconds=900,diagnostic_gpu_cap_seconds=3600,
        prior_evidence=dict(formal_round5_entry='nav3/fix_eval.py: setup uses autotune=1 before JAX initialization',
            old_probes=['metrics/fix_numeric_dreamer_level0.json','metrics/fix_numeric_dreamer_level1.json'],
            finding='The same constant-head incompatibility was already recorded in round5. The round6 LWY level0 qualification exercised only the rule planner; extending that setting to neural training/inference was an implementation mistake.'),
        evaluation_protocol_decision_pending=True,
        limitation='Restored training numerics do not establish learned navigation quality. Full smoke, validation and final comparison remain necessary. Autotune=0 also gives a spatially constant Dreamer head during fixed-100 inference; the explicit evaluation requirement needs a user decision.')
    Path('metrics/lwy_v2_training_numerics_diagnosis.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,batch in enumerate(['old','new']):
        for j,key in enumerate(['opt/loss','opt/grad_norm','opt/update_rms']):
            ax=axes[i,j]
            for level in ['0','1']:
                rows=comparisons[batch][level]['rows'];ax.plot(range(1,len(rows)+1),[r[key] for r in rows],'-o',label='autotune='+level)
            ax.set(title=batch+' batch: '+key,xlabel='Update');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig('figures/lwy_v2_training_numerics.png',dpi=170)
    print(json.dumps(dict(controlled_runtime_effect=True,training_autotune=1,evaluation_decision_pending=True)),flush=True)


if __name__=='__main__':main()
