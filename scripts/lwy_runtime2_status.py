"""Small read-only status for 25-minute SSH checks; no job starts here."""
import json
from pathlib import Path
import subprocess
q=json.loads(Path('metrics/lwy_runtime2_queue.json').read_text())
out={k:q.get(k) for k in ['status','active_job','active_log','active_logs','active_started_utc','error','selected_envs']}
out['charged_gpu_hours']=(q['gpu_job_wall_seconds']+q['preceding_g1_and_diagnostics_allowance_seconds'])/3600
out['validation']={name:[{k:r[k] for k in ['label','success_rate','mean_gates']} for r in run['rounds']] for name,run in q['training'].items()}
for method in ['dreamer','ppo']:
    path=Path(f'metrics/train_{method}_lwy_runtime2.jsonl')
    if not path.exists():continue
    rows=[json.loads(line) for line in subprocess.check_output(['tail','-n','25',str(path)],text=True).splitlines()]
    trains=[r for r in rows if r.get('event') in ['train','update_trace']]
    if trains:
        r=trains[-1];out[method]={k:r.get(k) for k in ['event','steps','updates','wall_seconds','mean_return','mean_gates','replay_old_sample_fraction','replay_old_cumulative_fraction']}
        out[method]['metrics']={k:v for k,v in r.get('metrics',{}).items() if any(s in k for s in ['grad_norm','opt/loss','value_loss','entropy','retnorm'])}
        mix=next((x.get('episode_mix') for x in reversed(trains) if x.get('episode_mix')),None)
        if mix:out[method]['episode_mix']={k:v for k,v in mix.items() if k!='map_episode_counts'}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
out['final_conditions']=len(list(Path('metrics').glob('lwy_runtime2_final_*_rt1.json')))
print(json.dumps(out,indent=2))
