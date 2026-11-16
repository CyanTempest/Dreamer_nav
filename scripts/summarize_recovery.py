"""Summarize measured RSS/GPU curves; no training or evaluation changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    if not Path(path).exists():return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


memory=rows('metrics/memory_dreamer_s0.jsonl')
training=[r for r in memory if r['stage']=='train' and r.get('replay_storage')=='lz4']
if not training:raise SystemExit('No resumed training samples yet')
pids={r['pid'] for r in training}
spans={pid:max(r['time'] for r in training if r['pid']==pid)-min(r['time'] for r in training if r['pid']==pid) for pid in pids}
pid=training[-1]['pid']
if spans[pid]<600:raise SystemExit(f'Training validation pending: {spans[pid]:.1f}/600 seconds')
memory=[r for r in memory if r['pid'] in pids]
telemetry=rows('metrics/queue_telemetry.jsonl')
gpu=[];external=[];swap=[]
for row in telemetry:
    processes=[p for p in row['processes'] if p['pid'] in pids]
    if not processes:continue
    used,total,_=map(int,row['gpu'].split(','));gpu.append(used/total)
    for process in processes:
        external.append(dict(time=row['time'],rss_gb=process['rss_kb']*1024/1e9))
        swap.append(process['swap_kb']*1024)
measured=[r['rss_gb'] for r in memory+external]
measured.extend(r['rss_before_trim_bytes']/1e9 for r in memory if 'rss_before_trim_bytes' in r)
for check in rows('metrics/ssh_supervision.jsonl'):
    measured.extend(p['rss_hwm_bytes']/1e9 for p in check.get('process_status',[])
                    if p['pid'] in pids)
evidence=json.loads(Path('metrics/oom_recovery_evidence.json').read_text())
# Full environment-size headroom is deliberately conservative: new snapshots
# copy only one leaf at a time. A further 2 GB covers remaining index growth,
# allocator overhead and fixed-shape evaluation/plotting workspaces.
storage=[r for r in memory if r.get('replay_rows',0) and r['pid']==pid]
chunksize=json.loads(Path('metrics/config_dreamer_s0.json').read_text())['replay_chunksize']
steady_rows=1000000+4608*(64+chunksize-1)
density=max((r['replay_bytes']/r['replay_rows'] for r in storage),default=None)
base=max((r['rss_gb']-r['replay_bytes']/1e9 for r in storage),default=0)
projection=max(max(measured),base+steady_rows*(density or 0)/1e9)+evidence['checkpoint_sections_bytes']['env']/1e9+2
at_capacity=any(r.get('replay_items',0)==1000000 and r['pid']==pid for r in memory)
summary=dict(dreamer_envs=4608,chunksize=chunksize,replay_capacity=1000000,bytes_per_step=39003,
    validation_pid=pid,validation_seconds=spans[pid],rss_min_gb=min(measured),rss_max_gb=max(measured),
    rss_last_gb=memory[-1]['rss_gb'],replay_max_gb=max(r.get('replay_bytes',0) for r in memory)/1e9,
    gpu_peak_fraction=max(gpu,default=0),process_swap_max_bytes=max(swap,default=0),
    five_hour_rss_allowance_gb=projection,
    projection_basis='latest process at capacity for >=600s; all recovery-process observed peaks versus latest memory-policy base + bounded rows * measured compressed bytes/row; add full old environment bytes + 2 GB index/allocator/evaluation headroom',
    compressed_bytes_per_row=density,steady_row_bound=steady_rows,
    projection_pid=pid,projection_memory_policy=storage[-1].get('memory_policy','checkpoint_only_trim'),
    restored_items=812960,restored_raw_steps=1111968,
    valid_training_pids=sorted(pids),latest_observed_steps=training[-1]['steps'],
    reached_capacity=at_capacity,
    validation_passed=spans[pid]>=600 and at_capacity and max(measured)<45 and projection<45)
Path('metrics/oom_recovery_summary.json').write_text(json.dumps(summary,indent=2))
origin=min(r['time'] for r in memory)
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
rss_curve=sorted(memory+external,key=lambda r:r['time'])
axes[0].plot([(r['time']-origin)/60 for r in rss_curve],[r['rss_gb'] for r in rss_curve],label='Process RSS (decimal GB)')
axes[0].axhline(45,color='#cb6b20',ls='--',label='RSS target 45 GB')
axes[0].axhline(50,color='#bc3b3b',ls=':',label='Checkpoint watchdog 50 GB')
axes[0].set(ylabel='RSS (GB)',ylim=(0,max(52,max(measured)+2)))
axes[0].legend();axes[0].grid(alpha=.2)
storage=[r for r in memory if 'replay_bytes' in r]
axes[1].plot([(r['time']-origin)/60 for r in storage],[r['replay_bytes']/1e9 for r in storage],label='Losslessly compressed replay payload')
axes[1].set(xlabel='Minutes since recovery process initialization',ylabel='Compressed replay (GB)')
axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig('figures/oom_recovery_memory.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
assert summary['validation_passed'],summary
