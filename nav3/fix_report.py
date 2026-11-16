"""Generate round-five tables, cluster bootstrap and figures from saved JSON."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,c-h)),float(min(1,c+h))]


def summarize(d):
    rows=d['rows'];n=len(rows);k=sum(r['success'] for r in rows)
    return dict(method=d['method'],split=d['split'],ob=int(d['random_obstacles']),sf=int(d['filtered']),n=n,
        success=k,rate=k/n,wilson=wilson(k,n),gates=float(np.mean([r['gates'] for r in rows])),
        failures=d['failures'],takeover=sum(r['takeovers'] for r in rows)/sum(r['steps'] for r in rows),
        speed=sum(r['distance_m'] for r in rows)/sum(r['duration_s'] for r in rows),iou=d.get('trav_iou'))


def paired(old,new):
    key=lambda r:(r['map'],r['episode'])
    a={key(r):r for r in old['rows']};b={key(r):r for r in new['rows']};assert a.keys()==b.keys()
    names=sorted({k[0] for k in a});rng=np.random.default_rng(9042)
    sampled=rng.integers(0,len(names),(10000,len(names)))
    result={}
    for metric,fn in [('success',lambda r:float(r['success'])),('gates',lambda r:r['gates'])]:
        delta=np.array([np.mean([fn(b[k])-fn(a[k]) for k in a if k[0]==m]) for m in names])
        ci=np.quantile(delta[sampled].mean(1),[.025,.975])
        result[metric]=dict(difference=float(delta.mean()),ci95=ci.tolist(),significant_regression=bool(ci[1]<0))
    ratios=[('speed','distance_m','duration_s',None),('takeover','takeovers','steps',None)]
    if old.get('trav_iou') is not None and new.get('trav_iou') is not None:
        ratios += [(f'iou_{j}','trav_intersection','trav_union',j) for j in range(3)]
    for metric,num,den,channel in ratios:
        def totals(rows):
            value=lambda r,k:r[k] if channel is None else r[k][channel]
            return np.array([[sum(value(r,num) for r in rows.values() if r['map']==m),
                sum(value(r,den) for r in rows.values() if r['map']==m)] for m in names],float)
        x,y=totals(a),totals(b)
        if x[:,1].sum()==0 or y[:,1].sum()==0:
            result[metric]=None;continue
        diff=float(y[:,0].sum()/y[:,1].sum()-x[:,0].sum()/x[:,1].sum())
        xx,yy=x[sampled].sum(1),y[sampled].sum(1)
        valid=(xx[:,1]>0)&(yy[:,1]>0)
        ci=np.quantile(yy[valid,0]/yy[valid,1]-xx[valid,0]/xx[valid,1],[.025,.975])
        result[metric]=dict(difference=diff,ci95=ci.tolist(),significantly_lower=bool(ci[1]<0),significantly_higher=bool(ci[0]>0))
    result.update(maps=len(names),resamples=10000,seed=9042)
    return result


def main():
    sources=sorted(Path('metrics').glob('fix_eval_v1_*.json'))
    data=[json.loads(p.read_text()) for p in sources]
    index={(d['method'],d['split'],int(d['random_obstacles']),int(d['filtered'])):d for d in data}
    rows=[summarize(d) for d in data];comparisons={}
    for method in ['dreamer','ppo']:
        for split in ['heldout','blind']:
            for ob in [0,1]:
                for sf in [0,1]:
                    k=(f'{method}_s0',split,ob,sf);nk=(f'{method}_s0_ft',split,ob,sf)
                    if k in index and nk in index:comparisons[f'{method}_{split}_{ob}_{sf}']=paired(index[k],index[nk])
    result=dict(rows=rows,paired_bootstrap=comparisons,expected_conditions=60,complete=len(index)==60)
    Path('metrics/fix_summary.json').write_text(json.dumps(result,indent=2))
    lines=['| 方法 | 数据集 | ob/sf | 通过 / N | Wilson 95% | 平均门 | 接管率 | 速度 m/s | IoU 全/遮挡/障碍邻域 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        ci=r['wilson'];iou='不适用' if r['iou'] is None else '/'.join('不适用' if x is None else f'{x:.3f}' for x in r['iou'])
        lines.append(f'| {r["method"]} | {r["split"]} | {r["ob"]}/{r["sf"]} | {r["success"]}/{r["n"]} ({r["rate"]:.1%}) | {ci[0]:.1%}–{ci[1]:.1%} | {r["gates"]:.3f} | {r["takeover"]:.1%} | {r["speed"]:.3f} | {iou} |')
    Path('metrics/fix_evaluation_table.md').write_text('\n'.join(lines)+'\n')
    paired_lines=['| 比较 | 全通过率差 | 地图 bootstrap 95% | 平均门差及区间 | 显著回退 |','| --- | --- | --- | --- | --- |']
    for k,v in comparisons.items():
        s=v['success'];g=v['gates'];ci=s['ci95'];gc=g['ci95']
        paired_lines.append(f'| {k} | {s["difference"]:+.1%} | {ci[0]:+.1%}–{ci[1]:+.1%} | {g["difference"]:+.3f} [{gc[0]:+.3f}, {gc[1]:+.3f}] | {s["significant_regression"] or g["significant_regression"]} |')
    Path('metrics/fix_bootstrap_table.md').write_text('\n'.join(paired_lines)+'\n')
    secondary=['| 比较 | 指标 | 差值 | 地图 bootstrap 95% |','| --- | --- | --- | --- |']
    for k,v in comparisons.items():
        for metric in ['speed','takeover','iou_0','iou_1','iou_2']:
            r=v.get(metric)
            if r is not None:secondary.append(f'| {k} | {metric} | {r["difference"]:+.4f} | {r["ci95"][0]:+.4f}–{r["ci95"][1]:+.4f} |')
    Path('metrics/fix_secondary_bootstrap_table.md').write_text('\n'.join(secondary)+'\n')
    diagnostics=[]
    diagnostic_table=['| 方法 | 诊断场景 | ob/sf | 段通过 / N | Wilson 95% | 失败分类 |','| --- | --- | --- | --- | --- | --- |']
    for p in sorted(Path('metrics').glob('fix_diagnostic_*.json')):
        if p.name=='fix_diagnostic_summary.json':continue
        d=json.loads(p.read_text());r=summarize(d);case=p.stem.split(d['method']+'_',1)[1].rsplit('_ob',1)[0]
        diagnostics.append(dict(case=case,**r));ci=r['wilson']
        diagnostic_table.append(f'| {r["method"]} | {case} | {r["ob"]}/{r["sf"]} | {r["success"]}/{r["n"]} | {ci[0]:.1%}–{ci[1]:.1%} | {json.dumps(r["failures"])} |')
    Path('metrics/fix_diagnostic_summary.json').write_text(json.dumps(diagnostics,indent=2))
    Path('metrics/fix_diagnostic_table.md').write_text('\n'.join(diagnostic_table)+'\n')
    failure_lines=['| 方法 | 数据集 | ob/sf | 失败分类 |','| --- | --- | --- | --- |']
    for r in rows:failure_lines.append(f'| {r["method"]} | {r["split"]} | {r["ob"]}/{r["sf"]} | {json.dumps(r["failures"])} |')
    Path('metrics/fix_failures_table.md').write_text('\n'.join(failure_lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(ob,sf) in zip(axes.flat,[(0,0),(0,1),(1,0),(1,1)]):
        for name in ['dreamer_s0','dreamer_s0_ft','ppo_s0','ppo_s0_ft']:
            d=index.get((name,'test',ob,sf))
            if d:
                ss=[r['progress_m'] for r in d['rows'] if r['failure']!='success']
                ax.hist(ss,bins=np.arange(0,221,5),histtype='step',label=name)
        ax.set(title=f'Mini-trail failures, ob={ob}, filter={sf}',xlabel='Route s (m)',ylabel='Episodes')
        ax.legend(fontsize=7)
    fig.savefig('figures/fix_failure_s.png',dpi=180);plt.close(fig)
    census=json.loads(Path('metrics/fix_census.json').read_text()) if Path('metrics/fix_census.json').exists() else None
    if census:
        keys=['right_rock','either_opposing_rock','triple'];fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
        for i,(name,d) in enumerate(census.items()):
            ax.bar(np.arange(len(keys))+(i-1)*.25,[d['counts'][k]/d['n'] for k in keys],.25,label=name)
        ax.set_xticks(np.arange(len(keys)),keys);ax.set_ylabel('Map fraction');ax.legend();fig.savefig('figures/fix_combinations.png',dpi=180);plt.close(fig)
    initial=json.loads(Path('metrics/fix_initial_checkpoints.json').read_text()) if Path('metrics/fix_initial_checkpoints.json').exists() else {}
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);curves={}
    for method in ['dreamer','ppo']:
        name=f'{method}_s0_ft';events=[];log=Path(f'metrics/train_{name}.jsonl')
        if log.exists():events=[json.loads(x) for x in log.read_text().splitlines()]
        points=[];base=initial.get(f'{method}_s0',{}).get('source_progress',{}).get('wall_seconds',0)
        for p in Path('metrics').glob(f'fix_curve_{name}_*.json'):
            steps=int(p.stem.rsplit('_',1)[1]);d=json.loads(p.read_text());matching=[r for r in events if r.get('event')=='end' and r['steps']==steps]
            if matching:points.append(dict(hours=(matching[-1]['wall_seconds']-base)/3600,steps=steps,rate=d['success_rate'],gates=d['mean_gates']))
        p=Path(f'metrics/fix_curve_initial_{method}_s0.json')
        if p.exists():
            d=json.loads(p.read_text());points.append(dict(hours=0,steps=initial[f'{method}_s0']['source_progress']['steps'],rate=d['success_rate'],gates=d['mean_gates']))
        points.sort(key=lambda p:p['hours']);curves[method]=points
        for ax,key in zip(axes,['rate','gates']):ax.plot([p['hours'] for p in points],[p[key] for p in points],'-o',label=method)
    axes[0].set(ylabel='All 8 gates success rate',ylim=(0,1.05));axes[1].set(ylabel='Mean gates',ylim=(0,8.2))
    for ax in axes:ax.set_xlabel('Fine-tuning wall time (h)');ax.legend();ax.grid(alpha=.2)
    fig.savefig('figures/fix_learning_curves.png',dpi=180);plt.close(fig)
    Path('metrics/fix_learning_curves.json').write_text(json.dumps(curves,indent=2))
    print(json.dumps(dict(conditions=len(index),paired_comparisons=len(comparisons))),flush=True)


if __name__=='__main__':main()
