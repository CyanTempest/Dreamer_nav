"""Summary plots from saved diagnostic statistics; no model inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    summary=json.loads(Path('metrics/diag_summary.json').read_text());rows=summary['counterfactuals']
    names=['Original','Remove log','Remove right rocks','Shift log left .30 m','Rotate log 30 deg','Complete obstacle input']
    y=np.array([r['rate'] for r in rows]);ci=np.array([r['collision_wilson_95'] for r in rows])
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.barh(names,y*100,color=['#bd4937','#4c8b6c','#4c8b6c','#4c8b6c','#b88943','#bd4937'])
    ax.errorbar(y*100,np.arange(len(y)),xerr=np.stack([y-ci[:,0],ci[:,1]-y])*100,fmt='none',ecolor='black',capsize=3)
    for i,r in enumerate(rows):ax.text(min(y[i]*100+4,95),i,f"{r['collisions']}/{r['n']}",va='center')
    ax.set(xlim=(0,110),xlabel='Collision rate (%) / Wilson 95% CI',title='Dreamer, s=150 to 171 m; 2 scenario seeds × 30 episodes');ax.invert_yaxis()
    fig.savefig('figures/diag_counterfactuals.png',dpi=170);plt.close(fig)
    c=json.loads(Path('metrics/diag_training_census.json').read_text());keys=['log_length_within_5cm','angle_within_5deg','width_within_10cm','curvature_within_005','gate_within_1m','fork_within_15m','opposing_rock_3_5m']
    labels=['Log length ±5 cm','Log angle ±5°','Trail width ±10 cm','Curvature ±.005/m','Gate within 1 m','Fork within 15 m','Right rock within 3.5 m']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    counts=[c['counts'][k] for k in keys];axes[0].barh(labels,counts,color=['#426f8d']*6+['#bd4937'])
    for i,v in enumerate(counts):axes[0].text(v+2,i,str(v),va='center')
    axes[0].set(xlim=(0,230),xlabel='Training maps / 200',title='Matching features: mini-trail log neighborhood');axes[0].invert_yaxis()
    axes[1].hist([r['s'] for r in c['rows']],bins=18,color='#426f8d',alpha=.8);axes[1].axvline(c['mini']['s'],color='#bd4937',label='Mini-trail log')
    axes[1].set(xlabel='Generated log station s (m)',ylabel='Maps',title='Original 200 training maps');axes[1].legend()
    fig.savefig('figures/diag_training_features.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
