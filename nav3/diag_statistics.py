"""Paired map-bootstrap intervals for secondary heldout decoder metrics."""
import json
from pathlib import Path
import numpy as np


def main():
    rows=[]
    for ob in [0,1]:
        for sf in [0,1]:
            old=json.loads(Path(f'metrics/eval_dreamer_s0_heldout_ob{ob}_sf{sf}.json').read_text())
            new=json.loads(Path(f'metrics/diag_eval_margin_heldout_ob{ob}_sf{sf}.json').read_text())
            names=sorted(set(r['map'] for r in old['rows']))
            def counts(d):
                return np.array([[np.sum([r[k] for r in d['rows'] if r['map']==name],axis=0) for k in ['trav_intersection','trav_union']] for name in names])
            before,after=counts(old),counts(new)
            idx=np.random.default_rng(9042).integers(0,len(names),(5000,len(names)))
            b=before[idx].sum(1);a=after[idx].sum(1)
            for channel in range(3):
                bu=before[:,1,channel].sum();au=after[:,1,channel].sum()
                if min(au,bu)==0:continue
                delta=a[:,0,channel]/a[:,1,channel]-b[:,0,channel]/b[:,1,channel]
                rows.append(dict(obstacles=ob,filtered=sf,channel=['overall','occluded','random_near'][channel],
                    old=float(before[:,0,channel].sum()/bu),new=float(after[:,0,channel].sum()/au),
                    delta_95_map_bootstrap=np.quantile(delta,[.025,.975]).tolist()))
    Path('metrics/diag_iou_comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
