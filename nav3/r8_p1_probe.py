"""Development ablations; configuration selection only on 27 probe poses."""
import numpy as np
from .r8_p1_validate import run,metrics,summary
from .r8_p1_common import ROOT,GEO,dump
from .r8_p1_online import Atlas
from .lwy_bank import LwyBank

def main():
    refs0=np.load(GEO/'r8_p1_references.npz');ids=np.flatnonzero(refs0['split']=='probe')
    refs={k:refs0[k][ids] if k!='local' else refs0[k] for k in refs0.files}
    bank=LwyBank('test',limit=1,root='maps_lwy');atlas=Atlas();rows=[]
    variants={
        'static_ground':{'p1_visible_ground':False},
        'terrain_horizon':{'p1_visible_ground':False,'p1_block_coverage':1.1},
        'visible_terrain_horizon':{'p1_visible_ground':True,'p1_block_coverage':1.1},
        'no_visibility':{'p1_visible_ground':False,'p1_visibility':False},
        'no_opening':{'p1_visible_ground':False,'p1_ground_open':False},
    }
    for name,cfg in variants.items():
        pred=run(bank,atlas,refs,cfg)
        r=dict(name=name,config=cfg,summary=summary([metrics(a,b) for a,b in zip(pred,refs['reference'])]))
        rows.append(r);print(name,r['summary'],flush=True)
    dump(ROOT/'metrics/r8_p1_development.json',dict(fit_indices=ids,holdout_used=False,rows=rows))

if __name__=='__main__':main()
