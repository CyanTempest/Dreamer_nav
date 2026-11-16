"""Diagnostic footprint-center connectivity near logs, with route-aligned yaw."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates,label


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='clearance_before');a=ap.parse_args()
    manifest=json.loads(Path('maps/manifest.json').read_text());rows=[]
    for rec in manifest['maps']:
        if rec['split']!='heldout':continue
        m=np.load(Path('maps')/rec['name']/'map.npz');layers=m['layers'];route=m['route'];arc=m['s']
        ss=rec['features']['log']+np.arange(-4,4.01,.1)
        widths=np.interp(ss,arc,m['width']);yaw=np.interp(ss,arc,m['yaw'])
        xy=np.stack([np.interp(ss,arc,route[:,k]) for k in range(2)],-1)
        normal=np.stack([-np.sin(yaw),np.cos(yaw)],-1)
        tangent=np.stack([np.cos(yaw),np.sin(yaw)],-1)
        offsets=np.arange(-1.4,1.401,.05)
        centers=xy[:,None]+offsets[None,:,None]*normal[:,None]
        fx,fy=np.meshgrid(np.linspace(-.515,.515,7),np.linspace(-.29,.29,5),indexing='ij')
        feet=centers[:,:,None]+fx.ravel()[None,None,:,None]*tangent[:,None,None]+fy.ravel()[None,None,:,None]*normal[:,None,None]
        ij=((feet-m['origin'])/.1).reshape(-1,2).T
        edge=map_coordinates(layers[...,2].astype(float),ij,order=0,mode='constant',cval=100).reshape(feet.shape[:-1])
        obstacle=map_coordinates(layers[...,5].astype(float),ij,order=0,mode='constant',cval=-100).reshape(feet.shape[:-1])
        free=(edge<=.025).all(-1)&(obstacle>=0).all(-1)
        labels,_=label(free,np.ones((3,3)))
        start=set(labels[0,abs(offsets)<.15])-{0};end=set(labels[-1,abs(offsets)<.15])-{0}
        rows.append(dict(map=rec['name'],connected=bool(start&end),log_s=float(rec['features']['log']),
                         min_width=float(widths.min()),min_free_centers=int(free.sum(-1).min())))
    out=dict(heading='route tangent; diagnostic sufficient-path test, not exhaustive SE(2) proof',rows=rows,connected=sum(r['connected'] for r in rows),maps=len(rows))
    Path(f'metrics/{a.name}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
