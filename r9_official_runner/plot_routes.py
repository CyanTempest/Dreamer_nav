"""Plot route adjustment against unchanged geometric proxy masks."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon


def main():
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('qualified',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=json.loads((a.qualified/'manifest.json').read_text())['maps']
    if not records:raise ValueError('No qualified corridors')
    fig,axes=plt.subplots(1,len(records),figsize=(6*len(records),6),layout='constrained',squeeze=False)
    for ax,record in zip(axes[0],records):
        old=np.load(a.original/record['name']/'map.npz');new=np.load(a.qualified/record['name']/'map.npz')
        # Check that route adaptation did not change physics or atlas data.
        np.testing.assert_array_equal(old['layers'][...,[0,1,2,5,6]],new['layers'][...,[0,1,2,5,6]])
        np.testing.assert_array_equal(old['atlas'],new['atlas'])
        layer=old['layers'];mask=np.zeros(layer.shape[:2],np.uint8)
        mask[layer[...,2]<=.025]=1;mask[layer[...,5]<0]=2
        origin=old['origin'];end=origin+(np.array(mask.shape)-1)*.1
        ax.imshow(mask.T,origin='lower',extent=(origin[0]-.05,end[0]+.05,origin[1]-.05,end[1]+.05),cmap=ListedColormap(['#d8dce0','#a4d5a3','#986744']),vmin=0,vmax=2)
        ax.plot(old['route'][:,0],old['route'][:,1],color='#3678b5',lw=1,label='Scanner path excerpt')
        ax.plot(new['route'][:,0],new['route'][:,1],color='#b71919',lw=2,label='Qualified subsegment')
        for i in np.linspace(0,len(new['route'])-1,5).astype(int):
            yaw=new['yaw'][i];c,s=np.cos(yaw),np.sin(yaw)
            box=np.array([[-.515,-.29],[.515,-.29],[.515,.29],[-.515,.29]])@np.array([[c,s],[-s,c]])+new['route'][i,:2]
            ax.add_patch(Polygon(box,fill=False,edgecolor='#1a1a1a',lw=.8))
        lo=old['route'][:,:2].min(0)-2.5;hi=old['route'][:,:2].max(0)+2.5
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Local X (m)',ylabel='Local Y (m)',title=f"{record['name']}: {record['segment_length_m']:.1f} m admitted")
        ax.set_aspect('equal');ax.legend(fontsize=8,loc='best')
    fig.suptitle('Envelope-qualified local corridors; physical maps unchanged\nGreen: support proxy | Brown: collision proxy | Grey: unsupported / unknown')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
    print(json.dumps({'physics_and_atlas_unchanged':True,'qualified_maps':len(records),'output':str(a.output)}))


if __name__=='__main__':main()
