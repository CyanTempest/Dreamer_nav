"""Original-protocol endpoint failure visualization."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

m=np.load('maps_lwy/test_862428/map.npz')
d=json.loads(Path('metrics/r8_p3_diagnosis.json').read_text())
t=np.load('metrics/r8_p3_diagnosis.npz')['trace']
end=m['route'][-1,:2]
x=np.arange(end[0]-12,end[0]+1,.025);y=np.arange(end[1]-3,end[1]+2,.025)
xx,yy=np.meshgrid(x,y);xy=np.stack([xx,yy],-1)
ij=np.floor((xy-m['origin'])/m['res']+.5).astype(int);ti=ij//32
lookup=np.full(m['dimensions'],-1,int)
lookup[m['tile_indices'][:,0],m['tile_indices'][:,1]]=np.arange(len(m['tiles']))
ids=lookup[ti[...,0],ti[...,1]]
assert (ids>=0).all()
edge=m['tiles'][ids,ij[...,0]%32,ij[...,1]%32,2]
fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
ax.contourf(xx,yy,edge,levels=[-100,.025,100],colors=['#e2f0dc','#f4e5df'])
ax.contour(xx,yy,edge,levels=[.025],colors=['#b13b27'],linewidths=1)
r=m['route'][m['s']>=987]
ax.plot(r[:,0],r[:,1],'k--',lw=1,label='Route center')
for i,row in d['rows'].items():
    if row['s']<990:continue
    tr=t[:int(row['steps']),int(i)]
    tr=tr[tr[:,3]>=988]
    ax.plot(tr[:,0],tr[:,1],color='#2166ac',alpha=.4,lw=1)
    pos=np.array(row['pos']);a=row['yaw'];rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    foot=np.array([[-.515,-.29],[.515,-.29],[.515,.29],[-.515,.29]])@rot.T+pos
    ax.add_patch(Polygon(foot,fill=False,ec='#2166ac',alpha=.5,lw=.8))
for station,color,label in [(1000,'#c72121','Original finish: 1000 m (unreachable)'),(999.15,'#207a39','Revised arrival: 999.15 m')]:
    point=np.array([np.interp(station,m['s'],m['route'][:,j]) for j in range(2)])
    a=np.interp(station,m['s'],m['yaw']);normal=np.array([-np.sin(a),np.cos(a)])
    line=point+np.array([[-1],[1]])*normal
    ax.plot(line[:,0],line[:,1],color=color,lw=2,label=label)
ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='World X [m]',ylabel='World Y [m]',title='R8: 15 terminal failures on original task; body footprints cross the side boundary')
ax.set_aspect('equal');ax.legend(loc='upper left',fontsize=8)
fig.savefig('figures/r8_p3_endpoint_failures.png',dpi=150)
