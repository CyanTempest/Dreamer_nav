import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def observation(bev,target,pred,path):
    image=bev.astype(float)/255
    height=np.where(image[...,3]<.5,(image[...,0]-.5)*4,np.nan)
    panels=[height,image[...,4],target,pred]
    titles=['Observed height (m)','Route corridor','Privileged traversability','Dreamer predicted traversability']
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained')
    for ax,arr,title in zip(axes,panels,titles):
        ax.imshow(arr,origin='lower',extent=(-4,4,-2,6),cmap='viridis',interpolation='nearest')
        ax.set(title=title,xlabel='Left (m)',ylabel='Forward (m)')
    fig.savefig(path,dpi=150);plt.close(fig)
