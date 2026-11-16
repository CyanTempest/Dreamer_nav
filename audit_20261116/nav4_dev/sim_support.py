import numpy as np
rng=np.random.default_rng(0)
r=.125; n=64
x=(np.arange(n)+.5)*r
def stairs(t=.28,h=.17): return np.floor(x/t)*h
def med3(z):
    p=np.pad(z,1,mode='edge');s=[p[1+i:n+1+i,1+j:n+1+j] for i in (-1,0,1) for j in (-1,0,1)]
    return np.median(np.stack(s),0)
def support(z,maxstep=.2,grade=np.tan(np.radians(35))):
    p=np.pad(z,1,mode='edge')
    step=np.zeros_like(z)
    for i in (-1,0,1):
        for j in (-1,0,1):
            step=np.maximum(step,np.abs(p[1+i:n+1+i,1+j:n+1+j]-z))
    def mm(a,b): return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    fx=(p[2:,1:-1]-z)/r; bx=(z-p[:-2,1:-1])/r
    fy=(p[1:-1,2:]-z)/r; by=(z-p[1:-1,:-2])/r
    g=np.hypot(mm(fx,bx),mm(fy,by))
    return (step<=maxstep)&(g<=grade)
for name,prof in [('flat',0*x),('ramp20',np.tan(np.radians(20))*x),('ramp30',np.tan(np.radians(30))*x),('ramp33',np.tan(np.radians(33))*x),('ramp40',np.tan(np.radians(40))*x),('stairs.28/.17',stairs()),('stairs.25/.17',stairs(.25)),('stairs.30/.15',stairs(.30,.15)),('ledge.5',np.where(x>4,-.5,0))]:
    z=np.repeat(prof[:,None],n,1)
    for med in (0,1):
        fr=[]
        for t in range(20):
            zn=z+rng.normal(0,.025,z.shape)
            zz=med3(zn) if med else zn
            fr.append(1-support(zz).mean())
        print(name,'median' if med else 'raw   ', 'unsupported frac %.4f'%np.mean(fr), 'noiseless %.4f'%(1-support(med3(z) if med else z).mean()))
