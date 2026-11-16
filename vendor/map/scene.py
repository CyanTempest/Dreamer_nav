"""Continuous-width mountain trail. All distances in metres; no policy code."""
from pathlib import Path
import argparse
import json
import os
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
os.environ.setdefault('__GL_SHADER_DISK_CACHE_PATH', str(ROOT / 'cache/nvidia'))
os.environ.setdefault('TMPDIR', str(ROOT / 'tmp'))
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree
from PIL import Image
import yaml


def smooth(t):
    """Quintic smoothstep: zero first AND second derivatives at either end."""
    t = np.clip(t, 0., 1.)
    return t**3 * (10 + t * (-15 + 6*t))


def window(s, start, end, transition=5.):
    return smooth((s-start)/transition) * smooth((end-s)/transition)


def project(points, path, arc=None):
    """Continuous segment projection, not rounded nearest-grid-cell lookup."""
    points = np.asarray(points).reshape(-1, 2)
    if arc is None:
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    _, idx = cKDTree(path).query(points, workers=4)
    result = []
    for k in [np.clip(idx-1, 0, len(path)-2), np.clip(idx, 0, len(path)-2)]:
        v = path[k+1]-path[k]
        u = np.clip(np.sum((points-path[k])*v, axis=1)/np.sum(v*v, axis=1), 0, 1)
        q = points-path[k]-u[:, None]*v
        dist = np.linalg.norm(q, axis=1)
        signed = np.sign(v[:, 0]*q[:, 1]-v[:, 1]*q[:, 0])*dist
        result.append((dist, arc[k]+u*(arc[k+1]-arc[k]), signed))
    use = result[0][0] <= result[1][0]
    return tuple(np.where(use, a, b) for a, b in zip(*result))


def corridor_boundary(path,width):
    """Closed offset curves with round endcaps; explicit continuous boundary."""
    delta=np.gradient(path,axis=0)
    yaw=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
    normal=np.c_[-np.sin(yaw),np.cos(yaw)]
    left=path+normal*width[:,None]/2;right=path-normal*width[:,None]/2
    a=np.linspace(yaw[-1]+np.pi/2,yaw[-1]-np.pi/2,81)
    end=path[-1]+width[-1]/2*np.c_[np.cos(a),np.sin(a)]
    a=np.linspace(yaw[0]-np.pi/2,yaw[0]-3*np.pi/2,81)
    start=path[0]+width[0]/2*np.c_[np.cos(a),np.sin(a)]
    p=np.concatenate([left,end[1:-1],right[::-1],start[1:-1],left[:1]])
    return p


def corridor_edge(points,path,width):
    from matplotlib.path import Path as PolygonPath
    boundary=corridor_boundary(path,np.asarray(width))
    distance,_,_=project(points,boundary)
    inside=PolygonPath(boundary).contains_points(points)
    return np.where(inside,-distance,distance)


def write_ply(path, points, colors=None):
    points = np.asarray(points, dtype='<f4')
    with open(path, 'wb') as f:
        header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
        if colors is None:
            f.write((header+'end_header\n').encode()); f.write(points.tobytes())
        else:
            f.write((header+'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
            a = np.empty(len(points), dtype=[('xyz','<f4',3),('rgb','u1',3)])
            a['xyz'] = points; a['rgb'] = colors
            f.write(a.tobytes())


class Scene:
    def __init__(self, seed=0, mini=True, res=.1):
        self.seed, self.mini, self.res = seed, mini, res
        self.rng = np.random.default_rng(871002 if mini else seed)
        self.length = 206. if mini else float(self.rng.uniform(198, 218))
        self.s = np.linspace(0, self.length, round(self.length/.05)+1)
        shift = 0. if mini else float(self.rng.uniform(-2, 2))
        self.bend_s = 82.625+shift
        # Integrated heading gives a regular curve (no spline overshoot/cusps).
        yaw = np.deg2rad(12 - 32*smooth((self.s-26)/24)
                        + 90*smooth((self.s-76-shift)/13.25)
                        - 85*smooth((self.s-112)/27)
                        + 35*smooth((self.s-165)/25))
        self.yaw = yaw
        self.xy = np.c_[cumulative_trapezoid(np.cos(yaw), self.s, initial=0),
                        cumulative_trapezoid(np.sin(yaw), self.s, initial=0)]
        if mini:
            self.features = dict(narrow=99., stairs_up=18.5, stairs_down=180., fork_tape=65., fork_plain=151., leaves=190., log=164.)
        else:
            slots = np.array([17., 67., 99., 150., 176., self.length-11.])
            self.rng.shuffle(slots)
            slots += self.rng.uniform(-1.0, 1.0, len(slots))
            self.features = dict(zip(['narrow','stairs_up','stairs_down','fork_tape','fork_plain'], map(float, slots[:5])))
            self.features.update(leaves=float(slots[5]), log=float(self.length-42))
        self.features.update(bend=self.bend_s, uphill=48., downhill=129.)
        self.width = self.width_at(self.s)
        grade = np.tan(np.deg2rad(15))*(window(self.s, 35, 61)-window(self.s, 116, 142))
        self.z_ramp = cumulative_trapezoid(grade, self.s, initial=0)
        self.step_positions = []
        self.z_steps = np.zeros_like(self.s)
        for name, sign in [('stairs_up',1), ('stairs_down',-1)]:
            c = self.features[name]
            for j in range(4):
                sj = c-1.0+j*.65
                self.step_positions.append(sj)
                self.z_steps += sign*.15*(self.s >= sj)
        self.z_smooth_steps = gaussian_filter1d(self.z_steps, 2.5/(self.s[1]-self.s[0]), mode='nearest')
        self.z = self.z_ramp+self.z_steps
        self.route = np.c_[self.xy, self.z]
        self.arc3d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))]
        self.gate_idx = np.searchsorted(self.s, self.length*np.array([.08,.20,.32,.44,.56,.68,.80,.95]))
        self.branches = []
        for name, side in [('fork_tape',1), ('fork_plain',-1)]:
            start = self.features[name]-3
            bs = np.linspace(start, start+14, 281)
            base = self.xy_at(bs)
            byaw = np.interp(bs, self.s, self.yaw)
            offset = side*4.6*smooth((bs-start)/14)
            path = base+offset[:,None]*np.c_[-np.sin(byaw), np.cos(byaw)]
            self.branches.append(dict(name=name, start_s=start, path=path, main_s=bs, side=side))
        points = np.concatenate([self.xy]+[b['path'] for b in self.branches])
        lo = np.floor((points.min(0)-10)/res)*res
        hi = np.ceil((points.max(0)+10)/res)*res
        self.origin = lo
        self.x = np.arange(round((hi[0]-lo[0])/res)+1)*res+lo[0]
        self.y = np.arange(round((hi[1]-lo[1])/res)+1)*res+lo[1]
        X,Y = np.meshgrid(self.x,self.y,indexing='ij')
        p = np.c_[X.ravel(),Y.ravel()]
        dist, ss, signed = project(p,self.xy,self.s)
        edge = corridor_edge(p,self.xy,self.width)
        self.main_edge = edge.reshape(X.shape).astype(np.float32)
        for b in self.branches:
            ba=np.r_[0,np.cumsum(np.linalg.norm(np.diff(b['path'],axis=0),axis=1))]
            bw = 2.6-.95*smooth(ba/5)
            branch_edge=corridor_edge(p,b['path'],bw)
            # C1 smooth union rounds the fork crotch without overwriting height.
            h=np.maximum(.35-np.abs(edge-branch_edge),0)/.35
            edge=np.minimum(edge,branch_edge)-h*h*.35*.25
        edge = edge.reshape(X.shape)
        self.edge = edge.astype(np.float32)
        self.support = edge <= 0
        self.projected_s = ss.reshape(X.shape).astype(np.float32)
        outside = smooth(edge/2.8)
        # Global x is monotone: all base profiles are single-valued and continuous.
        # Steps alone are intentional jumps, confined to the road/shoulder influence.
        zr_base = np.broadcast_to(gaussian_filter1d(np.interp(self.x,self.xy[:,0],self.z_ramp),6/self.res)[:,None],X.shape)
        zr = zr_base+(1-smooth(edge/6.))*(np.interp(self.projected_s,self.s,self.z_ramp)-zr_base)
        zs = np.interp(self.projected_s,self.s,self.z_steps)
        zsm = np.interp(X,self.xy[:,0],self.z_smooth_steps)
        side = Y-np.interp(X,self.xy[:,0],self.xy[:,1])
        bank = .9*np.tanh(-side/3.5)+.24*np.sin(X/8)*np.cos(Y/6)
        ns = self.features['narrow']
        # A globally single-valued field avoids nearest-curve cut-locus jumps.
        dx0=float(self.xy_at(ns-14)[0])-2;dx1=float(self.xy_at(ns+14)[0])+2
        drop = window(X,dx0,dx1,5)
        self.drop_weight = (drop*smooth(side/.8)*smooth(edge/3.5)*(1-smooth((edge-4)/5))).astype(np.float32)
        rough = .007*np.sin(X*3.2)*np.cos(Y*2.4)
        self.ground = (zr+zsm+(1-outside)*(zs-zsm)+outside*bank-2.1*self.drop_weight+rough).astype(np.float32)
        self.leaf = ((edge>0)&(edge<.85)&(np.abs(self.projected_s-self.features['leaves'])<4)&(side>0))
        # Leaves are a visual shoulder annotation; never counted as trail support.
        self.albedo = self.make_albedo(X,Y,edge,outside)
        self.props = []
        self.make_props()

    def width_at(self,s):
        w = np.zeros_like(np.asarray(s),dtype=float)+2.6
        w -= .45*(window(s,33,63,5)+window(s,114,144,5))
        for name in ['stairs_up','stairs_down']:
            c = self.features[name]
            blend = window(s,c-7,c+7,5.)
            w = w*(1-blend)+2.2*blend
        n = self.features['narrow']
        blend = window(s,n-9,n+9,5.)
        w = w*(1-blend)+1.08*blend
        return w

    def xy_at(self,s):
        return np.stack([np.interp(s,self.s,self.xy[:,i]) for i in range(2)],axis=-1)

    def pose_at(self,s):
        xy = self.xy_at(s)
        yaw = float(np.interp(s,self.s,self.yaw))
        z = float(self.elevation(*xy))
        # Axle-span grade avoids instantaneous body pitching on a single riser.
        a,b = self.xy_at(np.clip([s-.42,s+.42],0,self.length))
        pitch = float(np.arctan2(self.elevation(*b)-self.elevation(*a),.84))
        return np.array([*xy,z,yaw,pitch])

    def elevation(self,x,y):
        x,y = np.broadcast_arrays(x,y)
        coords = np.array([(x-self.origin[0])/self.res,(y-self.origin[1])/self.res])
        return map_coordinates(self.ground,coords.reshape(2,-1),order=1,mode='nearest').reshape(x.shape)

    def clearance(self,points):
        return -corridor_edge(np.asarray(points).reshape(-1,2),self.xy,self.width)

    def make_albedo(self,X,Y,edge,outside):
        rng = self.rng
        coarse = .035*np.sin(X*1.8)*np.cos(Y*2.3)+.022*np.sin(X*7+Y*5)
        noise = rng.normal(0,.024,X.shape)+coarse
        forest = np.array([.29,.35,.19])
        earth = np.array([.54,.43,.30])
        shoulder = np.array([.40,.32,.19])
        road = 1-smooth((edge+.06)/.16)
        leafband = (1-road)*(1-smooth((edge-.20)/.75))
        rgb = forest[None,None,:]+noise[:,:,None]
        rgb = rgb*(1-leafband[:,:,None])+shoulder*leafband[:,:,None]
        rgb = rgb*(1-road[:,:,None])+(earth+noise[:,:,None]*.6)*road[:,:,None]
        flecks = rng.random(X.shape)
        mulch = ((leafband>.15)|(self.leaf))&(flecks>.74)
        rgb[mulch] = np.array([.58,.38,.16])+noise[mulch,None]
        stone = np.zeros(X.shape,bool)
        for name in ['stairs_up','stairs_down']:
            stone |= (np.abs(self.projected_s-self.features[name])<1.5)&(edge<.08)
        rgb[stone] = np.array([.48,.46,.39])+noise[stone,None]*.6
        return np.uint8(np.clip(rgb,0,1)*255)

    def add(self,kind,shape,pos,size,rgba,**extra):
        self.props.append(dict(kind=kind,shape=shape,pos=list(map(float,pos)),size=list(map(float,size)),rgba=list(rgba),**extra))

    def make_props(self):
        rng = self.rng
        for flight in ['stairs_up','stairs_down']:
            for j in range(4):
                s=self.features[flight]-1+j*.65+.325
                xy=self.xy_at(s);yaw=float(np.interp(s,self.s,self.yaw))
                z=float(np.interp(s,self.s,self.z))+.015
                self.add('stone_tread','box',[*xy,z-.075],[.324,float(self.width_at(s))/2,.075],[.48,.46,.39,1],yaw=yaw)
        # Two offset lines plus sparse outer groves; crowns do not hide the path.
        for s in np.arange(4,self.length-2,2.4):
            yaw = np.interp(s,self.s,self.yaw); n=np.array([-np.sin(yaw),np.cos(yaw)])
            for side in [-1,1]:
                if rng.random()<.24: continue
                xy=self.xy_at(s)+side*n*rng.uniform(3.0,8.0)
                if self.clearance([xy])[0]>-.9: continue
                z=float(self.elevation(*xy)); h=rng.uniform(2.4,4.8)
                self.add('tree','cylinder',[*xy,z+h*.46],[rng.uniform(.09,.16),h*.46],[.23,.16,.09,1])
                for level in range(3):
                    self.add('canopy','mesh',[*xy,z+h*(.54+.18*level)],[1],
                             [float(rng.uniform(.15,.23)),float(rng.uniform(.27,.37)),.12,1],
                             mesh='pine',scale=[h*(.25-.045*level)]*2+[h*.41])
        n=self.features['narrow']
        rail_s=np.arange(n-9,n+9+.1,.85)
        tops=[]
        for s in rail_s:
            yaw=np.interp(s,self.s,self.yaw); normal=np.array([-np.sin(yaw),np.cos(yaw)])
            xy=self.xy_at(s)+normal*(self.width_at(s)/2+.13)
            z=float(self.elevation(*xy)); tops.append([*xy,z+.69])
            self.add('rail_post','cylinder',[*xy,z+.40],[.018,.40],[.11,.35,.20,1])
        for a,b in zip(tops[:-1],tops[1:]):
            self.add('rail_bar','capsule',[0,0,0],[.016],[.11,.35,.20,1],fromto=a+b)
        b=self.branches[0]; j=150
        xy=b['path'][j]; direction=b['path'][j+1]-b['path'][j-1];direction/=np.linalg.norm(direction)
        normal=np.array([-direction[1],direction[0]])
        base=float(self.elevation(*xy)); a=xy-normal*.94;c=xy+normal*.94
        # 4.5 cm vertical ribbon breadth, 4 mm thickness; centreline 0.40 m AGL.
        self.add('suspended_black_tape','box',[*xy,base+.40],[.94,.002,.0225],[.018,.018,.016,1],yaw=float(np.arctan2(normal[1],normal[0])))
        for q in [a,c]:
            self.add('tape_stake','cylinder',[*q,base+.32],[.014,.32],[.35,.24,.13,1])
        self.features['tape_view']=b['start_s']+6
        for index,k in enumerate(self.gate_idx):
            s=self.s[k];yaw=self.yaw[k];normal=np.array([-np.sin(yaw),np.cos(yaw)])
            half=self.width[k]/2+.20
            for side in [-1,1]:
                xy=self.xy[k]+normal*side*half;z=float(self.elevation(*xy))
                self.add('gate_post','cylinder',[*xy,z+.75],[.035,.75],[.68,.45,.20,1])
                self.add('gate_marker','box',[*xy,z+1.5],[.015,.16,.12],[.84,.72,.46,1],yaw=float(yaw),gate=index+1)
        # Rocks deliberately visible at the path edge; no traversability claim.
        for s in list(np.linspace(9,self.length-6,24))+[self.features['log']+1.4]:
            if self.width_at(s)<1.5: continue
            yaw=np.interp(s,self.s,self.yaw);normal=np.array([-np.sin(yaw),np.cos(yaw)])
            xy=self.xy_at(s)+normal*rng.choice([-1,1])*rng.uniform(.85,1.5)
            h=rng.uniform(.16,.34);z=float(self.elevation(*xy))
            self.add('rock','ellipsoid',[*xy,z+h*.6],[rng.uniform(.2,.4),rng.uniform(.18,.32),h],[.38,.37,.32,1],yaw=float(rng.uniform(-3,3)))
        s=self.features['log'];yaw=np.interp(s,self.s,self.yaw);normal=np.array([-np.sin(yaw),np.cos(yaw)])
        xy=self.xy_at(s)+normal*.86
        a=xy-normal*.8; b=xy+normal*.8
        a3=[*a,float(self.elevation(*a))+.14];b3=[*b,float(self.elevation(*b))+.14]
        self.add('fallen_log','cylinder',[0,0,0],[.14],[.29,.18,.08,1],fromto=a3+b3)
        # Bark ridges follow the log, visibly distinct from an untextured capsule.
        for t in np.linspace(0,1,10):
            p=(1-t)*np.array(a3)+t*np.array(b3)
            self.add('bark','sphere',p,[.145],[.25,.14,.07,1])
        # A patch of individually modelled leaves, all lying on the shoulder.
        for _ in range(280):
            s=self.features['leaves']+rng.uniform(-4,4)
            yaw=np.interp(s,self.s,self.yaw);normal=np.array([-np.sin(yaw),np.cos(yaw)])
            xy=self.xy_at(s)+normal*(self.width_at(s)/2+rng.uniform(.04,.82))
            z=float(self.elevation(*xy))+.009
            self.add('leaf','ellipsoid',[*xy,z],[rng.uniform(.025,.055),.018,.003],
                     [float(rng.uniform(.35,.64)),float(rng.uniform(.22,.4)),.08,1],yaw=float(rng.uniform(-3,3)))

    def metadata(self):
        # Keep the previous convert_scan.py schema, with extra annotations.
        selected=np.unique(np.r_[np.arange(0,len(self.route),4),self.gate_idx,len(self.route)-1])
        return dict(frame='map',units='m',directed=True,seed=self.seed,mini=self.mini,
                    points=self.route[selected].tolist(),arc_length_m=self.s[selected].tolist(),
                    corridor_width_m=self.width[selected].tolist(),
                    gates=[dict(id=i+1,route_index=int(np.searchsorted(selected,k)),center=self.route[k].tolist(),width=float(self.width[k]),s_m=float(self.s[k])) for i,k in enumerate(self.gate_idx)],
                    branches=[b['path'].tolist() for b in self.branches],
                    graph=dict(nodes=['start']+[f'gate{i}' for i in range(1,9)],edges=[[i,i+1] for i in range(8)]),
                    features=self.features,length_m=float(self.arc3d[-1]),horizontal_length_m=self.length,
                    transition_m=5.,min_width_m=float(self.width.min()),
                    stairs=dict(riser_m=.15,tread_m=.65,risers_per_flight=4,positions_s_m=self.step_positions),
                    body_size_m=[.93,.48,.585],clearance_envelope_m=[1.03,.58],
                    sensor_mounts=dict(front=[.465,0,.54],rear=[-.465,0,.54],axes='hemisphere axes point +body_x / -body_x; assumed mounts'),
                    training_use=False)

    def export(self,directory):
        directory=Path(directory); assets=directory/'assets';assets.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'heightfield.npz',ground=self.ground,support=self.support,leaf=self.leaf,
                            origin=self.origin,res=self.res,route=self.route,s=self.s,width=self.width,
                            main_edge=self.main_edge,edge=self.edge,projected_s=self.projected_s,drop_weight=self.drop_weight)
        (directory/'route.yaml').write_text(yaml.safe_dump(self.metadata(),sort_keys=False),encoding='utf-8')
        (directory/'props.json').write_text(json.dumps(self.props,indent=2))
        with open(assets/'terrain.bin','wb') as f:
            f.write(struct.pack('<ii',len(self.y),len(self.x)))
            f.write(np.asarray(self.ground.T,'<f4').tobytes())
        # PNG rows top-down; texture pixels must correspond exactly to hfield UV.
        Image.fromarray(self.albedo.transpose(1,0,2)[::-1]).save(assets/'terrain_albedo.png')
        self.write_mjcf(directory)
        X,Y=np.meshgrid(self.x,self.y,indexing='ij')
        write_ply(directory/'terrain.ply',np.c_[X.ravel(),Y.ravel(),self.ground.ravel()],self.albedo.reshape(-1,3))
        return directory

    def write_mjcf(self,directory):
        def elem(parent,tag,**kwargs):
            return ET.SubElement(parent,tag,{k:str(v) for k,v in kwargs.items()})
        def vec(v):return ' '.join(f'{float(x):.8g}' for x in v)
        root=ET.Element('mujoco',model='M1 mini trail | map-only kinematic scene')
        elem(root,'compiler',angle='radian',meshdir='assets',texturedir='assets')
        elem(root,'option',timestep='.01',gravity='0 0 -9.81')
        visual=elem(root,'visual')
        elem(visual,'global',offwidth='1600',offheight='1000',azimuth='110',elevation='-35')
        elem(visual,'quality',shadowsize='8192',offsamples='4')
        elem(visual,'map',znear='.0001',zfar='10',fogstart='1.6',fogend='4')
        elem(visual,'headlight',ambient='.35 .35 .32',diffuse='.5 .5 .46',specular='.05 .05 .05')
        elem(visual,'rgba',fog='.73 .79 .81 1',haze='.76 .82 .85 1')
        default=elem(root,'default');elem(default,'geom',friction='.9 .01 .001',condim='3')
        assets=elem(root,'asset')
        elem(assets,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.43 .59 .72',rgb2='.86 .89 .83',width='512',height='3072')
        elem(assets,'texture',name='terrain_color',type='2d',file='terrain_albedo.png')
        elem(assets,'material',name='terrain_mat',texture='terrain_color',texrepeat='1 1',texuniform='false',specular='0',shininess='0')
        from PIL import ImageDraw,ImageFont
        for gate in range(1,9):
            im=Image.new('RGB',(128,128),(215,193,137));draw=ImageDraw.Draw(im)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',74)
            draw.text((38,16),str(gate),font=font,fill=(35,49,33))
            draw.rectangle((4,4,123,123),outline=(61,72,42),width=4)
            im.save(directory/f'assets/gate_{gate}.png')
            elem(assets,'texture',name=f'gate_{gate}_tex',type='2d',file=f'gate_{gate}.png')
            elem(assets,'material',name=f'gate_{gate}_mat',texture=f'gate_{gate}_tex',texuniform='false',specular='0')
        zmin=float(self.ground.min());zrange=float(np.ptp(self.ground))
        elem(assets,'hfield',name='terrain',file='terrain.bin',size=vec([(self.x[-1]-self.x[0])/2,(self.y[-1]-self.y[0])/2,zrange,1]))
        # A closed low-poly pine crown; no convex-collision approximation of terrain.
        verts=[[0,0,1]]+[[np.cos(a),np.sin(a),0] for a in np.linspace(0,2*np.pi,13)[:-1]]+[[0,0,-.08]]
        faces=[]
        for j in range(12):faces.extend([[0,1+j,1+(j+1)%12],[13,1+(j+1)%12,1+j]])
        pine=directory/'assets/pine.obj'
        pine.write_text('\n'.join(['v '+vec(v) for v in verts]+['f '+' '.join(str(k+1) for k in f) for f in faces])+'\n')
        # Each scaled mesh is an asset: MuJoCo mesh scale belongs to the asset.
        for i,p in enumerate(self.props):
            if p['shape']=='mesh':elem(assets,'mesh',name=f'pine_{i}',file='pine.obj',scale=vec(p['scale']))
        world=elem(root,'worldbody')
        elem(world,'light',name='sun',pos='20 -30 70',dir='-.3 .4 -1',directional='true',diffuse='.8 .77 .67',ambient='.18 .2 .18',castshadow='true')
        elem(world,'geom',name='terrain',type='hfield',hfield='terrain',material='terrain_mat',pos=vec([(self.x[-1]+self.x[0])/2,(self.y[-1]+self.y[0])/2,zmin]))
        for i,p in enumerate(self.props):
            kw=dict(name=f"{p['kind']}_{i}",type=p['shape'],rgba=vec(p['rgba']),group='0')
            if p['shape']=='mesh':kw['mesh']=f'pine_{i}'
            else:kw['size']=vec(p['size'])
            if 'fromto' in p:kw['fromto']=vec(p['fromto'])
            else:kw['pos']=vec(p['pos'])
            if 'yaw' in p:kw['euler']=f"0 0 {p['yaw']}"
            if 'gate' in p:kw['material']=f"gate_{p['gate']}_mat";kw['rgba']='1 1 1 1'
            elem(world,'geom',**kw)
            if 'gate' in p:
                # Named sites can be toggled in the interactive viewer.
                elem(world,'site',name=f"G{p['gate']}_{i}",pos=vec(np.array(p['pos'])+[0,0,.2]),size='.01',rgba='1 .9 .6 0')
        pose=self.pose_at(2)
        body=elem(world,'body',name='m1',mocap='true',pos=vec(pose[:3]))
        elem(body,'geom',name='m1_chassis',type='box',size='.40 .185 .12',pos='0 0 .40',rgba='.73 .76 .73 1',group='1')
        elem(body,'geom',name='m1_top',type='box',size='.28 .17 .0325',pos='0 0 .5525',rgba='.15 .20 .22 1',group='1')
        elem(body,'geom',name='m1_front',type='box',size='.009 .14 .05',pos='.407 0 .42',rgba='.91 .55 .11 1',group='1')
        for i,(x,y) in enumerate([(x,y) for x in [-.345,.345] for y in [-.195,.195]]):
            elem(body,'geom',name=f'leg_{i}',type='capsule',fromto=vec([x*.7,y*.8,.36,x,y,.12]),size='.033',rgba='.26 .29 .29 1',group='1')
            elem(body,'geom',name=f'wheel_{i}',type='cylinder',pos=vec([x,y,.12]),size='.12 .045',euler='1.5707963 0 0',rgba='.055 .065 .065 1',group='1')
            elem(body,'geom',name=f'hub_{i}',type='cylinder',pos=vec([x,np.sign(y)*.242,.12]),size='.058 .003',euler='1.5707963 0 0',rgba='.65 .69 .68 1',group='1')
        for name,side,color in [('front',1,'.06 .72 .83 1'),('rear',-1,'.93 .31 .12 1')]:
            elem(body,'geom',name=f'airy_{name}',type='cylinder',pos=f'{side*.465} 0 .54',size='.03 .0315',euler=f'0 {side*np.pi/2} 0',rgba=color,group='2')
            elem(body,'site',name=f'lidar_{name}',pos=f'{side*.465} 0 .54',size='.008',rgba=color,group='2')
        elem(body,'camera',name='front_rgb',pos='.425 0 .555',xyaxes='0 -1 0 0.139173 0 0.990268',fovy='68')
        elem(body,'camera',name='rear_rgb',pos='-.425 0 .555',xyaxes='0 1 0 -0.139173 0 0.990268',fovy='68')
        ET.indent(root)
        ET.ElementTree(root).write(directory/'scene.xml',encoding='utf-8',xml_declaration=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);ap.add_argument('--random',action='store_true');ap.add_argument('--res',type=float,default=.1);ap.add_argument('--output',type=Path)
    a=ap.parse_args();directory=a.output or ROOT/'maps'/ (f'random_{a.seed:03d}' if a.random else 'mini_trail')
    scene=Scene(a.seed,not a.random,a.res);scene.export(directory)
    print(json.dumps(dict(output=str(directory),length_m=float(scene.arc3d[-1]),shape=list(scene.ground.shape),min_width=float(scene.width.min()),props=len(scene.props)),indent=2))


if __name__=='__main__':main()
