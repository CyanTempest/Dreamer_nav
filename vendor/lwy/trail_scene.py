"""Shared metre-scale geometry for MuJoCo contacts and Cycles RGB-D rendering.

World is right-handed Z-up. s is the 3-D arclength of a smooth survey guide,
not world X. Stair risers discretize that guide and do not change its stationing.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np


def zone_at(spec, s):
    zones = spec['track']['zones']
    for i, z in enumerate(zones):
        if z['start_m'] <= s < z['end_m'] or (i == len(zones)-1 and s == z['end_m']):
            return z
    raise ValueError(f'No zone at station {s}')


class Route:
    def __init__(self, spec):
        self.spec = spec
        self.length = float(spec['length_m'])
        points = np.array(spec['track']['control_points_xyz'], float)
        if len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('Need at least four finite 3-D guide points')
        # Dense Catmull-Rom keeps curvature gradual, except marked entry flights
        # which stay straight. Resample by measured 3-D arclength afterwards.
        dense = []
        anchor_indices = []
        straight_segments = {int(f['start_anchor']) for f in spec['track'].get('stair_flights', [])
                             if 'start_anchor' in f and f['end_anchor'] == f['start_anchor'] + 1}
        for i in range(len(points)-1):
            anchor_indices.append(len(dense))
            p0,p1,p2,p3 = points[max(0,i-1)],points[i],points[i+1],points[min(len(points)-1,i+2)]
            n = max(8, int(np.linalg.norm(p2-p1)/.08))
            t = np.arange(n, dtype=float)[:,None]/n
            if i < 5 or i in straight_segments:
                segment = p1+(p2-p1)*t
            else:
                segment = .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)
            dense.extend(segment.tolist())
        anchor_indices.append(len(dense))
        dense.append(points[-1].tolist())
        dense = np.array(dense)
        station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
        self.scale = self.length/station[-1]
        self.xyz = dense*self.scale
        self.s = station*self.scale
        self.anchor_s = self.s[anchor_indices]
        self.stairs = []
        for flight in spec['track'].get('stair_flights', []):
            if 'start_anchor' in flight:
                start,end = self.anchor_s[flight['start_anchor']],self.anchor_s[flight['end_anchor']]
            else:
                start,end = flight['start_m'],flight['end_m']
            z0,z1 = self.guide(start)[2],self.guide(end)[2]
            count = int(flight.get('count', max(2, round(abs(z1-z0)/.15))))
            if count <= 0 or end <= start or abs(z1-z0) < .02:
                raise ValueError(f'Invalid staircase: {flight}')
            self.stairs.append(dict(start=float(start),end=float(end),count=count,
                                    rise=float((z1-z0)/count),tread_station=float((end-start)/count),
                                    z0=float(z0), label=flight['label']))

    def guide(self, s):
        return np.array([np.interp(s,self.s,self.xyz[:,i]) for i in range(3)])

    def frame(self, s):
        a,b = self.guide(max(0,s-.1)),self.guide(min(self.length,s+.1))
        t=b-a;t[2]=0;t/=np.linalg.norm(t)
        return t,np.array([-t[1],t[0],0.])

    def width(self, s):
        # Blend widths over 2 m rather than introduce a jagged edge at a zone boundary.
        zones=self.spec['track']['zones']
        width=float(zone_at(self.spec,s).get('width_m',self.spec['track']['default_width_m']))
        for a,b in zip(zones,zones[1:]):
            boundary=b['start_m']
            if abs(s-boundary)<1.:
                q=(s-boundary+1)/2
                q=q*q*(3-2*q)
                width=(1-q)*a['width_m']+q*b['width_m']
        return max(self.spec['track']['minimum_width_m'],width)

    def height(self,s):
        for f in self.stairs:
            if f['start'] <= s < f['end']:
                # First tread begins one rise above the lower landing.
                k=min(f['count'],int((s-f['start'])/f['tread_station'])+1)
                return f['z0']+k*f['rise']
        return float(self.guide(s)[2])

    def position(self,s,lateral=0., surface=True):
        p=self.guide(s);_,n=self.frame(s)
        p=p+n*lateral
        if surface:p[2]=self.height(s)
        return p

    def stations(self,step):
        if not math.isfinite(step) or step <= 0:raise ValueError('step must be positive')
        values=list(np.arange(0,self.length,step))+[self.length]
        values += [float(z['start_m']) for z in self.spec['track']['zones']]
        for f in self.stairs:
            values += np.linspace(f['start'],f['end'],f['count']+1).tolist()
        return np.unique(np.round(values,8))

    def bank_height(self,s,lateral):
        width=self.width(s);offset=max(0.,abs(lateral)-width/2)
        zone=zone_at(self.spec,s)
        side=1 if lateral>0 else -1
        # Uphill bank on the right; downhill woodland on the left, as in the video.
        # +lateral points to camera-left; camera-right is -lateral.
        slope=zone.get('right_bank_slope',.65) if side<0 else zone.get('left_bank_slope',-.55)
        ripple=.10*math.sin(.87*s+2.1*offset)*math.sin(1.2*offset)
        # Hillsides follow the continuous survey guide. Only the narrow shared
        # road edge follows stair treads; the entire mountain must not become
        # a gigantic staircase when the path has steps.
        base=self.guide(s)[2]+(self.height(s)-self.guide(s)[2])*math.exp(-offset/.15)
        return base+slope*min(offset,7.)+ripple*min(offset,1.)

    def bank_position(self,s,lateral):
        # A normal-offset ribbon folds whenever its width exceeds the bend
        # radius. The guide is longitudinal (monotonic X), so extend the banks
        # along world Y after the exact shared road edge instead. This keeps
        # the wide terrain strip injective even at tight stair landings.
        side=1 if lateral>=0 else -1;hw=self.width(s)/2;t,n=self.frame(s)
        p=self.guide(s)+n*side*hw
        p+=np.array([0.,side*max(0,abs(lateral)-hw)/max(t[0],.1),0.])
        p[2]=self.bank_height(s,lateral)
        return p


def prism(vertices, thickness=.5):
    """Exact triangular top, downward extrusion; always a small convex collider."""
    a=np.array(vertices,float)
    if np.cross(a[1]-a[0],a[2]-a[0])[2] < 0:a=a[[0,2,1]]
    b=a-[0,0,thickness]
    v=np.vstack([a,b])
    faces=np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[1,4,5],[1,5,2],[2,5,3],[2,3,0]])
    return v,faces


def build_bundle(spec,root,step=1.):
    route=Route(spec);ss=route.stations(step)
    groups={};colliders=[];primitives=[];plants=[]
    def surface(name,quad,material,uv,collision):
        q=np.array(quad,float)
        faces=np.array([[0,1,2],[0,2,3]])
        if np.cross(q[1]-q[0],q[2]-q[0])[2]<0:faces=faces[:,[0,2,1]]
        g=groups.setdefault(material,dict(vertices=[],faces=[],uv=[]))
        off=len(g['vertices']);g['vertices'].extend(q.tolist());g['faces'].extend((faces+off).tolist());g['uv'].extend(uv)
        if collision:
            for j,f in enumerate(faces):
                v,fs=prism(q[f]);colliders.append(dict(name=f'{name}_{j}',vertices=v.tolist(),faces=fs.tolist(),material=material))

    for idx,(s0,s1) in enumerate(zip(ss,ss[1:])):
        if s1-s0<1e-6:continue
        eps=min(1e-5,(s1-s0)*.001)
        c0,c1=route.position(s0+eps),route.position(s1-eps)
        _,n0=route.frame(s0);_,n1=route.frame(s1)
        w0,w1=route.width(s0)/2,route.width(s1)/2
        q=[c0-n0*w0,c1-n1*w1,c1+n1*w1,c0+n0*w0]
        z=zone_at(spec,(s0+s1)/2);mat=z['surface']
        surface(f'road_{idx}',q,mat,[[0,s0/2],[0,s1/2],[w1,s1/2],[w0,s0/2]],True)
        # Explicit vertical riser faces; no sloped ramp connecting stair treads.
        before=route.height(max(0,s0-eps));after=route.height(s0+eps)
        if abs(after-before)>.015:
            c=route.position(s0);p0=c-n0*w0;p1=c+n0*w0
            rv=[p0.copy(),p1.copy(),p1.copy(),p0.copy()]
            rv[0][2]=rv[1][2]=before;rv[2][2]=rv[3][2]=after
            surface(f'riser_{idx}',rv,'concrete',[[0,0],[1,0],[1,abs(after-before)],[0,abs(after-before)]],False)
        for side in [-1,1]:
            edges=[0.,.6,1.6,3.5,7.,12.,float(spec['environment']['side_extent_m'])]
            for j,(e0,e1) in enumerate(zip(edges,edges[1:])):
                quad=[]
                for st,n,hw,edge in [(s0,n0,w0,e0),(s1,n1,w1,e0),(s1,n1,w1,e1),(s0,n0,w0,e1)]:
                    p=route.bank_position(st,side*(hw+edge))
                    p[2]=route.bank_height(np.clip(st+ (eps if st==s0 else -eps),0,route.length),side*(hw+edge))
                    quad.append(p)
                surface(f'bank_{idx}_{side}_{j}',quad,'forest_floor' if j<3 else 'soil',
                            [[s0/2,e0/2],[s1/2,e0/2],[s1/2,e1/2],[s0/2,e1/2]],True)
                # Stair-stepped banks also require their vertical riser faces.
                # Without them, the sky shows through between adjacent treads.
                if abs(after-before)>.015:
                    seam=[]
                    for edge,station in [(e0,max(0,s0-eps)),(e1,max(0,s0-eps)),(e1,s0+eps),(e0,s0+eps)]:
                        point=route.bank_position(s0,side*(w0+edge))
                        point[2]=route.bank_height(station,side*(w0+edge))
                        seam.append(point)
                    if max(abs(seam[2][2]-seam[1][2]),abs(seam[3][2]-seam[0][2]))>.001:
                        surface(f'bank_riser_{idx}_{side}_{j}',seam,'forest_floor' if j<3 else 'soil',
                                [[0,0],[e1-e0,0],[e1-e0,abs(after-before)],[0,abs(after-before)]],False)

    def cylinder(name,a,b,r,mat,collision=True):
        primitives.append(dict(name=name,type='capsule',fromto=np.r_[a,b].tolist(),radius=r,material=mat,collision=collision))

    # Concrete cheek walls on the uphill side of stair flights. Their physical
    # boxes and rendered boxes are generated from precisely the same records.
    wall_ranges=[(route.stairs[0]['start'],route.stairs[1]['end']),
                 (route.stairs[-1]['start'],route.stairs[-1]['end'])]
    for fi,(begin,end) in enumerate(wall_ranges):
        stations=np.linspace(begin,end,math.ceil((end-begin)/.5)+1)
        for j,(s0,s1) in enumerate(zip(stations,stations[1:])):
            st=(s0+s1)/2;t,n=route.frame(st);pos=route.guide(st)-n*(route.width(st)/2+.09)+[0,0,.10]
            axis=route.guide(s1)-route.guide(s0);length=np.linalg.norm(axis);axis/=length
            primitives.append(dict(name=f'stair_wall_{fi}_{j}',type='box',position=pos.tolist(),size=[float(length/2+.002),.075,.35],axis_x=axis.tolist(),axis_y=n.tolist(),material='concrete_wall',collision=True))

    # Continuous physical rails, including both horizontal bars and close-spaced posts.
    for zi,z in enumerate(spec['track']['zones']):
        if not z.get('rail'):continue
        posts=np.linspace(z['start_m'],z['end_m'],max(2,math.ceil((z['end_m']-z['start_m'])/2.)+1))
        for side in z.get('rail_sides',[-1]):
            last=None
            for j,s in enumerate(posts):
                lateral=side*(route.width(s)/2+.09);base=route.position(s,lateral);base[2]=route.guide(s)[2]
                cylinder(f'rail_post_{zi}_{side}_{j}',base,base+[0,0,1.0],.027,'metal')
                bars=[base+[0,0,h] for h in [.48,1.0]]
                if last is not None:
                    for k,(a,b) in enumerate(zip(last,bars)):cylinder(f'rail_bar_{zi}_{side}_{j}_{k}',a,b,.022,'metal')
                last=bars

    rng=np.random.default_rng(int(spec['seed']))
    for i,s in enumerate(np.arange(0,route.length,2.)):
        zone=zone_at(spec,float(s));density=zone.get('vegetation_density',1.)
        for side in [-1,1]:
            # Thick understory beside the path, sparse patches in the exposed section.
            for j in range(7):
                if rng.random()>density:continue
                st=float(np.clip(s+rng.uniform(0,2),0,route.length));lat=side*(route.width(st)/2+.60+j*.60+rng.uniform(-.1,.3))
                p=route.bank_position(st,lat)
                typ='fern_02' if j==0 or rng.random()<.12 else 'shrub_04'
                height=float(rng.uniform(.25,.65) if typ=='fern_02' else rng.uniform(1.4,2.3) if j<4 else rng.uniform(.9,1.8))
                plants.append(dict(asset=typ,position=p.tolist(),yaw=float(rng.uniform(0,2*np.pi)),height=height,station=st))
            if i%2==0 and rng.random()<density:
                st=float(np.clip(s+rng.uniform(0,2),0,route.length));lat=side*(route.width(st)/2+rng.uniform(1.0,3.))
                p=route.bank_position(st,lat)
                plants.append(dict(asset='grass_medium_01',position=p.tolist(),yaw=float(rng.uniform(0,2*np.pi)),height=float(rng.uniform(.4,.8)),station=st))
            if rng.random()<.80*density:
                st=float(s);lat=side*(route.width(st)/2+rng.uniform(1.8,7.5));p=route.bank_position(st,lat)
                height=float(rng.uniform(3.5,6.5))
                plants.append(dict(asset='tree_small_02',position=p.tolist(),yaw=float(rng.uniform(0,2*np.pi)),height=height,station=st))
                cylinder(f'tree_trunk_collision_{i}_{side}',p,p+[0,0,height*.65],.12,'bark')
            if i%2==0 and rng.random()<.8*density:
                st=float(s);lat=side*(route.width(st)/2+rng.uniform(8.,17.));p=route.bank_position(st,lat)
                height=float(rng.uniform(7.,11.))
                plants.append(dict(asset='tree_small_02',position=p.tolist(),yaw=float(rng.uniform(0,2*np.pi)),height=height,station=st))
        if zone.get('rocks') and i%3==0:
            side=-1 if i%2 else 1;lat=side*(route.width(s)/2+.16);p=route.bank_position(float(s),lat)
            primitives.append(dict(name=f'rock_{i}',type='ellipsoid',position=(p+[0,0,.13]).tolist(),size=[.22,.15,.16],material='rock',collision=True))
        if zone.get('roots') and i%6==0:
            a=route.position(float(s),-.6);b=route.position(float(s)+.35,.6)
            cylinder(f'root_{i}',a+[0,0,.018],b+[0,0,.018],.025,'bark')

    # Reference blue trail signs are separate 3-D boards, not image overlays.
    for i,landmark in enumerate(spec['track'].get('landmarks',[])):
        s=landmark['s_m'];p=route.bank_position(s,-(route.width(s)/2+.45))
        cylinder(f'sign_pole_{i}',p,p+[0,0,1.8],.025,'metal')
        t,n=route.frame(s)
        primitives.append(dict(name=f'sign_board_{i}',type='box',position=(p+[0,0,1.65]).tolist(),size=[.025,.20,.15],yaw=math.atan2(t[1],t[0]),material='sign',collision=True))

    arrays={}
    for mat,g in groups.items():
        for k,v in g.items():arrays[f'{mat}_{k}']=np.array(v,dtype=np.int32 if k=='faces' else np.float32)
    scene_dir=Path(root)/'scene';scene_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(scene_dir/'trail_surfaces.npz',**arrays)
    bundle=dict(schema_version=2,coordinate_system='right-handed Z-up metres',route_length_m=route.length,
                spec_digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                route_guide_length_m=float(np.linalg.norm(np.diff(route.xyz,axis=0),axis=1).sum()),
                route_samples=len(ss),materials=list(groups),primitives=primitives,plants=plants,
                stairs=route.stairs,colliders=colliders,
                calibration=spec['calibration'],surface_file='trail_surfaces.npz')
    (scene_dir/'scene_bundle.json').write_text(json.dumps(bundle,separators=(',',':'))+'\n')
    return route,bundle,arrays
