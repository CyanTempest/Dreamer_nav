"""Estimate pipe diameters from thin held-out axial slices of each reconstructed rail."""
import argparse
from collections import defaultdict
from copy import deepcopy
import xml.etree.ElementTree as ET
import json

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.compose import save_xml, vector
from nav4.matrix.survey import OUT


def perpendicular_basis(axis):
    e1=np.cross(axis,[0.,0.,1.])
    if np.linalg.norm(e1)<.10:
        e1=np.cross(axis,[0.,1.,0.])
    e1/=np.linalg.norm(e1)
    return np.column_stack([e1,np.cross(axis,e1)])


def estimate_circle(xy,rng):
    """RANSAC of circular arcs with angular and radial support checks."""
    if len(xy)<24:
        return None
    if len(xy)>520:
        xy=xy[rng.choice(len(xy),520,replace=False)]
    ix=rng.integers(0,len(xy),size=(1200,3))
    a,b,c=xy[ix].transpose(1,0,2)
    u,v=b-a,c-a
    aa=np.einsum("ij,ij->i",u,u)
    bb=np.einsum("ij,ij->i",v,v)
    det=2*(u[:,0]*v[:,1]-u[:,1]*v[:,0])
    ok=abs(det)>2e-4
    if not ok.any():
        return None
    u,v,a,aa,bb,det=u[ok],v[ok],a[ok],aa[ok],bb[ok],det[ok]
    center=a+np.c_[v[:,1]*aa-u[:,1]*bb,u[:,0]*bb-v[:,0]*aa]/det[:,None]
    radius=np.linalg.norm(a-center,axis=1)
    ok=(radius>.012)&(radius<.073)&(np.linalg.norm(center,axis=1)<.066)
    center,radius=center[ok],radius[ok]
    if not len(radius):
        return None
    rel=xy[None]-center[:,None]
    error=abs(np.linalg.norm(rel,axis=2)-radius[:,None])
    count=(error<.009).sum(axis=1)
    ranking=np.argsort(count/(radius/.035)**.37)[-16:][::-1]
    best=None
    for i in ranking:
        good=error[i]<.009
        if good.sum()<18:
            continue
        angles=np.mod(np.arctan2(rel[i,good,1],rel[i,good,0]),2*np.pi)
        sectors,count=np.unique(np.floor(angles/(np.pi/8)),return_counts=True)
        rich=(count>=3).sum()
        if rich<5:
            continue
        points=xy[good]
        def fun(w):
            return np.linalg.norm(points-w[:2],axis=1)-w[2]
        fit=least_squares(fun,np.r_[center[i],radius[i]],loss="soft_l1",f_scale=.006)
        if not .013<fit.x[2]<.072 or np.linalg.norm(fit.x[:2])>.065:
            continue
        criterion=good.sum()*(.70+.06*min(rich,10))/(fit.x[2]/.035)**.37
        if best is None or criterion>best[0]:
            best=(criterion,dict(radius_m=float(fit.x[2]),center=fit.x[:2],
                                 n=len(points),sectors=int(rich),rms_m=float(np.sqrt(np.mean(fit.fun**2)))))
    return None if best is None else best[1]


def update_geometry(objects, fits):
    groups=defaultdict(list)
    for item in objects["rails"]:
        original=item.setdefault("before_diameter_refinement",deepcopy(item["geom"]))
        item["geom"]=deepcopy(original)
        group=item["name"].rsplit("_",1)[0]
        groups[group].append(item)
    for key, group in groups.items():
        fit=fits.get(key)
        if not fit:
            continue
        diameter_cm=fit["range_cm"]
        compatible=[f for f in fit["fits"] if diameter_cm[0]<=200*f["radius_m"]<=diameter_cm[2]]
        by_name=defaultdict(list)
        for f in compatible:
            by_name[f["name"]].append(f)
        radius=diameter_cm[1]/200
        if not key.endswith("_post"):
            endpoints=np.stack([np.asarray(item["geom"]["fromto"]).reshape(2,3) for item in group])
            lengths=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1)
            parameter=np.r_[0.,np.cumsum(lengths)]
            samples=[]
            for i,item in enumerate(group):
                for f in by_name[item["name"]]:
                    t=f["t"]
                    nominal=(1-t)*endpoints[i,0]+t*endpoints[i,1]
                    delta=np.clip(f["world_center"]-nominal,-.048,.048)
                    samples.append((parameter[i]+t*lengths[i],*delta))
            samples=np.array(sorted(samples,key=lambda p:p[0]))
            if len(samples)>=3:
                knots=np.vstack((endpoints[0,0],endpoints[:,1]))
                for j in range(3):
                    displacement=median_filter(samples[:,j+1],size=3,mode="nearest")
                    knots[:,j]+=np.interp(parameter,samples[:,0],displacement)
                for i,item in enumerate(group):
                    item["geom"]["fromto"]=knots[i:i+2].ravel()
        for item in group:
            item["geom"]["size"]=[radius]
            item["evidence"]["diameter_fit_group"]=key
            item["evidence"]["heldout_arc_support"]=len(by_name[item["name"]])
    for row in ("_top","_middle"):
        for left,right in (("entry_left","bend_left"),("bend_left","upper_left")):
            a=groups[left+row][-1]["geom"]
            b=groups[right+row][0]["geom"]
            u=np.array(a["fromto"]).reshape(2,3)
            v=np.array(b["fromto"]).reshape(2,3)
            if np.linalg.norm(u[1]-v[0])<.20:
                u[1]=v[0]=(u[1]+v[0])/2
                a["fromto"], b["fromto"]=u.ravel(),v.ravel()
    root=ET.parse(OUT/"scene.xml").getroot()
    keyed={item["name"]:item for item in objects["rails"]}
    for geom in root.findall("./worldbody/geom"):
        if geom.get("name") in keyed:
            attrs=keyed[geom.get("name")]["geom"]
            for key in ("fromto","size"):
                geom.set(key,vector(attrs[key]))
    save_xml(root,OUT/"scene.xml")
    write_json(OUT/"objects.json",objects)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    stats=json.loads((OUT/"geometry.json").read_text())
    stride=stats["source_record_stride"]
    # Build the position search from ordinary reconstruction input, but use
    # a different interleaving for the diameter measurements.
    raw=np.load(OUT/"removed_crop.npy",mmap_mode="r")
    records=raw[stride//2::stride]
    selected=np.column_stack([records[k] for k in ("x","y","z")])
    info=json.loads((OUT/"removed_registration.json").read_text())
    pivot=np.array(info["pivot"])
    raw=((selected-pivot)@np.array(info["rotation"]).T+pivot+info["translation_m"]).astype(np.float32)
    tree=cKDTree(raw)
    objects=json.loads((OUT/"objects.json").read_text())
    items=objects["rails"]
    rng=np.random.default_rng(384)
    found=defaultdict(list)
    for item in items:
        name=item["name"]
        kind=name.rsplit("_",1)[0]
        geometry=item.get("before_diameter_refinement",item["geom"])
        a,b=np.array(geometry["fromto"]).reshape(2,3)
        length=np.linalg.norm(b-a)
        if length<.095:
            continue
        axis=(b-a)/length
        basis=perpendicular_basis(axis)
        for t in (.31,.63):
            middle=a+t*(b-a)
            pts=raw[tree.query_ball_point(middle,.112)]-middle
            hit=abs(pts@axis)<.025
            pts=pts[hit]@basis
            fit=estimate_circle(pts,rng)
            if fit:
                found[kind].append(dict(name=name,t=t,world_center=middle+basis@fit["center"],**fit))
    result={}
    for key, fits in found.items():
        r=np.array([f["radius_m"] for f in fits])
        result[key]=dict(slices=len(r),range_cm=np.percentile(200*r,[10,50,90]), fits=fits)
        print(key,len(r),np.round(np.percentile(200*r,[10,50,90]),2),flush=True)
    write_json(OUT/"rail_diameter_check.json",result)
    if args.apply:
        update_geometry(objects,result)


if __name__=="__main__":
    main()
