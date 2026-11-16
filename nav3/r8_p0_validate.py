"""Independent MuJoCo legacy-scene parity using the exact calibrated ray definitions."""
import sys,time,json,gc
import numpy as np
import warp as wp
import mujoco
from scipy.spatial.transform import Rotation
from .r8_p0_common import ROOT,GEO,dump,json_ready,GPUMonitor
from .r8_p0_ray import Scene,airy
from .calibration import mounts

@wp.kernel
def simple(mesh:wp.uint64,o:wp.array(dtype=wp.vec3),d:wp.array(dtype=wp.vec3),maxdist:float,t:wp.array(dtype=float),face:wp.array(dtype=int)):
 i=wp.tid();q=wp.mesh_query_ray(mesh,o[i],d[i],maxdist);t[i]=q.t if q.result else -1.0;face[i]=q.face if q.result else -1

def mesh_cast(v,f,o,d,maxdist):
 mesh=wp.Mesh(wp.array(v,dtype=wp.vec3,device='cuda'),wp.array(f.ravel(),dtype=int,device='cuda'),support_winding_number=False)
 oa=wp.array(o,dtype=wp.vec3,device='cuda');da=wp.array(d,dtype=wp.vec3,device='cuda');t=wp.empty(len(d),dtype=float,device='cuda');fi=wp.empty(len(d),dtype=int,device='cuda')
 wp.launch(simple,len(d),inputs=[mesh.id,oa,da,maxdist,t,fi],device='cuda');wp.synchronize();return t.numpy(),fi.numpy()

def primitive(kind,size):
 if kind==6:
  v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],float)*size
  f=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
  return v,f
 n=96;az=np.arange(n)*2*np.pi/n
 if kind in [2,4]:
  el=np.linspace(-np.pi/2,np.pi/2,49);v=np.array([[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)] for e in el for a in az]);v*=size if kind==4 else size[0]
 elif kind==3:
  r,h=size[:2];el=np.linspace(-np.pi/2,np.pi/2,49);v=np.array([[r*np.cos(e)*np.cos(a),r*np.cos(e)*np.sin(a),r*np.sin(e)+(-h if e<0 else h)] for e in el for a in az])
  # duplicate equator for exact cylindrical side.
  half=len(el)//2;v=np.insert(v,half*n,np.c_[r*np.cos(az),r*np.sin(az),np.full(n,-h)],axis=0)
 elif kind==5:
  r,h=size[:2];v=np.array([[r*np.cos(a),r*np.sin(a),z] for z in [-h,h] for a in az]);v=np.r_[v,[[0,0,-h],[0,0,h]]]
  f=[]
  for k in range(n):q=(k+1)%n;f.extend([[k,q,n+q],[k,n+q,n+k],[2*n,q,k],[2*n+1,n+k,n+q]])
  return v,np.array(f)
 else:raise ValueError(kind)
 rings=len(v)//n;f=[]
 for j in range(rings-1):
  for k in range(n):a=j*n+k;b=j*n+(k+1)%n;c=b+n;d=a+n;f.extend([[a,b,c],[a,c,d]])
 return v,np.array(f)

def model_geometry(m,d):
 vs=[];fs=[];gids=[];off=0
 for g in range(m.ngeom):
  if m.geom_group[g] not in [0,1]:continue
  kind=int(m.geom_type[g]);size=m.geom_size[g]
  if kind==1:
   h=int(m.geom_dataid[g]);nr=int(m.hfield_nrow[h]);nc=int(m.hfield_ncol[h]);sx,sy,sz,_=m.hfield_size[h];z=m.hfield_data[m.hfield_adr[h]:m.hfield_adr[h]+nr*nc].reshape(nr,nc)*sz
   xx,yy=np.meshgrid(np.linspace(-sx,sx,nc),np.linspace(-sy,sy,nr));v=np.c_[xx.ravel(),yy.ravel(),z.ravel()]
   a=(np.arange(nr-1)[:,None]*nc+np.arange(nc-1)[None]).ravel();f=np.r_[np.c_[a,a+1,a+nc+1],np.c_[a,a+nc+1,a+nc]]
  elif kind==7:
   k=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[k]);fa=int(m.mesh_faceadr[k]);v=m.mesh_vert[va:va+m.mesh_vertnum[k]];f=m.mesh_face[fa:fa+m.mesh_facenum[k]]
  else:v,f=primitive(kind,size)
  v=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];vs.append(v.astype(np.float32));fs.append(f.astype(np.int32)+off);gids.append(np.full(len(f),g,np.int32));off+=len(v)
 return np.concatenate(vs),np.concatenate(fs),np.concatenate(gids)

def compare(a,b):
 ha=a>=.1;hb=b>=.1;common=ha&hb;diff=np.abs(a-b)[common]
 return dict(rays=len(a),reference_hits=int(ha.sum()),warp_hits=int(hb.sum()),hit_agreement=float((ha==hb).mean()),common_hits=int(common.sum()),range_mae_m=float(diff.mean()) if len(diff) else None,range_p95_m=float(np.percentile(diff,95)) if len(diff) else None,range_max_m=float(diff.max()) if len(diff) else None,within_1cm_fraction=float((diff<.01).mean()) if len(diff) else None)

def main():
 mon=GPUMonitor('validation');sys.path.insert(0,str(ROOT/'vendor/map'));from runtime import load_scene,place_body
 from sensors import rays
 m=mujoco.MjModel.from_xml_path(str(ROOT/'ray_scene/scene.xml'));d=mujoco.MjData(m);sc=load_scene(ROOT/'ray_scene')
 for k,name in enumerate(['front','rear']):
  m.site_pos[m.site('lidar_'+name).id]=mounts()[k,:3,3]+[0,0,.54];m.geom_pos[m.geom('airy_'+name).id]=mounts()[k,:3,3]+[0,0,.54]
 ref=json.loads((ROOT/'metrics/ray_reference.json').read_text());rows=[]
 for idx in [4,14,44]:
  rr=ref['rows'][idx];pose=place_body(m,d,sc,rr['s_m']);native,ch,az,gap=rays(1,1,idx);body=m.body('m1').id;R=d.xmat[body].reshape(3,3)
  v,f,geom=model_geometry(m,d);o=[];dirs=[];tr=[];gi=[];start=time.perf_counter()
  for k,name in enumerate(['front','rear']):
   ray=np.ascontiguousarray(native@mounts()[k,:3,:3].T@R.T);origin=d.site_xpos[m.site('lidar_'+name).id].copy();ranges=np.empty(len(ray),np.float64);ids=np.empty(len(ray),np.int32)
   cache=GEO/f'r8_legacy_parity_{idx:03d}.npz'
   if cache.exists():
    old=np.load(cache);ranges[:]=old['mujoco_range'].reshape(2,-1)[k];ids[:]=old['mujoco_geom'].reshape(2,-1)[k]
   else:mujoco.mj_multiRay(m,d,origin,ray.ravel(),np.array([1,1,0,0,0,0],np.uint8),True,-1,ids,ranges,None,len(ray),60.)
   o.append(np.tile(origin,(len(ray),1)));dirs.append(ray);tr.append(ranges);gi.append(ids)
  o=np.concatenate(o);dirs=np.concatenate(dirs);tr=np.concatenate(tr);gi=np.concatenate(gi);refsecs=time.perf_counter()-start
  wt,wf=mesh_cast(v,f,o,dirs,60.);wg=np.where(wf>=0,geom[np.maximum(wf,0)],-1)
  # Both sensor pipelines reject self returns after nearest-hit search.
  tr[(gi>=0)&(m.geom_bodyid[np.maximum(gi,0)]==body)]=-1;wt[(wg>=0)&(m.geom_bodyid[np.maximum(wg,0)]==body)]=-1
  metric=compare(tr,wt);metric.update(station_m=rr['s_m'],index=idx,feature=rr['feature'],triangles=len(f),reference_load_or_trace_seconds=refsecs,reference_cached=cache.exists(),pose=pose.tolist())
  for label,condition in [('terrain',gi==m.geom('terrain').id),('other',gi!=m.geom('terrain').id)]:metric[label]=compare(tr[condition],wt[condition])
  rows.append(metric);np.savez_compressed(GEO/f'r8_legacy_parity_{idx:03d}.npz',mujoco_range=tr,warp_range=wt,mujoco_geom=gi,warp_geom=wg,pose=pose);print('LEGACY',json.dumps(metric),flush=True)
  dump(ROOT/'metrics/r8_p0_ray_validation.json',dict(status='in_progress',legacy=rows))
 # Exact instance-transform test against flattened world mesh, opaque and short range.
 scene=Scene();base=np.load(GEO/'r8_baseline.npz');pose=base['poses'][2];o,d,r=airy(4,4);r[:,1]=3.
 result=scene.cast(o,d,r,pose,alpha=False);cy,sy=np.cos(pose[3]),np.sin(pose[3]);R=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
 ow=o@R.T+pose[:3];dw=d@R.T
 vv=[];ff=[];offset=0
 for rec in scene.records:
  # Conservative ball around both origins (their offset norm <1m).
  lo=np.array(rec['bounds_min']);hi=np.array(rec['bounds_max']);dist=np.linalg.norm(np.maximum(np.maximum(lo-pose[:3],pose[:3]-hi),0))
  if dist>4:continue
  z=scene.data[rec['template']];mat=np.array(rec['matrix_world']);vv.append((z['vertices']@mat[:3,:3].T+mat[:3,3]).astype(np.float32));ff.append(z['faces']+offset);offset+=len(z['vertices'])
 total_faces=sum(len(x) for x in ff)
 # Estimate buffers + BVH < 4GB before allocating (in addition to <1GB instanced scene).
 assert sum(x.nbytes for x in vv+ff)+total_faces*80<4*1024**3
 wt,wf=mesh_cast(np.concatenate(vv),np.concatenate(ff),ow,dw,3.);wt[wt<.1]=-1
 transform=compare(result['distance'],wt);transform.update(triangles=total_faces,range_limit_m=3,alpha=False)
 dump(ROOT/'metrics/r8_p0_ray_validation.json',dict(status='complete',legacy=rows,instance_vs_world=transform,limitations=['legacy curved primitives approximated at 96 sides and 49 latitude rings; heightfield triangulation compared empirically','full geometry alpha is material texture cutoff 0.5, not measured lidar/IR transmittance']))
 print('INSTANCE',json.dumps(transform),flush=True);mon.close()
if __name__=='__main__':main()
