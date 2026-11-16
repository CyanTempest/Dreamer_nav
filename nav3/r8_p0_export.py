"""Blender 4.5.3 evaluated full-scene export; streaming world arrays + exact instances."""
import os,time,json,hashlib,collections,resource
from pathlib import Path
import numpy as np
import bpy
from PIL import Image
from .r8_p0_common import ROOT,PKG,GEO,dump
CLASSES={0:'unclassified',1:'road',2:'stairs',3:'rail',4:'bank_soil',5:'rock',6:'root',7:'tree_trunk',8:'tree_leaves',9:'shrub_mixed',10:'grass',11:'fern',12:'wall',13:'sign',15:'tree_branch'}
def semantic(name,mat):
 if name=='surface_concrete':return 1
 if name.startswith('surface_'):return 4
 if name.startswith('plant_'):
  if 'fern' in mat:return 11
  if 'grass' in mat:return 10
  if 'shrub' in mat:return 9
  if 'leaves' in mat:return 8
  if 'branch' in mat:return 15
  return 7
 if mat=='metal':return 3
 if mat=='concrete_wall':return 12
 if mat=='rock':return 5
 if mat=='bark':return 6
 if mat=='sign':return 13
 return 0

def main():
 t=time.perf_counter();bpy.ops.wm.open_mainfile(filepath=str(PKG/'scene/atec_trail_pbr.blend'))
 dg=bpy.context.evaluated_depsgraph_get()
 items=[(i.object,np.array(i.matrix_world,dtype=np.float32),bool(i.show_self)) for i in dg.object_instances if i.object.type=='MESH']
 items=[(o,m) for o,m,show in items if show and not o.hide_render]
 assert not any(len(o.modifiers) for o,m in items), 'Audit modifiers before template deduplication'
 bundle=json.loads((PKG/'scene/scene_bundle.json').read_text());sur=np.load(PKG/'scene/trail_surfaces.npz')
 # Road station assignment uses the shared builder route, read-only import.
 import sys
 sys.path.insert(0,str(PKG/'tools'))
 from trail_scene import Route
 spec=json.loads((PKG/'config/track_spec.json').read_text());route=Route(spec)
 templates={};arrays={};records=[];checks=[];mats=list(bpy.data.materials);mid={m.name:i for i,m in enumerate(mats)}
 alpha=[];alpha_linear=[];alpha_meta=[]
 for m in mats:
  nodes=[n for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and '_alpha_' in n.image.filepath] if m.use_nodes else []
  if nodes:
   node=nodes[0];p=Path(bpy.path.abspath(node.image.filepath));a=np.asarray(Image.open(p).convert('L'),np.uint8)
   alpha_meta.append(dict(material=m.name,path=str(p),width=a.shape[1],height=a.shape[0],offset=sum(x.size for x in alpha),colorspace=node.image.colorspace_settings.name,uv_mapping='identity, repeat; bilinear cutoff 0.5 in linear colorspace'))
   alpha.append(a.ravel());linear=a.astype(np.float32)/255
   if node.image.colorspace_settings.name=='sRGB':linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
   alpha_linear.append(linear.ravel())
  else:alpha_meta.append(dict(material=m.name,width=0,height=0,offset=0))
 counts=np.zeros(16,np.int64);nv=nf=0
 for o,m in items:
  key=o.data.as_pointer()
  if key not in templates:
   k=len(templates);templates[key]=k;me=o.data;me.calc_loop_triangles()
   v=np.empty((len(me.vertices),3),np.float32);me.vertices.foreach_get('co',v.ravel())
   f=np.empty((len(me.loop_triangles),3),np.int32);me.loop_triangles.foreach_get('vertices',f.ravel())
   mi=np.empty(len(f),np.int32);me.loop_triangles.foreach_get('material_index',mi)
   names=[x.name for x in me.materials];sem=np.array([semantic(o.name,x) for x in names],np.uint8)[mi]
   material=np.array([mid[x] for x in names],np.int32)[mi]
   uv=np.zeros((len(f),3,2),np.float32)
   if me.uv_layers.active:
    loops=np.empty((len(f),3),np.int32);me.loop_triangles.foreach_get('loops',loops.ravel())
    alluv=np.empty((len(me.loops),2),np.float32);me.uv_layers.active.data.foreach_get('uv',alluv.ravel());uv=alluv[loops]
   if o.name.startswith('surface_'):
    from scipy.spatial import cKDTree
    road_centers=np.array([np.array(c['vertices'])[:3].mean(0) for c in bundle['colliders'] if c['name'].startswith('road_')])
    road_distance=cKDTree(road_centers).query(v[f].mean(1))[0]
    sem[road_distance<.002]=1
   if o.name=='surface_concrete':
    from scipy.spatial import cKDTree
    # Dense route stations avoid classifying concrete risers as ordinary road.
    ss=np.linspace(0,1000,20001);rp=np.array([route.position(float(s)) for s in ss]);tree=cKDTree(rp[:,:2]);st=ss[tree.query(v[f].mean(1)[:,:2])[1]]
    for stairs in bundle['stairs']:sem[(st>=stairs['start']-.02)&(st<=stairs['end']+.02)]=2
   if o.name.startswith('surface_'):
    name=o.name[8:];checks.append(dict(object=o.name,vertex_max_abs_error_m=float(cKDTree(sur[name+'_vertices']).query(v)[0].max()),triangles=len(f),source_triangles=len(sur[name+'_faces'])))
   arrays[k]=(v,f,sem,material,uv)
   np.savez(GEO/f'r8_template_{k:03d}.npz',vertices=v,faces=f,semantic=sem,material=material,uv=uv)
  k=templates[key];v,f,sem,material,uv=arrays[k]
  w=v@m[:3,:3].T+m[:3,3];counts+=np.bincount(sem,minlength=16)
  records.append(dict(name=o.name,template=k,matrix_world=m.tolist(),vertex_offset=nv,face_offset=nf,vertices=len(v),triangles=len(f),bounds_min=w.min(0).tolist(),bounds_max=w.max(0).tolist()))
  nv+=len(v);nf+=len(f)
 print('TEMPLATE_DONE',len(templates),'world_vertices',nv,'world_triangles',nf,'seconds',time.perf_counter()-t,flush=True)
 np.save(GEO/'r8_alpha.npy',np.concatenate(alpha));np.save(GEO/'r8_alpha_linear.npy',np.concatenate(alpha_linear));dump(GEO/'r8_materials.json',alpha_meta)
 dump(GEO/'r8_instances.json',records)
 # Independently stored fully expanded world geometry. File-level mapping bounds RSS.
 shapes=[('vertices',np.float32,(nv,3)),('faces',np.uint32,(nf,3)),('semantic',np.uint8,(nf,))]
 for name,dtype,shape in shapes:
  mm=np.lib.format.open_memmap(GEO/f'r8_world_{name}.npy',mode='w+',dtype=dtype,shape=shape);del mm
 for j,r in enumerate(records):
  v,f,s,_,_=arrays[r['template']];m=np.array(r['matrix_world'],np.float32)
  for name,data,offset,shape in [('vertices',v@m[:3,:3].T+m[:3,3],r['vertex_offset'],(nv,3)),('faces',f.astype(np.uint32)+r['vertex_offset'],r['face_offset'],(nf,3)),('semantic',s,r['face_offset'],(nf,))]:
   # Ordinary positional file writes, avoiding 20+ GB dirty mmap pages.
   p=GEO/f'r8_world_{name}.npy'
   with p.open('r+b') as out:
    np.lib.format.read_magic(out);np.lib.format.read_array_header_1_0(out);header=out.tell()
    out.seek(header+offset*data.dtype.itemsize*(3 if data.ndim==2 else 1));out.write(data.tobytes())
  if j%500==0:print('EXPANDED',j,'/',len(records),'seconds',time.perf_counter()-t,flush=True)
 # Primitive positions are compared by bounding-center (capsules by endpoint midpoint).
 byname={r['name']:r for r in records};pr=[]
 for p in bundle['primitives']:
  if p['name'] not in byname:continue
  r=byname[p['name']];center=(np.array(r['bounds_min'])+r['bounds_max'])/2
  target=np.mean(np.array(p['fromto']).reshape(2,3),0) if p['type']=='capsule' else np.array(p['position'])
  pr.append(dict(name=p['name'],type=p['type'],center_error_m=float(np.linalg.norm(center-target))))
 # Float32 translated vertices and meshes have inherent sub-mm rounding at x~850 m.
 result=dict(status='complete',bpy=bpy.app.version_string,seconds=time.perf_counter()-t,objects=len(records),unique_meshes=len(templates),vertices=nv,triangles=nf,world_array_bytes=sum((GEO/f'r8_world_{n}.npy').stat().st_size for n,_,_ in shapes),template_array_bytes=sum(sum(a.nbytes for a in x) for x in arrays.values()),class_triangles={CLASSES[i]:int(counts[i]) for i in CLASSES},classes=CLASSES,surface_checks=checks,primitive_checks=pr,missing_primitive_names=[p['name'] for p in bundle['primitives'] if p['name'] not in byname],maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,source_sha256=hashlib.sha256((PKG/'scene/atec_trail_pbr.blend').read_bytes()).hexdigest(),bundle_sha256=hashlib.sha256((PKG/'scene/scene_bundle.json').read_bytes()).hexdigest(),scene_bundle_hash_matches=bpy.context.scene.get('geometry_sha256')==hashlib.sha256((PKG/'scene/scene_bundle.json').read_bytes()).hexdigest(),alpha_cutoff=.5,shrub_semantic='mixed woody and leaf geometry: do not claim all shrubs soft')
 dump(ROOT/'metrics/r8_p0_export.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['primitive_checks','missing_primitive_names']},indent=2),flush=True)
if __name__=='__main__':main()
