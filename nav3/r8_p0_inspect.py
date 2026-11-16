"""Read-only blend inventory before full geometry export."""
import bpy, json,time,collections
from pathlib import Path
p='/home/hpf/wang-sm/ATEC_NAV_WSM/ATEC_Env/scene/atec_trail_pbr.blend'
t=time.time();bpy.ops.wm.open_mainfile(filepath=p);print('OPEN_SECONDS',time.time()-t,flush=True)
s=bpy.context.scene;d=bpy.context.evaluated_depsgraph_get();num_instances=0
rows=[]
for inst in d.object_instances:
 num_instances+=1
 o=inst.object
 if o.type!='MESH':continue
 rows.append(dict(name=o.name,original=o.original.name,vertices=len(o.data.vertices),polygons=len(o.data.polygons),materials=[m.name if m else '' for m in o.data.materials],hide_render=o.hide_render,hide_viewport=o.hide_viewport,is_instance=inst.is_instance,show_self=inst.show_self,collections=[c.name for c in o.original.users_collection]))
mats=[]
for m in bpy.data.materials:
 ns=[]
 if m.use_nodes:
  for n in m.node_tree.nodes:
   ns.append(dict(type=n.type,name=n.name,image=dict(path=n.image.filepath,size=list(n.image.size),packed=bool(n.image.packed_file)) if n.type=='TEX_IMAGE' and n.image else None,links={i.name:[(l.from_node.name,l.from_socket.name) for l in i.links] for i in n.inputs if i.is_linked}))
 mats.append(dict(name=m.name,nodes=ns))
out=dict(bpy=bpy.app.version_string,seconds=time.time()-t,objects=len(s.objects),instances=num_instances,meshes=rows,materials=mats,scene_properties=dict(s.items()))
Path('fullgeo/r8_inventory.json').write_text(json.dumps(out,indent=2))
print('TOTAL',len(rows),'VERTICES',sum(r['vertices'] for r in rows),'POLYGONS',sum(r['polygons'] for r in rows),flush=True)
print('MATERIALS',[m['name'] for m in mats],flush=True)
print('HIDDEN',[(r['name'],r['hide_render'],r['hide_viewport'],r['show_self']) for r in rows if r['hide_render'] or r['hide_viewport'] or not r['show_self']],flush=True)
