"""Keep a documented robot collision envelope and merge exactly adjacent voxels."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius',type=float,default=.9)
    p.add_argument('--height',type=float,default=1.25)
    args=p.parse_args();src=args.source.resolve();dst=args.output.resolve()
    dst.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((src/'manifest.json').read_text());voxel=manifest['obstacle_voxel_m']
    original=np.load(src/'obstacles.npz')['boxes']
    route=np.loadtxt(src/'centerline.csv',delimiter=',',skiprows=1)
    dist,idx=cKDTree(route[:,1:3]).query(original[:,:2])
    keep=(dist-np.linalg.norm(original[:,3:5],axis=1)<=args.radius)&(original[:,2]-original[:,5]<=route[idx,3]+args.height)
    boxes=original[keep]
    bounds=np.rint(np.c_[boxes[:,:3]-boxes[:,3:],boxes[:,:3]+boxes[:,3:]]/voxel).astype(np.int64)
    volume_before=np.prod(bounds[:,3:]-bounds[:,:3],axis=1).sum()
    for _ in range(3):
        before=len(bounds)
        for axis in (0,1,2):
            others=[i for i in range(6) if i not in (axis,axis+3)]
            groups={}
            for box in bounds:groups.setdefault(tuple(box[others]),[]).append(box)
            merged=[]
            for group in groups.values():
                group.sort(key=lambda b:b[axis]);current=group[0].copy()
                for nxt in group[1:]:
                    if current[axis+3]==nxt[axis]:current[axis+3]=nxt[axis+3]
                    else:merged.append(current);current=nxt.copy()
                merged.append(current)
            bounds=np.asarray(merged)
        if len(bounds)==before:break
    volume_after=np.prod(bounds[:,3:]-bounds[:,:3],axis=1).sum()
    if volume_after!=volume_before:raise AssertionError('Occupied voxel volume changed during merging')
    np.savez_compressed(dst/'obstacles.npz',boxes=np.c_[(bounds[:,:3]+bounds[:,3:])*voxel/2,(bounds[:,3:]-bounds[:,:3])*voxel/2])
    for path in src.iterdir():
        if path.name not in ('manifest.json','obstacles.npz'):os.link(path,dst/path.name)
    manifest.update(scene_id='atec_competition_forest_100m_'+dst.name,obstacle_boxes=len(bounds),
                    obstacle_corridor_radius_m=args.radius,obstacle_height_max_m=args.height)
    manifest['collision_envelope_compaction']={'source':str(src),'source_boxes':len(original),'selected_boxes':len(boxes),
        'method':'intersecting_robot_envelope_then_exact_adjacent_integer_box_union',
        'selected_volume_preserved':True,'selected_volume_voxels':int(volume_after),
        'visual_context_unchanged':True,'collision_outside_envelope_not_claimed':True}
    (dst/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['collision_envelope_compaction'],indent=2));print('Final boxes:',len(bounds))


if __name__=='__main__':main()
