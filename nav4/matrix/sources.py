"""Compare dense LAS sources in the same stable overlap as the planning PLY."""
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from nav4.mapping.common import bilinear, write_json
from nav4.matrix.detail import local_v8, xyz
from nav4.matrix.register import sample_records, summary
from nav4.matrix.survey import OUT


def info(name):
    return json.loads((OUT/f"{name}_registration.json").read_text())


def own_to_ply(p, reg):
    pivot = np.array(reg["pivot"])
    return (p-pivot)@np.array(reg["rotation"]).T + pivot + reg["translation_m"]


def main():
    _,origin,layers,_,_,res,*_ = local_v8()
    records = {name:sample_records(np.load(OUT/f"{name}_crop.npy",mmap_mode="r"))[0]
               for name in ("removed","map0")}
    regs = {name:info(name) for name in records}
    p = {name:own_to_ply(xyz(rec),regs[name]).astype(np.float32)
         for name,rec in records.items()}
    a,b = regs["removed"],regs["map0"]
    # Relative rigid transform expressed about their shared registration pivot.
    to0 = np.array(b["rotation"]).T
    shift = to0@(np.array(a["translation_m"])-b["translation_m"])
    rot = to0@np.array(a["rotation"])
    result = dict(pivot=a["pivot"], removed_to_map0_translation_m=shift,
                  removed_to_map0_rotation_deg=Rotation.from_matrix(rot).as_rotvec(degrees=True))
    ref = np.load(OUT/"ply_crop.npy",mmap_mode="r")
    q = xyz(ref[::17])
    near = abs(q[:,2]-bilinear(layers[...,0],origin,q[:,:2],res=res))<.16
    stable = q[near & (abs(ref["normal_z"][::17])>.62)]
    time_comparison = {}
    for name,other in (("removed","map0"),("map0","removed")):
        points = p[name]
        tree = cKDTree(points)
        d,i = tree.query(stable,workers=4)
        over = d<.04
        picks = np.unique(i[over])
        distance,_ = cKDTree(p[other]).query(points[picks],workers=4)
        result[f"{name}_to_{other}"] = dict(total=len(distance),coverage_10cm=float((distance<.1).mean()),
                                          all_euclidean_m=summary(distance),overlap_euclidean_m=summary(distance[distance<.1]))
        time_comparison[name]=dict(gps_time_quantiles=np.percentile(records[name]["gps_time"],[0,25,50,75,100]),
                                   classes=np.unique(records[name]["classification"]).tolist(),
                                   intensity_quantiles=np.percentile(records[name]["intensity"],[0,25,50,75,100]))
    result["sensor_records"] = time_comparison
    write_json(OUT/"source_comparison.json",result)


if __name__ == "__main__":
    main()
