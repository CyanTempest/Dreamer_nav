"""Compare horizontal hard-obstacle clearances to frozen v8 body-band cells."""
import csv
import mujoco
import numpy as np

from nav4.mapping.common import cell_index, write_json
from nav4.matrix.detail import local_v8
from nav4.matrix.register import summary
from nav4.matrix.stairs import heading, path_at
from nav4.matrix.survey import OUT
from nav4.matrix.verify import height

GEOM_GROUP = np.array([0,1,0,0,0,0],dtype=np.uint8)
HEIGHT_BAND = np.linspace(.3025,.7975,23)
MAX_DISTANCE = 2.
RAY_STEP = .025


def measure():
    model=mujoco.MjModel.from_xml_path(str(OUT/"scene.xml"))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    path,origin,layers,_,_,res,*_ = local_v8()
    rows=[]
    for station in np.arange(0,30.01,.10):
        center=path_at(path,station)
        tangent=heading(path,station)
        across=np.array([-tangent[1],tangent[0]])
        ground=height(model,data,center)[0]
        for label,direction in (("left",across),("right",-across)):
            along=np.arange(0,MAX_DISTANCE+RAY_STEP/2,RAY_STEP)
            ij=cell_index(center+along[:,None]*direction,origin,res)
            ix=np.flatnonzero(layers[ij[:,0],ij[:,1],6]==1)
            v8=float(along[ix[0]]) if len(ix) else None
            hits=[]
            for dz in HEIGHT_BAND:
                hit=np.array([-1],dtype=np.int32)
                d=mujoco.mj_ray(model,data,np.r_[center,ground+dz],np.r_[direction,0.],
                                GEOM_GROUP,1,-1,hit)
                if 0<=d<=MAX_DISTANCE:
                    hits.append((d,model.geom(int(hit[0])).name,dz))
            d,name,z=min(hits) if hits else (None,None,None)
            rows.append(dict(station_m=round(station,3),side=label,v8_m=v8,pilot_m=d,pilot_geom=name,
                             ray_height_m=z,delta_m=None if v8 is None or d is None else d-v8))
    with (OUT/"hard_clearance.csv").open("w",newline="") as f:
        csv_writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    result={}
    for label in ("left","right"):
        lines=[r for r in rows if r["side"]==label]
        paired=[r for r in lines if r["delta_m"] is not None]
        result[label]=dict(
            all_stations=len(lines),paired_delta_m=summary(np.array([r["delta_m"] for r in paired])),
            both_clear=sum(r["v8_m"] is None and r["pilot_m"] is None for r in lines),
            pilot_only=sum(r["v8_m"] is None and r["pilot_m"] is not None for r in lines),
            v8_only=sum(r["v8_m"] is not None and r["pilot_m"] is None for r in lines),
            v8_only_station_head=[r["station_m"] for r in lines if r["v8_m"] is not None and r["pilot_m"] is None][:12],
            paired_max=[r for r in sorted(paired,key=lambda p:abs(p["delta_m"]),reverse=True)[:6]])
    write_json(OUT/"hard_clearance.json",dict(radius_m=MAX_DISTANCE, relative_probe_heights_m=HEIGHT_BAND,
                                             statistics=result))
    print(result,flush=True)


if __name__=="__main__":
    measure()
