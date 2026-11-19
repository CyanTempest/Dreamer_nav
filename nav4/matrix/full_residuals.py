"""Locate large held-out vertical residuals for the exact-distance report."""
import json
import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.full_ground import FULL, PITCH, WORK, grid_sample
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import LIMITS
from nav4.matrix.ground_mesh import in_polygon
from nav4.matrix.register import summary as stats
from nav4.matrix.survey import Corridor


def summary(values):
    return stats(values) if len(values) else {"count":0}


def main():
    with np.load(WORK/"full_raster_input.npz") as first:
        samples = first["samples"]
    prng = np.random.default_rng(1678)
    if len(samples)>500000:
        samples = samples[prng.choice(len(samples),500000,replace=False)]
    _, path = route()
    cor = Corridor(path)
    s, lat, use = cor.project(samples[:, :2])
    p, s, lat = samples[use], s[use], lat[use]
    d = cKDTree(path).query(p[:, :2],workers=4)[0]
    with np.load(FULL/"surface.npz") as surface:
        z, base, gap = surface["z"],surface["origin"],surface["gap"]
    floor = grid_sample(z,base,p[:, :2])
    ij = np.rint((p[:, :2]-base)/PITCH).astype(int).clip(0,np.array(z.shape)-1)
    raw_distance = gap[tuple(ij.T)]
    fitted_stair = np.zeros(len(p),bool)
    for record in json.loads((FULL/"stair_geometry.json").read_text()):
        poly = np.asarray(record["polygon"])
        ids = np.flatnonzero(((p[:, :2]>=poly.min(0)) & (p[:, :2]<=poly.max(0))).all(1))
        if not len(ids):
            continue
        q = ids[in_polygon(p[ids,:2],poly)]
        slab = np.asarray(record["vertices"])[4:]
        top = np.linalg.lstsq(np.c_[slab[:,:2],np.ones(4)],slab[:,2],rcond=None)[0]
        floor[q] = np.maximum(floor[q],p[q,:2]@top[:2]+top[2])
        fitted_stair[q] = True
    diff = p[:,2]-floor
    all_rows = []
    for i,(lo,hi) in enumerate(zip(LIMITS[:-1],LIMITS[1:])):
        use = (s>=lo)&(s<hi)&np.isfinite(diff)
        near = use&(d<=1.)
        bins = []
        for a in np.arange(lo,hi,5):
            selected = use&(s>=a)&(s<a+5)
            local = near&(s>=a)&(s<a+5)
            bins.append(dict(from_m=a, corridor=summary(diff[selected]), near_route=summary(diff[local])))
        record = dict(segment=i, range_m=[lo,hi],
                      exact_area_context="Vertical offsets at XY only, versus the upper ground or fitted stair top.",
                      near_rms_peaks=sorted(bins,key=lambda a:a["near_route"].get("rms",0) or 0,reverse=True)[:3],
                      corridor_rms_peaks=sorted(bins,key=lambda a:a["corridor"].get("rms",0) or 0,reverse=True)[:3],
                      observed_corridor=summary(diff[use&(raw_distance<=.15)]),
                      fill_corridor=summary(diff[use&(raw_distance>.15)]),
                      near_and_in_stair_footprint=summary(diff[near&fitted_stair]),
                      near_and_outside_stair_footprint=summary(diff[near&~fitted_stair]))
        all_rows.append(record)
    write_json(FULL/"residual_locations.json",all_rows)
    print("diagnostic",len(p),"eligible terrain points",flush=True)


if __name__ == "__main__":
    main()
