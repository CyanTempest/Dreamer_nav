"""Build the A-trail pilot and write its geometry and comparison measurements."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import bilinear, cell_index, load_config, write_json
from nav4.mapping.grid import load_centreline
from nav4.mapping.report_trail_map import detect_stairs
from nav4.matrix.compose import new_scene, place_hard, place_terrain, save_xml, write_obj
from nav4.matrix.detail import local_v8, xyz
from nav4.matrix.ground_mesh import ground_raster, point_mesh_distances, underlay_treads
from nav4.matrix.objects import rail_shapes, solid_shapes, wall_shapes, woody_shapes
from nav4.matrix.register import sample_records, summary, voxel_ids
from nav4.matrix.stairs import fit_flight
from nav4.matrix.survey import OUT
from nav4.matrix.terrain import triangulate


def raw_cloud(name):
    source = np.load(OUT/f"{name}_crop.npy",mmap_mode="r")
    records, stride = sample_records(source)
    p = xyz(records)
    if name != "ply":
        info = json.loads((OUT/f"{name}_registration.json").read_text())
        origin = np.array(info["pivot"])
        p = ((p-origin) @ np.array(info["rotation"]).T + origin+info["translation_m"]).astype(np.float32)
    return p, stride


def coarse_stairs(layers, origin, res):
    config, _ = load_config()
    center = load_centreline()
    s = np.arange(0,30.5,.05)
    xy = np.column_stack([np.interp(s,center["s"],center["xy"][:,i]) for i in range(2)])
    ij = cell_index(xy,origin,res)
    z = layers[ij[:,0],ij[:,1],0]
    return detect_stairs(s,z,config)


def write_terrain(out, root, raster, active, xy0, config):
    pitch = config["pitch_m"]
    width = config["tile_cells"]
    (out/"terrain").mkdir(exist_ok=True)
    meshes = []
    for i in range(0,raster.shape[0],width):
        for j in range(0,raster.shape[1],width):
            stop = np.minimum(raster.shape,[i+width,j+width])
            start = np.maximum([0,0],[i-2,j-2])
            bound = np.minimum(raster.shape,stop+2)
            region = tuple(slice(a,b) for a,b in zip(start,bound))
            select = np.zeros(bound-start,dtype=bool)
            ij = tuple(slice(a,b) for a,b in zip(np.array([i,j])-start,stop-start))
            select[ij] = active[region][ij]
            if not select.any():
                continue
            v,f,_ = triangulate(raster[region],active[region],select,pitch,
                                config["jump_m"],config["centre_tolerance_m"])
            # Center in all dimensions for MJCF mesh inertia computations.
            offset = np.r_[xy0+start*pitch,0.]
            center = v.mean(0)
            name = f"terrain_{i//width:02d}_{j//width:02d}"
            obj = f"terrain/{name}.obj"
            write_obj(out/obj,v-center,f)
            place_terrain(root,name,obj,offset+center,config["collision_skin_m"])
            meshes.append((v+offset,f))
    return meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",choices=["ply","removed","map0"],required=True)
    parser.add_argument("--skip-distances",action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    out = OUT/"preview" if args.source=="ply" else OUT
    out.mkdir(exist_ok=True)
    (out/"hard").mkdir(exist_ok=True)
    (out/"stairs").mkdir(exist_ok=True)
    settings = json.loads(Path(__file__).with_name("settings.json").read_text())
    root = new_scene("A-trail 0 to 30 m",settings)
    path,origin,layers,known,observed,res,_,_ = local_v8()
    src, stride = raw_cloud(args.source)
    # Keep local source unshifted in all output assets so route and poses share its frame.
    ply = np.load(OUT/"ply_crop.npy",mmap_mode="r")
    p = xyz(ply)
    ref_height = bilinear(layers[...,0],origin,p[:,:2],res=res)
    keep = (abs(p[:,2]-ref_height)<.36) & (abs(ply["normal_z"])>.62)
    p = p[keep]
    p = p[voxel_ids(p,.024)]
    pitch = settings["terrain"]["pitch_m"]
    ground_start = time.monotonic()
    xy0,raster,active,observations,supported,holdout,source_noise = ground_raster(
        src,p,layers[...,0],origin,res,path,pitch)
    print("ground raster",raster.shape,active.sum(),observations.sum(),time.monotonic()-ground_start,flush=True)
    # Use a spatial shortcut for the point-to-plane and riser fits.
    sleeve = cKDTree(path).query(src[:,:2],workers=4)[0] < 1.38
    stair_points = src[sleeve]
    # Same interleaved raw records are held out from both regular ground and treads.
    train = stair_points[(np.flatnonzero(sleeve)*37)%11 >= 2]
    staircase, stair_record = [], []
    v8_flights, v8_steps = coarse_stairs(layers, origin, res)
    for flight in settings["stairs"]["flights"]:
        print("fitting",flight["name"],flush=True)
        shapes,metrics = fit_flight(train,path,flight,settings["stairs"])
        for metric,shape in zip(metrics,shapes):
            s = metric["station_m"]
            nearest = min(v8_steps,key=lambda step:abs(step["s"]-s))
            metric["v8_nearest_station_delta_m"] = s-nearest["s"]
            if abs(s-nearest["s"]) < .28:
                metric.update(v8_riser_m=nearest["h"],v8_station_m=nearest["s"])
            center = shape["vertices"].mean(0)
            name = f"stair_{len(staircase):02d}"
            filename = f"stairs/{name}.obj"
            write_obj(out/filename,shape["vertices"]-center,shape["faces"])
            item = dict(name=name,file=filename,material="stone",group=2,geom=dict(type="mesh",pos=center))
            place_hard(root,item)
            staircase.append(shape)
        stair_record.append(dict(flight=flight["name"],steps=metrics))
    write_json(out/"stairs.json",dict(flights=stair_record,v8_flights=v8_flights,v8_step_locations=v8_steps))
    write_json(out/"stair_geometry.json",staircase)
    replaced, underlay = underlay_treads(raster,xy0,pitch,staircase)
    rails = rail_shapes(train,path,raster,xy0,pitch,settings["rails"])
    walls = wall_shapes(train,path,settings["walls"],out)
    solids = walls + solid_shapes(src,path,origin,layers,res,raster,xy0,pitch,staircase,rails,settings["hard"],out)
    wood = woody_shapes(src,path,raster,xy0,pitch,settings["trees"],out)
    solids.extend(wood)
    for item in rails+solids:
        place_hard(root,item)
    write_json(out/"objects.json",dict(rails=rails,solids=solids))
    print("terrain tessellation",int((active&~replaced).sum()),"solids",len(solids),"capsules",len(rails),flush=True)
    meshes = write_terrain(out,root,underlay,active,xy0,settings["terrain"])
    save_xml(root,out/"scene.xml")
    np.savez_compressed(out/"raster.npz",origin=xy0,pitch=pitch,z=raster,active=active,
                        observed=observations,supported=supported,stairs=replaced)
    bounds = ~replaced & active
    route_dist = cKDTree(path).query((np.argwhere(bounds)*pitch+xy0),workers=4)[0]
    result = dict(source=args.source, working_returns=len(src), source_record_stride=stride,
                  pitch_m=pitch, cells=int(active.sum()), observations=int((observations&bounds).sum()),
                  fitted_ground_cells=int(bounds.sum()), v8_fill_cells=int((bounds&~supported).sum()),
                  v8_fill_cells_within_1m=int(np.sum(~supported[bounds] & (route_dist<1))),
                  triangles_terrain=sum(len(f) for _,f in meshes),
                  triangles_stairs=sum(len(s["faces"]) for s in staircase),
                  triangles_solids=sum(s["faces"] for s in solids),
                  capsules=len(rails), terrain_tiles=len(meshes),
                  measured_walls=len(walls), woody_solids=len(wood), source_match_distance_m=summary(source_noise),
                  build_seconds_before_error=time.monotonic()-start)
    if not args.skip_distances:
        holdout_distance = cKDTree(path).query(holdout[:,:2],workers=4)[0]
        for label,mask in (("corridor",np.ones(len(holdout),dtype=bool)),("within_1m",holdout_distance<=1),
                           ("well_supported_within_1m",(holdout_distance<=1)&(source_noise<=.03))):
            result[f"ground_holdout_{label}"] = point_mesh_distances(
                meshes+[(s["vertices"],s["faces"][2:4]) for s in staircase],holdout[mask])
    result["build_seconds"] = time.monotonic()-start
    write_json(out/"geometry.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    main()
