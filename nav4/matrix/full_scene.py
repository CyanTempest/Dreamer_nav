"""Build the pilot revision and welded, sectioned full-route MJCF assets."""
import argparse
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix import compose
from nav4.matrix.build import raw_cloud
from nav4.matrix.detail import xyz
from nav4.matrix.full_ground import (
    FULL, PILOT, PITCH, LOCAL_SUPPORT_M, V8Map, apply_grades, edge_statistics,
    grid_geometry, make_ground, patches, region, route_anchors,
)
from nav4.matrix.full_scan import WORK, route
from nav4.matrix.ground_mesh import in_polygon, point_mesh_distances
from nav4.matrix.survey import Corridor, OUT

LIMITS = np.r_[np.arange(0., 701., 50.), 722.]
GEOM_TILE = 64
CONFIG = json.loads(Path(__file__).with_name("settings.json").read_text())
CONFIG["materials"]["vegetation"] = [.28, .39, .20, 1]


def load_pilot_objects():
    tree = json.loads((OUT / "objects.json").read_text())
    items = tree["rails"] + tree["solids"]
    shapes = json.loads((OUT / "stair_geometry.json").read_text())
    for shape in shapes:
        for key, value in shape.items():
            if isinstance(value, list):
                shape[key] = np.asarray(value)
    for obj in items:
        obj["old"] = True
    return shapes, items


def pilot_cloud():
    raw, _ = raw_cloud("removed")
    holdout = (np.arange(len(raw)) * 37) % 11 < 2
    table = np.load(OUT / "ply_crop.npy", mmap_mode="r")
    ply = np.column_stack([table[key] for key in
                           ("x", "y", "z", "normal_x", "normal_y", "normal_z", "classification")])
    return np.c_[raw, holdout], ply


def raw_obj(path):
    v, f = [], []
    with path.open() as file:
        for line in file:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:]])
            if line.startswith("f "):
                f.append([int(x.split("/")[0])-1 for x in line.split()[1:]])
    return np.asarray(v), np.asarray(f)


def add_terrain(root, name, rel, center, is_fill):
    compose.place_terrain(root, name, rel, center, CONFIG["terrain"]["collision_skin_m"])
    geom = root.find("worldbody").findall("geom")[-1]
    geom.set("group", "4" if is_fill else "0")
    geom.set("rgba", ".51 .50 .46 1" if is_fill else ".55 .54 .49 1")


def place_soft(root, item):
    """Write a ray-visible visual mesh with both MuJoCo collision masks clear."""
    compose.xml(root.find("asset"), "mesh", name=item["name"], file=item["file"], inertia="shell")
    compose.xml(root.find("worldbody"), "geom", **item["geom"], name=item["name"], group=5,
                mesh=item["name"], contype=0, conaffinity=0, material="vegetation")


def scene_dir(out, i):
    return out if out == PILOT else out / "segments" / f"seg_{i:02d}"


def aggregate(current, values):
    for key, v in values.items():
        current[key] = current.get(key, 0) + v


def make_scenes(out, path, ranges, base, ground, gap, active, stations, stairs, objects, samples):
    n = len(ranges) - 1
    trees = [compose.new_scene(f"A trail {a:g}–{b:g} m", CONFIG) for a, b in zip(ranges[:-1], ranges[1:])]
    metadata = [dict(range_m=[a, b], surface={}, ground_projected_m2=0.,
                     fill_projected_m2=0., fill_near_route_m2=0., objects={},
                     fill_area_by_5m={}) for a, b in zip(ranges[:-1], ranges[1:])]
    recordings = [[] for _ in range(n)]
    for i in range(n):
        for k in ("terrain", "stairs", "hard"):
            (scene_dir(out, i) / k).mkdir(parents=True, exist_ok=True)

    observe = gap <= LOCAL_SUPPORT_M
    c = ~observe
    is_fill_grid = c[:-1, :-1] | c[1:, :-1] | c[1:, 1:] | c[:-1, 1:]
    owners = np.searchsorted(ranges[1:-1], stations, side="right")
    route_tree = cKDTree(path)
    owner_grid = np.where(active, owners, -1).astype(np.int8)
    mesh_z = ground

    for lo, hi in patches(is_fill_grid.shape, GEOM_TILE):
        sl = region(lo, hi)
        own = owner_grid[sl]
        if (own < 0).all():
            continue
        z = mesh_z[region(lo, hi + 1)]
        v, faces = grid_geometry(z, base + lo * PITCH)
        fill = is_fill_grid[sl]
        for i in np.unique(own[own >= 0]):
            for nominal in (False, True):
                selected = (own == i) & (fill == nominal)
                if not selected.any():
                    continue
                use = faces[selected.ravel()].reshape(-1, 3)
                ids, fi = np.unique(use, return_inverse=True)
                fi = fi.reshape(-1, 3)
                vi = v[ids]
                center = np.round(vi.mean(0), 5)
                label = "filled_ground" if nominal else "observed_ground"
                name = f"s{i:02d}_{label}_{lo[0]}_{lo[1]}"
                file = f"terrain/{name}.obj"
                directory = scene_dir(out, int(i))
                compose.write_obj(directory / file, vi - center, fi)
                add_terrain(trees[i], name, file, center, nominal)
                recordings[i].append((vi, fi))
                m = metadata[i]
                aggregate(m["surface"].setdefault(label, {}), edge_statistics(vi, fi))
                m["ground_projected_m2"] += selected.sum() * PITCH ** 2
                if nominal:
                    m["fill_projected_m2"] += selected.sum() * PITCH ** 2
                    select_station = stations[sl][selected]
                    spans, counts = np.unique((select_station // 5).astype(int), return_counts=True)
                    for span, count in zip(spans, counts):
                        key = str(5 * span)
                        m["fill_area_by_5m"][key] = m["fill_area_by_5m"].get(key, 0) + count * PITCH**2
                    xy = (np.argwhere(selected) + lo + .5) * PITCH + base
                    m["fill_near_route_m2"] += (route_tree.query(xy)[0] <= 1).sum() * PITCH**2

    cor = Corridor(path, width=6.)
    def owner(point):
        s = cor.project(np.atleast_2d(point))[0][0]
        return int(np.searchsorted(ranges[1:-1], s, side="right"))

    for j, stair in enumerate(stairs):
        v, f = stair["vertices"], stair["faces"]
        i = owner(v.mean(0)[:2])
        name = f"stair_{j:04d}"
        center = v.mean(0)
        file = f"stairs/{name}.obj"
        compose.write_obj(scene_dir(out, i) / file, v - center, f)
        compose.place_hard(trees[i], dict(name=name, file=file, material="stone", group=2,
                                         geom=dict(type="mesh", pos=center)))
        aggregate(metadata[i]["surface"].setdefault("stair_solids", {}), edge_statistics(v, f))
        recordings[i].append((v, f[[2, 3, 10, 11]]))

    for obj in objects:
        kind = obj.get("kind", "rail")
        geom = obj["geom"]
        p = np.asarray(geom["pos"] if "pos" in geom else np.asarray(geom["fromto"]).reshape(2, 3).mean(0))
        i = owner(p[:2])
        if geom["type"] == "mesh":
            directory = scene_dir(out, i)
            if "vertices" in obj:
                v, f = obj["vertices"], obj["triangles"]
                compose.write_obj(directory / obj["file"], v, f)
            else:
                shutil.copyfile(OUT / obj["file"], directory / obj["file"])
                v, f = raw_obj(directory / obj["file"])
            label = ("visual_vegetation" if kind == "soft" else "walls_and_rocks" if obj["kind"] in ("retaining_wall", "wall_rock_or_root", "rock_face")
                     else "woody_solids")
            aggregate(metadata[i]["surface"].setdefault(label, {}), edge_statistics(v, f))
        counts = metadata[i]["objects"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "soft":
            place_soft(trees[i], obj)
        else:
            compose.place_hard(trees[i], obj)

    stats_arrays = np.zeros((n, 2, 3), np.int64)
    unique, counts = np.unique(owner_grid[active], return_counts=True)
    # Exact cell ownership must cover every active face once across sections.
    for i, count in zip(unique, counts):
        expected = metadata[i]["ground_projected_m2"] / PITCH**2
        assert int(round(expected)) == count, (i, expected, count)
    a = owner_grid[:, :-1]
    b = owner_grid[:, 1:]
    c = owner_grid[:-1, :]
    d = owner_grid[1:, :]
    for dim, left, right in ((0, c, d), (1, a, b)):
        loc = (left >= 0) & (right >= 0) & (left != right)
        for i, j in zip(left[loc], right[loc]):
            stats_arrays[i, dim, 0] += 1
            stats_arrays[j, dim, 0] += 1
    sample_station, _, selected = cor.project(samples[:, :2])
    # Corridor.project uses the true distance and clips near the endpoints.
    is_near = route_tree.query(samples[:, :2], workers=4)[0] <= 1.
    sample_idx = np.searchsorted(ranges[1:-1], sample_station, side="right")
    for i, (tree, result) in enumerate(zip(trees, metadata)):
        target = scene_dir(out, i)
        compose.save_xml(tree, target / "scene.xml")
        result["pitch_m"] = PITCH
        result["shared_boundary_edges"] = int(stats_arrays[i, :, 0].sum())
        result["xy_bounds_m"] = [np.min([v[:, :2].min(0) for v, _ in recordings[i]], axis=0),
                                 np.max([v[:, :2].max(0) for v, _ in recordings[i]], axis=0)]
        result["max_seam_xyz_disagreement_m"] = 0.0
        if out == PILOT:
            available = recordings[i]
        else:
            available = sum(recordings[max(0, i-1):min(n, i+2)], [])
        for label, mask in (("corridor", selected), ("within_1m", selected & is_near)):
            result[f"ground_holdout_{label}"] = point_mesh_distances(
                available, samples[(sample_idx == i) & mask])
        result["bytes"] = sum(p.stat().st_size for p in target.rglob("*.obj")) + (target/"scene.xml").stat().st_size
        write_json(target / "geometry.json", result)
        print("geometry", i, result["ground_holdout_within_1m"], result["ground_holdout_corridor"], flush=True)
    write_json(out / "segments.json", metadata)
    return metadata


def old_audit(path, samples):
    root = ET.parse(OUT / "scene.xml").getroot()
    mapping = {node.get("name"): node.get("file") for node in root.find("asset").findall("mesh")}
    stats = {}
    meshes = []
    by_station = {}
    cor = Corridor(path)
    for node in root.find("worldbody").findall("geom"):
        name = node.get("name")
        if name not in mapping:
            continue
        v, f = raw_obj(OUT / mapping[name])
        v += np.fromstring(node.get("pos"), sep=" ")
        label = ("observed_and_filled_terrain" if name.startswith("terrain_") else
                 "stair_solids" if name.startswith("stair_") else
                 "walls_and_rocks" if name.startswith(("wall_", "solid_")) else "woody_solids")
        aggregate(stats.setdefault(label, {}), edge_statistics(v, f))
        if name.startswith("terrain_"):
            meshes.append((v, f))
            xyz = v[f]
            normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            vertical = abs(normal[:, 2]) < np.linalg.norm(normal, axis=1)*np.cos(np.deg2rad(75))
            st, lat, _ = cor.project(xyz[vertical, :, :2].mean(1))
            for a, b in zip(st, lat):
                k = f"{int(a//5)*5:02d}m {'left' if b >= 0 else 'right'} {'near' if abs(b) <= 1 else 'side'}"
                by_station[k] = by_station.get(k, 0) + 1
        elif name.startswith("stair_"):
            meshes.append((v, f[[2, 3, 10, 11]]))
    _, _, selected = cor.project(samples[:, :2])
    is_near = cKDTree(path).query(samples[:, :2], workers=4)[0] <= 1.
    return dict(surface=stats, vertical_terrain_by_5m=by_station,
                ground_holdout_corridor=point_mesh_distances(meshes, samples[selected]),
                ground_holdout_within_1m=point_mesh_distances(meshes, samples[selected & is_near]))


def depress_under_stairs(z, base, stairs):
    underlay = z.copy()
    for step in stairs:
        poly = step["polygon"]
        lo = np.maximum(np.floor((poly.min(0)-base)/PITCH).astype(int)-1, 0)
        hi = np.minimum(np.ceil((poly.max(0)-base)/PITCH).astype(int)+2, z.shape)
        ij = np.indices(hi-lo).reshape(2, -1).T + lo
        xy = ij*PITCH+base
        slab = step["vertices"][4:]
        coef = np.linalg.lstsq(np.c_[slab[:, :2], np.ones(len(slab))], slab[:, 2], rcond=None)[0]
        use = in_polygon(xy, poly, margin=.01)
        ids = tuple(ij[use].T)
        top = xy[use]@coef[:2]+coef[2]
        underlay[ids] = np.minimum(underlay[ids], top-.035)
        edge = step["edge"]
        span = edge[1]-edge[0]
        norm = np.linalg.norm(span)
        forward = np.array([span[1], -span[0]])/norm
        d = (xy-edge.mean(0))@forward
        across = (xy-edge[0])@span/(norm*norm)
        apron = (d >= -.060) & (d < 0) & (across > 0) & (across < 1)
        ids = tuple(ij[apron].T)
        prior = step["before_plane_coeff"]
        height = (xy[apron]-step["before_plane_anchor"])@prior[:2]+prior[2]
        underlay[ids] = np.minimum(underlay[ids], height-.005)
    return underlay


def main(pilot):
    start = time.monotonic()
    out, ranges = (PILOT, [0., 30.]) if pilot else (FULL, LIMITS)
    out.mkdir(parents=True, exist_ok=True)
    _, xy = route()
    path = xy[:601] if pilot else xy
    stairs, objects = load_pilot_objects()
    v8 = V8Map.load()
    key = "pilot" if pilot else "full"
    cache = WORK / f"{key}_raster_input.npz"
    if not cache.is_file():
        base, z, gap, active, stations, samples, rejects = make_ground(
            path, v8, pilot_cloud() if pilot else None)
        np.savez(cache, base=base, z=z, gap=gap, active=active, stations=stations,
                 samples=samples, rejects=rejects)
    else:
        with np.load(cache) as f:
            base, z, gap, active, stations, samples, rejects = [f[k] for k in (
                "base", "z", "gap", "active", "stations", "samples", "rejects")]
    anchors = route_anchors(path, base, active, gap)
    natural = apply_grades(z, active, anchors)
    if not pilot:
        from nav4.matrix.full_stairs import survey_stairs
        stairs += survey_stairs(path, v8, natural, base)
    if not pilot:
        from nav4.matrix.full_objects import detect_objects
        objects += detect_objects(path, v8, natural, base)
    elif Path(__file__).with_name("full_objects.py").exists():
        from nav4.matrix.full_objects import detect_objects
        objects += detect_objects(path, v8, natural, base, pilot=True)
    depressed = depress_under_stairs(natural, base, stairs)
    terrain = apply_grades(depressed, active, anchors)
    np.savez(out / "surface.npz", origin=base, pitch=PITCH, z=terrain, gap=gap, active=active, station=stations)
    write_json(out / "stair_geometry.json", stairs)
    items = [{k: v for k, v in obj.items() if k not in ("vertices", "triangles")} for obj in objects]
    write_json(out / "objects.json", items)
    make_scenes(out, path, ranges, base, terrain, gap, active, stations, stairs, objects, samples)
    if pilot:
        write_json(out / "pilot_before_on_same_samples.json", old_audit(path, samples))
    write_json(out / "build.json", dict(seconds=time.monotonic()-start,
                                        rejected_observation_patch_incidents=int(rejects)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    main(args.pilot)
