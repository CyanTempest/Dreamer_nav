"""CPU scene loads, v8 comparisons and local M1-envelope contact tests."""
import argparse
import csv
import functools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nav4.mapping.common import write_json
from nav4.matrix.compose import xml
from nav4.matrix.full_ground import FULL, V8Map
from nav4.matrix.full_package import assemble
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import CONFIG, LIMITS
from nav4.matrix.full_visual import measure_compile
from nav4.matrix.register import summary
from nav4.matrix.stairs import heading, path_at

GEOM_GROUND = np.array([1, 0, 1, 0, 1, 0], np.uint8)
GEOM_HARD = np.array([0, 1, 0, 0, 0, 0], np.uint8)
GEOM_TRAVERSE = np.array([1, 1, 1, 0, 1, 0], np.uint8)
BAND = (.30, .53, .76)
RANGE = 2.


def cast(model, data, origin, direction, groups, result):
    return mujoco.mj_ray(model, data, origin, direction, groups, 1, -1, result)


def floor_z(model, data, xy):
    key = np.empty(1, np.int32)
    hit = cast(model, data, np.r_[xy, 100.], np.array([0., 0., -1.]), GEOM_GROUND, key)
    return 100.-hit if hit >= 0 else np.nan


def cross_clearance(model, data, xy, normal, floor, groups):
    result = np.empty(1, np.int32)
    names = [None, None]
    edge = np.full(2, RANGE)
    for side, sign in enumerate((1., -1.)):
        axis = np.r_[sign*normal, 0.]
        for h in BAND:
            hit = cast(model, data, np.r_[xy, floor+h], axis, groups, result)
            if 0 <= hit < edge[side]:
                edge[side] = hit
                names[side] = model.geom(result[0]).name
    return edge, names


def loaded_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def static_scene(i):
    return FULL/"segments"/f"seg_{i:02d}"/"scene.xml"


def record_rays(row, model, data, point, normal):
    z = floor_z(model,data,point)
    hard, names = cross_clearance(model,data,point,normal,z,GEOM_HARD) if np.isfinite(z) else (
        [np.nan,np.nan],[None,None])
    free, block = cross_clearance(model,data,point,normal,z,GEOM_TRAVERSE) if np.isfinite(z) else (
        [np.nan,np.nan],[None,None])
    row.update(scene_z_m=z,signed_ground_delta_m=z-row["v8_z_m"],width_m=float(sum(free)),
               terrain_plus_hard_left_m=free[0],terrain_plus_hard_right_m=free[1],
               left_geom=block[0],right_geom=block[1],
               scan_hard_left_m=hard[0],scan_hard_right_m=hard[1],
               scan_hard_left_geom=names[0],scan_hard_right_geom=names[1])


def centerline_comparisons(path, only_last=False):
    v8 = V8Map.load()
    rows = []
    fits = []
    if only_last:
        with (FULL/"route_vs_v8.csv").open() as stream:
            for raw in csv.DictReader(stream):
                if int(raw["segment"]) == len(LIMITS)-2:
                    continue
                row = {}
                for key,value in raw.items():
                    if key == "seam_local_rechecked":
                        row[key] = value == "True"
                    elif key.endswith("_geom") or key in ("left_geom","right_geom"):
                        row[key] = value or None
                    elif key == "segment":
                        row[key] = int(value)
                    else:
                        row[key] = float(value)
                rows.append(row)
        fits = json.loads((FULL/"load_segments.json").read_text())[:-1]
    for i, (lo, hi) in enumerate(zip(LIMITS[:-1], LIMITS[1:])):
        if only_last and i != len(LIMITS)-2:
            continue
        model, stats = measure_compile(static_scene(i))
        scene = static_scene(i).parent
        stats["scene_bytes"] = sum(k.stat().st_size for k in scene.rglob("*.obj"))+static_scene(i).stat().st_size
        write_json(scene/"load.json", stats)
        fits.append(stats)
        d = loaded_data(model)
        # Section scenes own unique faces. At boundaries the neighboring
        # section is incorporated in the subsequent local contact checks.
        ss = np.arange(lo, hi if hi < LIMITS[-1] else hi+.01, .10)
        xy = path_at(path, ss)
        a = heading(path, ss)
        n = np.c_[-a[:, 1], a[:, 0]]
        old_ground = v8.z(xy)
        grid_offset = np.arange(0., RANGE+.01, .05)
        for st, p, side, g8 in zip(ss, xy, n, old_ground):
            prior = []
            for sign in (1, -1):
                ray = p + sign*grid_offset[:, None]*side
                nearest = np.flatnonzero(v8.classes(ray) == 1)
                prior.append(grid_offset[nearest[0]] if len(nearest) else RANGE)
            row = dict(segment=i, station_m=round(float(st), 3), v8_z_m=g8,
                       v8_hard_left_m=prior[0], v8_hard_right_m=prior[1],
                       boundary_margin_m=min(st-lo, hi-st), seam_local_rechecked=False)
            record_rays(row, model, d, p, side)
            rows.append(row)
        print("loaded segment", i, stats["seconds"], flush=True)
        del d, model

    seams = LIMITS[-2:-1] if only_last else LIMITS[1:-1]
    for seam in seams:
        i = int(np.searchsorted(LIMITS[1:-1], seam))
        close = [r for r in rows if abs(r["station_m"]-seam) < 2.05]
        mid = path_at(path, seam)
        root = contact_local(mid, seam, padding=2.2)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = loaded_data(model)
        for r in close:
            p = path_at(path, r["station_m"])
            a = heading(path, r["station_m"])
            record_rays(r,model,d,p,np.r_[-a[1],a[0]])
            r["seam_local_rechecked"] = True
        del d, model

    with (FULL/"route_vs_v8.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    dz = np.array([a["signed_ground_delta_m"] for a in rows])
    fit = np.isfinite(dz)
    best = []
    for idx in np.argsort(-np.where(fit, abs(dz), -np.inf)):
        row = rows[idx]
        if any(abs(t["station_m"]-row["station_m"]) < 1.5 for t in best):
            continue
        best.append(row)
        if len(best) == 16:
            break
    hard_report = {}
    for side in ("left", "right"):
        new = np.asarray([r[f"scan_hard_{side}_m"] for r in rows])
        old = np.asarray([r[f"v8_hard_{side}_m"] for r in rows])
        good = np.isfinite(new)
        both = good & (new < RANGE) & (old < RANGE)
        anomalies = []
        for j in np.argsort(-np.where(good, abs(new-old), -np.inf)):
            row = rows[j]
            if any(abs(t["station_m"]-row["station_m"]) < 3.0 for t in anomalies):
                continue
            anomalies.append(row)
            if len(anomalies) == 12:
                break
        hard_report[side] = dict(both=int(both.sum()), only_v8=int((good&(new>=RANGE)&(old<RANGE)).sum()),
                                 only_scene=int((good&(new<RANGE)&(old>=RANGE)).sum()),
                                 neither=int((good&(new>=RANGE)&(old>=RANGE)).sum()),
                                 both_signed_median_m=float(np.median(new[both]-old[both])) if both.any() else None,
                                 largest_differences=anomalies)
    per_seg = {}
    for i in range(len(LIMITS)-1):
        take = [r["signed_ground_delta_m"] for r in rows if r["segment"] == i and np.isfinite(r["scene_z_m"])]
        per_seg[str(i)] = summary(np.asarray(take))
    write_json(FULL/"v8_difference.json", dict(ground=summary(dz[fit]), ground_per_segment=per_seg,
                                                 missing_ground_rays=int((~fit).sum()), ground_outliers=best,
                                                 hard_by_side=hard_report, body_band_m=BAND, lateral_cap_m=RANGE))
    write_json(FULL/"load_segments.json", fits)
    return rows


@functools.lru_cache(maxsize=1)
def assembled_contact_root(segment):
    lower, upper = max(0, segment-1), min(len(LIMITS)-1, segment+2)
    return assemble(range(lower, upper), FULL/"segments", Path("segments"))


def contact_local(xy, station, padding=0.):
    segment = int(np.searchsorted(LIMITS[1:-1], station, side="right"))
    mixed = assembled_contact_root(segment)
    root = ET.Element("mujoco", model="M1 static and slow push")
    for sections in ("compiler", "option", "statistic"):
        root.append(mixed.find(sections))
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    keys = set()
    material = set()
    for node in mixed.find("worldbody"):
        if node.tag not in {"flexcomp", "geom"} or node.get("group") == "5":
            continue
        center = np.fromstring(node.get("pos", ""), sep=" ")
        if node.get("fromto"):
            center = np.fromstring(node.get("fromto"), sep=" ").reshape(2, 3).mean(0)
        limit = 5.2 if node.tag == "flexcomp" or node.get("group") in ("0", "4") else 2.70
        if not len(center) or np.linalg.norm(xy-center[:2]) > limit+padding:
            continue
        copied = ET.fromstring(ET.tostring(node))
        if copied.get("file"):
            copied.set("file", str((FULL/copied.get("file")).resolve()))
        world.append(copied)
        if node.get("mesh"):
            keys.add(node.get("mesh"))
        if node.get("material"):
            material.add(node.get("material"))
    for node in mixed.find("asset"):
        if node.get("name") not in keys and node.get("name") not in material:
            continue
        copy = ET.fromstring(ET.tostring(node))
        if copy.get("file"):
            copy.set("file", str((FULL/copy.get("file")).resolve()))
        assets.append(copy)
    return root


def proxy_case(case):
    cfg = CONFIG["physics"]
    axis = case["forward"]
    xy = case["xy"]
    half = np.asarray(cfg["box_half_size_m"])
    root = contact_local(xy, case["station_m"])
    world = root.find("worldbody")
    body = xml(world, "body", name="contact_proxy")
    xml(body, "freejoint", name="proxy_joint")
    xml(body, "geom", type="box", name="proxy_box", size=half, mass=cfg["mass_kg"],
        friction=[.12, .001, .0001], priority=2, solref=[.007, 1.], contype=2, conaffinity=1, group=5)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = loaded_data(model)
    g = model.geom("proxy_box").id
    body_id = model.body("contact_proxy").id
    side = np.array([-axis[1], axis[0]])
    corners = np.array([xy+x*axis+y*side for x in (-half[0], 0, half[0]) for y in (-half[1], 0, half[1])])
    support = np.array([floor_z(model, d, p) for p in corners])
    if not np.isfinite(support).any():
        return dict(status="missing terrain at spawn")
    yaw = np.arctan2(axis[1], axis[0])
    q = d.qpos[:3]
    q[:] = np.r_[xy, np.nanmax(support)+half[2]+.014]
    d.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    init = q.copy()
    mujoco.mj_forward(model, d)

    def run(duration, push):
        start = q.copy()
        normal_force = np.zeros(6)
        stats = dict(min_contact_gap_m=0., support_steps=0, hard_contact_steps=0, hard_geoms={},
                     side_force_n_max=0., peak_center_gain_m=0., min_center_minus_ground_m=np.inf,
                     max_forward_speed_mps=0., max_backward_speed_mps=0., braking_steps=0)
        hard_touch = stats["hard_geoms"]
        for step in range(round(duration/model.opt.timestep)):
            if push:
                speed = d.qvel[:2]@axis
                force = np.clip((.095-speed)*1750, -330., cfg["push_n"])
                side_force = np.clip(-1100.*(d.qvel[:2]@side), -140., 140.)
                d.xfrc_applied[body_id, :2] = force*axis+side_force*side
                stats["braking_steps"] += bool(force < 0)
            mujoco.mj_step(model, d)
            speed = float(d.qvel[:2]@axis)
            stats["max_forward_speed_mps"] = max(stats["max_forward_speed_mps"], speed)
            stats["max_backward_speed_mps"] = max(stats["max_backward_speed_mps"], -speed)
            weight = False
            hard = False
            for k in range(d.ncon):
                c = d.contact[k]
                if c.efc_address < 0 or g not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == g else c.geom1
                stats["min_contact_gap_m"] = min(stats["min_contact_gap_m"], float(c.dist))
                mujoco.mj_contactForce(model, d, k, normal_force)
                n = c.frame[:3]
                weight |= abs(n[2]) > .55 and normal_force[0] > 2.
                if abs(n[2]) < .60 and normal_force[0] > 3.:
                    hard = True
                    name = model.geom(other).name if other >= 0 else "terrain_flex"
                    hard_touch[name] = hard_touch.get(name, 0)+1
                    force = abs(n[:2]@axis)*normal_force[0]
                    stats["side_force_n_max"] = max(stats["side_force_n_max"], float(force))
            stats["support_steps"] += bool(weight)
            stats["hard_contact_steps"] += hard
            stats["peak_center_gain_m"] = max(stats["peak_center_gain_m"], float(q[2]-start[2]))
            if step % 25 == 0:
                fz = floor_z(model, d, q[:2])
                if np.isfinite(fz):
                    stats["min_center_minus_ground_m"] = min(stats["min_center_minus_ground_m"], float(q[2]-fz))
        if not np.isfinite(stats["min_center_minus_ground_m"]):
            stats["min_center_minus_ground_m"] = None
        return dict(**stats, forward_m=float((q-start)[:2]@axis), lateral_m=float((q-start)[:2]@side),
                    vertical_m=float(q[2]-start[2]), pose=d.qpos.copy(), seconds=duration)

    initial_qpos = d.qpos.copy()
    settle = run(cfg["settle_s"], False)
    # Gravity may cause the freely resting box to slide beyond the intended
    # site. Start a separate controlled push from the same placement.
    mujoco.mj_resetData(model,d)
    d.qpos[:] = initial_qpos
    mujoco.mj_forward(model,d)
    push = run(cfg["push_s"], True)
    return dict(initial_center_m=init, settle=settle, push=push, local_triangles=model.nflexelem,
                max_contact_overlap_m=-min(settle["min_contact_gap_m"], push["min_contact_gap_m"]),
                center_crossed_below_terrain=any(a["min_center_minus_ground_m"] is not None and
                                                a["min_center_minus_ground_m"] < -.06 for a in (settle,push)),
                stalled=push["forward_m"] < .15,
                unexpected_climb_candidate=push["peak_center_gain_m"] > .14 and push["hard_contact_steps"] > 10)


def stairs_cases(path):
    data = json.loads((FULL/"stairs.json").read_text())["accepted"]
    shapes = json.loads((FULL/"stair_geometry.json").read_text())
    starts = {}
    pilot_accepted = [a for a in data if a["origin"] == "pilot_dense_LAS"]
    n = len(pilot_accepted)
    for row, shape in zip(pilot_accepted, shapes[:n]):
        starts.setdefault(row["flight"], []).append((row, shape))
    lookup = {round(r["station_m"], 4): r for r in data if r["origin"] == "dense_LAS_survey"}
    for shape in shapes[n:]:
        row = lookup[round(shape["station_m"], 4)]
        starts.setdefault(row["flight"], []).append((row, shape))
    tests = []
    for label, rows in starts.items():
        # The long descent is checked at distinct grade/landing subflights.
        positions = [0]
        for j, (row, _) in enumerate(rows[1:], 1):
            last = positions[-1]
            if (row["station_m"]-rows[last][0]["station_m"] >= 8.0 or
                    row["station_m"]-rows[j-1][0]["station_m"] >= 1.35):
                positions.append(j)
        for n, i in enumerate(positions):
            row, shape = rows[i]
            station = row["station_m"]
            endpoints = np.asarray(shape["edge"])
            span = endpoints[1]-endpoints[0]
            axis = np.r_[span[1], -span[0]]/np.linalg.norm(span)
            if axis @ heading(path, station) < 0:
                axis *= -1
            tests.append(dict(name=f"stairs_{label}" if n == 0 else f"stairs_{label}_part{n+1}",
                              kind="stairs", station_m=station, riser_m=row["riser_m"],
                              fitted_steps=len(rows), forward=axis, xy=endpoints.mean(0)-.58*axis))
    return tests


def narrow_cases(rows, path):
    used = []
    for row in sorted((r for r in rows if np.isfinite(r["width_m"]) and .2 < r["station_m"] < 721.8),
                      key=lambda a: a["width_m"]):
        s = row["station_m"]
        if any(abs(s-a["station_m"]) < 4.0 for a in used):
            continue
        used.append(row)
        if len(used) == 5:
            break
    return [dict(name=f"narrow_{n+1}", kind="narrow", station_m=r["station_m"], measured_width_m=r["width_m"],
                 constraining_geoms=[r["left_geom"], r["right_geom"]],
                 xy=path_at(path,r["station_m"]-.34), forward=heading(path,r["station_m"])) for n,r in enumerate(used)]


def load_rows():
    with (FULL/"route_vs_v8.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in ("scene_z_m", "station_m", "width_m", "boundary_margin_m"):
            r[k] = float(r[k])
    return rows


def physics(path, rows):
    cases = []
    for case in stairs_cases(path)+narrow_cases(rows,path):
        result = proxy_case(case)
        cases.append(dict(case, **result))
        print("physics", case["name"], result.get("max_contact_overlap_m"), result.get("stalled"), flush=True)
        write_json(FULL/"physics.json", dict(cases=cases, body_band_m=BAND, contact_settings=CONFIG["physics"],
                                               proxy_envelope_length_width_m=[1.03, .58],
                                               push_control=dict(target_mps=.095, forward_limit_n=140,
                                                                 braking_limit_n=330, lateral_limit_n=140,
                                                                 restart_from_initial_pose=True)))


def assembled_loads(whole, entry=None):
    output = FULL/"load_assemblies.json"
    payload = json.loads(output.read_text()) if entry and output.exists() else {}
    matrix = FULL/"matrix/UeSim/Content/model/zgws"
    todo = [FULL/"window_00.xml", matrix/"zgws_atec_window_00.xml"]
    if whole or entry:
        todo += [FULL/"scene_full.xml", matrix/"zgws_atec_full.xml"]
    if entry:
        todo = [p for p in todo if p.name == entry]
        if not todo:
            raise ValueError(f"Unknown entry: {entry}")
    for entry in todo:
        model, stats = measure_compile(entry)
        stats["actuators"] = model.nu
        if model.nu:
            d = loaded_data(model)
            stats["base_link_origin_m"] = d.body("base_link").xpos.copy()
            stats["base_link_quat"] = d.body("base_link").xquat.copy()
            del d
        payload[entry.name] = stats
        del model
        write_json(FULL/"load_assemblies.json", payload)
        print(entry.name, stats, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["compare", "tail-compare", "physics", "assemblies", "all-assemblies"])
    p.add_argument("--entry")
    args = p.parse_args()
    phase = args.phase
    if phase.endswith("assemblies"):
        assembled_loads(phase == "all-assemblies", args.entry)
    elif phase.endswith("compare"):
        centerline_comparisons(route()[1], phase=="tail-compare")
    else:
        physics(route()[1], load_rows())


if __name__ == "__main__":
    main()
