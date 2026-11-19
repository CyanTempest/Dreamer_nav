"""Benchmark released ZGWS assemblies, one model per process and one CPU core."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nav4.matrix.full_scan import FULL, route
from nav4.matrix.stairs import heading, path_at

V2 = FULL.with_name("full_v2")
MODEL = Path("matrix/UeSim/Content/model/zgws")
# An open landing, an ascending flight, and the narrowest measured site.
SITES = [("landing", .4), ("flight_2", 10.4), ("narrow_1", 6.2)]


def hold(model, data, target):
    q = data.qpos[7:]
    v = data.qvel[6:]
    torque = 190 * (target - q) - 7 * v
    data.ctrl[:] = np.clip(torque, -140, 140)
    data.ctrl[3::4] = np.clip(torque[3::4], -38, 38)


def benchmark(version, label, resume=False):
    assert mujoco.__version__ == "3.3.0"
    root = FULL if version == "v1" else V2
    _, path = route()
    shift = np.array(json.loads((FULL / "assembly.json").read_text())["shared_shift_m"])
    with (FULL / "route_vs_v8.csv").open() as stream:
        floors = {round(float(r["station_m"]), 1): float(r["scene_z_m"]) for r in csv.DictReader(stream)}
    entry = root / MODEL / f"zgws_atec_{label}.xml"
    start = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(str(entry))
    d = mujoco.MjData(m)
    assert abs(m.opt.timestep - .002) < 1e-12
    output = dict(version=mujoco.__version__, entry=str(entry), load_s=time.perf_counter()-start,
                  nflexelem=m.nflexelem, ngeom=m.ngeom, sites=[], timestep_s=m.opt.timestep,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), bvactive=int(m.vis.global_.bvactive),
                  warmup_steps=100)
    print("compiled", output, flush=True)
    source_key = ET.parse(FULL / MODEL / f"zgws_atec_robot_{label}.xml").find("./keyframe/key[@name='home']")
    target = np.fromstring(source_key.get("qpos"), sep=" ")[7:]
    result_path = V2 / "work" / f"bench_{version}_{label}.json"
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        assert previous["entry"]==str(entry) and previous["version"]==mujoco.__version__
        output["sites"] = previous["sites"]
        output["previous_load_s"] = previous["load_s"]
    for kind, station in SITES:
        if any(row["kind"]==kind and row["steps"]==2000 for row in output["sites"]):
            continue
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[7:] = target
        d.qpos[:3] = np.r_[path_at(path, station), floors[station] + .66] - shift
        yaw = np.arctan2(*heading(path, station)[::-1])
        d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(m, d)
        for _ in range(100):
            hold(m, d, target)
            mujoco.mj_step(m, d)
        elapsed = 0
        counts = []
        start_pos = d.qpos[:3].copy()
        for i in range(2000):
            hold(m, d, target)
            tick = time.perf_counter_ns()
            mujoco.mj_step(m, d)
            elapsed += time.perf_counter_ns() - tick
            counts.append(d.ncon)
            if i in (0, 199, 999):
                print(kind, i+1, elapsed/(i+1)/1e6, d.ncon, flush=True)
        ms = elapsed/2000/1e6
        output["sites"].append(dict(kind=kind, station_m=station, steps=len(counts), ms=ms,
                                    ncon=float(np.mean(counts)), contact_range=[min(counts),max(counts)],
                                    realtime=2/ms, displacement_m=(d.qpos[:3]-start_pos).tolist(),
                                    warnings=d.warning.number.tolist()))
        output["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_path.write_text(json.dumps(output, indent=2))
        print(output["sites"][-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=["v1", "v2"])
    parser.add_argument("label", choices=["window_00", "full"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    benchmark(args.version, args.label, args.resume)
