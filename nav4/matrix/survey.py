"""Stream the official points into a 0–30 m window, retaining original attributes."""
import argparse
import time
import zipfile
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import PLANNING_PLY, LIDAR_ZIP, open_ply, write_json

V8 = Path("data/trail_a_v8_frozen/forward")
OUT = Path("data/trail_a_mujoco/pilot_0_30")
MEMBERS = {
    "removed": "01_LiDAR_point_cloud/developer_data/map_remov.las",
    "map0": "01_LiDAR_point_cloud/map-result/0.las",
}


def route_data():
    with np.load(V8 / "map.npz") as f:
        print({key: f[key].shape for key in ("route", "s", "yaw", "origin")}, flush=True)
        s = f["s"]
        route = f["route"]
        station = np.linspace(0, 30, 601)
        return station, np.column_stack([np.interp(station, s, axis) for axis in route[:, :2].T])


class Corridor:
    def __init__(self, path, width=6.0):
        self.path = path
        self.width = width
        self.edges = np.diff(path, axis=0)
        self.length2 = np.sum(self.edges**2, axis=1)
        self.tree = cKDTree((path[1:] + path[:-1]) / 2)

    def project(self, xy):
        _, ix = self.tree.query(xy, k=4)
        rel = xy[:, None] - self.path[ix]
        edge = self.edges[ix]
        u = np.einsum("ijk,ijk->ij", rel, edge) / self.length2[ix]
        v = np.clip(u, 0, 1)
        delta = rel - v[..., None] * edge
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        row = np.arange(len(xy))
        closest = d2.argmin(axis=1)
        i = ix[row, closest]
        t = u[row, closest]
        along = edge[row, closest]
        across = np.array([-along[:, 1], along[:, 0]]).T
        lateral = np.sum(delta[row, closest] * across, axis=1) / np.sqrt(self.length2[i])
        in_ends = ~(((i == 0) & (t < 0)) | ((i == len(self.edges)-1) & (t > 1)))
        return (i + v[row, closest]) * .05, lateral, (d2[row, closest] <= self.width**2) & in_ends


def density(points, corridor, path):
    # Area from 10 cm cell-center quadrature; counts use the exact curved corridor.
    r = .1
    axes = [np.arange(a-6, b+6, r) for a, b in zip(path.min(0), path.max(0))]
    xy = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    station, _, selected = corridor.project(xy)
    mask = np.array([np.sum(selected & (station >= a) & (station <= a+5)) * r*r
                     for a in range(0, 30, 5)])
    position = np.load(OUT / f"{points}_crop.npy", mmap_mode="r")
    counts = np.zeros(6, dtype=int)
    for chunk in np.array_split(position, max(1, len(position)//100000)):
        s, _, use = corridor.project(np.column_stack([chunk["x"], chunk["y"]]))
        counts += np.histogram(s[use], bins=np.arange(0, 31, 5))[0]
    return dict(area_m2=selected.sum()*r*r, points=int(counts.sum()),
                density_per_m2=float(counts.sum()/(selected.sum()*r*r)),
                station_5m_density_per_m2=counts/mask)


def crop(name, corridor, path):
    start = time.monotonic()
    chunks = []
    margin = corridor.width
    low, high = path.min(0)-margin, path.max(0)+margin
    count = 0

    def select(x, y):
        idx = np.flatnonzero((x >= low[0]) & (x <= high[0]) & (y >= low[1]) & (y <= high[1]))
        if not len(idx):
            return idx
        _, _, keep = corridor.project(np.column_stack([x[idx], y[idx]]))
        return idx[keep]

    if name == "ply":
        source = open_ply(PLANNING_PLY)
        fields = ["x", "y", "z", "normal_x", "normal_y", "normal_z", "classification", "intensity"]
        dtype = [(f, source.dtype[f]) for f in fields]
        for i in range(0, len(source), 1_000_000):
            block = source[i:i+1_000_000]
            idx = select(block["x"], block["y"])
            if len(idx):
                result = np.empty(len(idx), dtype)
                for f in fields:
                    result[f] = block[f][idx]
                chunks.append(result)
        count = len(source)
        header = dict(fields=source.dtype.names)
    else:
        p = PLANNING_PLY.parents[2] / MEMBERS[name]
        with laspy.open(p) as las:
            h = las.header
            count = h.point_count
            header = dict(version=str(h.version), format=h.point_format.id, fields=list(h.point_format.dimension_names),
                          mins=h.mins, maxs=h.maxs, offsets=h.offsets, scales=h.scales)
            print(name, header, flush=True)
            fields = ["x", "y", "z", "classification", "intensity"]
            fields.extend(f for f in ("red", "green", "blue", "gps_time") if f in header["fields"])
            dtype = [(f, "f4" if f in ("x", "y", "z") else "f8" if f == "gps_time" else "u2")
                     for f in fields]
            for i, block in enumerate(las.chunk_iterator(1_000_000)):
                x = np.asarray(block.X)*h.scales[0] + h.offsets[0]
                y = np.asarray(block.Y)*h.scales[1] + h.offsets[1]
                idx = select(x, y)
                if len(idx):
                    result = np.empty(len(idx), dtype)
                    result["x"], result["y"] = x[idx], y[idx]
                    result["z"] = block.Z[idx]*h.scales[2] + h.offsets[2]
                    for f in fields[3:]:
                        result[f] = getattr(block, f)[idx]
                    chunks.append(result)
                if (i+1) % 200 == 0:
                    print(name, (i+1)*1_000_000, sum(map(len, chunks)), flush=True)
    if not chunks:
        raise ValueError(f"No {name} overlap in the v8 window")
    cloud = np.lib.format.open_memmap(OUT / f".{name}_crop.partial.npy", mode="w+", dtype=dtype,
                                      shape=(sum(map(len, chunks)),))
    offset = 0
    for chunk in chunks:
        cloud[offset:offset+len(chunk)] = chunk
        offset += len(chunk)
    del cloud
    (OUT / f".{name}_crop.partial.npy").replace(OUT / f"{name}_crop.npy")
    stats = density(name, corridor, path)
    write_json(OUT / f"{name}_crop.json", dict(header=header, full_count=count, **stats,
                                              seconds=time.monotonic()-start))
    print(name, stats, time.monotonic()-start, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["headers", "ply", *MEMBERS])
    args = parser.parse_args()
    station, path = route_data()
    if args.source == "headers":
        print("route first/last in world", path[[0, -1]])
        with zipfile.ZipFile(LIDAR_ZIP) as z:
            for key, member in MEMBERS.items():
                with z.open(member) as stream:
                    h = laspy.LasHeader.read_from(stream)
                    print(key, h.version, h.point_format, h.point_count, h.scales, h.offsets, h.mins, h.maxs)
        return
    np.save(OUT / "route_xy.npy", np.column_stack([station, path]))
    crop(args.source, Corridor(path), path)


if __name__ == "__main__":
    main()
