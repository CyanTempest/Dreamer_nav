"""Stage spatially tiled LAS and annotated PLY samples for the reconstruction."""
import argparse
from pathlib import Path
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import PLANNING_PLY, open_ply, write_json
from nav4.matrix.survey import MEMBERS, V8

FULL = Path("data/trail_a_mujoco/full_v1")
WORK = FULL / "work"
TILE = 12.
STRIDE = 7


def route():
    with np.load(V8 / "map.npz") as data:
        s = np.arange(14441) * .05
        xy = np.column_stack([np.interp(s, data["s"], col) for col in data["route"][:, :2].T])
    return s, xy


def files(folder, low, high):
    minimum = np.floor(np.asarray(low) / TILE).astype(int)
    maximum = np.floor(np.asarray(high) / TILE).astype(int)
    return [folder / f"{i}_{j}.bin" for i in range(minimum[0], maximum[0] + 1)
            for j in range(minimum[1], maximum[1] + 1) if (folder / f"{i}_{j}.bin").exists()]


def read_tiles(file_paths, width):
    if not file_paths:
        return np.empty((0, width), np.float32)
    return np.concatenate([np.fromfile(p, np.float32).reshape(-1, width) for p in file_paths])


def scan(name):
    start = time.monotonic()
    target = WORK / f"{name}.partial"
    target.mkdir(parents=True)
    _, xy = route()
    pitch = .25
    base = np.floor((xy.min(0) - 8) / pitch) * pitch
    shape = np.ceil((xy.max(0) + 8 - base) / pitch).astype(int) + 1
    grid = np.indices(shape, dtype=np.float32).reshape(2, -1).T * pitch + base
    distance = cKDTree(xy[::4]).query(grid, workers=4)[0].reshape(shape)
    bounds = np.floor(base / TILE).astype(int)
    accepted = 0

    def partition(x, y):
        i = np.rint((x - base[0]) / pitch).astype(np.int32)
        j = np.rint((y - base[1]) / pitch).astype(np.int32)
        take = np.flatnonzero((i >= 0) & (j >= 0) & (i < shape[0]) & (j < shape[1]))
        return take[distance[i[take], j[take]] <= 7.3]

    def append(rows):
        nonlocal accepted
        accepted += len(rows)
        if not len(rows):
            return
        tiles = np.floor(rows[:, :2] / TILE).astype(np.int32) - bounds
        keys = 1024 * tiles[:, 0] + tiles[:, 1]
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
        for a, b in zip(starts[:-1], starts[1:]):
            i, j = np.array(divmod(int(keys[a]), 1024)) + bounds
            with (target / f"{i}_{j}.bin").open("ab") as stream:
                rows[order[a:b]].tofile(stream)

    if name == "ply":
        p = open_ply(PLANNING_PLY)
        count = len(p)
        for start_index in range(0, count, 800000):
            block = p[start_index:start_index + 800000]
            a = block[partition(block["x"], block["y"])]
            append(np.column_stack([a[field] for field in (
                "x", "y", "z", "normal_x", "normal_y", "normal_z", "classification")]).astype(np.float32))
    else:
        source = PLANNING_PLY.parents[2] / MEMBERS["removed"]
        with laspy.open(source) as las:
            h = las.header
            count = h.point_count
            offset = 0
            for k, block in enumerate(las.chunk_iterator(1_000_000)):
                first = (-offset) % STRIDE
                x = block.X[first::STRIDE] * h.scales[0] + h.offsets[0]
                y = block.Y[first::STRIDE] * h.scales[1] + h.offsets[1]
                take = partition(x, y)
                local = first + take * STRIDE
                z = block.Z[local] * h.scales[2] + h.offsets[2]
                holdout = ((offset + local) % 11 < 2)
                append(np.column_stack([x[take], y[take], z, holdout]).astype(np.float32))
                offset += len(block)
                if k % 200 == 199:
                    print(name, offset, accepted, flush=True)
    result = dict(total=count, staged=accepted, stride=STRIDE if name == "las" else 1,
                  spatial_tile_m=TILE, frame="native shared LAS/PLY XYZ", seconds=time.monotonic() - start)
    write_json(target / "source.json", result)
    target.rename(WORK / name)
    print(name, result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["ply", "las"])
    scan(parser.parse_args().source)
