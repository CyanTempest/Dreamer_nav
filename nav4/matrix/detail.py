"""Inspect source returns against v8 ground and obstacle labels."""
import sys

import numpy as np
from matplotlib import pyplot as plt

from nav4.mapping.common import bilinear, cell_index
from nav4.matrix.survey import Corridor, OUT, V8


def xyz(cloud):
    return np.column_stack([cloud[key] for key in ("x", "y", "z")])


def local_v8():
    path = np.load(OUT / "route_xy.npy")[:, 1:]
    with np.load(V8 / "map.npz") as f:
        s = f["s"]
        route = f["route"]
        origin = f["origin"]
        res = float(f["res"])
        lower = cell_index(path.min(0)-7, origin, res)
        upper = cell_index(path.max(0)+7, origin, res)
        region = tuple(slice(a, b) for a, b in zip(lower, upper))
        layers = f["layers"][region].copy()
    with np.load(V8 / "aux.npz") as f:
        known = f["known"][region].copy()
        observed = f["observed"][region].copy()
    return path, origin+lower*res, layers, known, observed, res, s, route


def main(name):
    path, origin, layers, _, _, res, _, _ = local_v8()
    cloud = np.load(OUT / f"{name}_crop.npy", mmap_mode="r")
    jump = max(1, len(cloud)//400000)
    cloud = cloud[::jump]
    p = xyz(cloud)
    s, cross, _ = Corridor(path).project(p[:, :2])
    height = bilinear(layers[..., 0], origin, p[:, :2], res=res)
    fig, ax = plt.subplots(5, figsize=(18, 15), layout="constrained")
    for a, width in zip(ax[:3], (.20, .55, 1.4)):
        m = np.abs(cross) <= width
        a.scatter(s[m], p[m, 2], s=.9, alpha=.45, c=p[m,2]-height[m], cmap="turbo", vmin=-.2, vmax=1)
        a.plot(np.arange(len(path))*.05, bilinear(layers[...,0],origin,path,res=res), c="k", lw=1)
        a.set(title=f"{name} |lateral| < {width} m", xlim=(0,30), ylim=(-1,18))
        a.set_xticks(np.arange(0,31))
        a.grid()
    m = (np.abs(cross) < 2) & (p[:,2]-height < 3) & (p[:,2]-height > -.45)
    colors = np.where(np.isin(cloud["classification"],[13,14]), "tab:green", "k")
    if name == "ply":
        colors = np.where(np.abs(cloud["normal_z"]) > .8, "#3679ad", colors)
    ax[3].scatter(s[m], cross[m], c=colors[m], s=.8, alpha=.35)
    ax[3].set(xlim=(0,30), ylim=(-2,2), title="XY projection (green: PLY 13/14; blue: nearly horizontal)")
    ax[3].set_xticks(np.arange(0,31)); ax[3].grid()
    i,j = np.nonzero(layers[...,6]==1)
    st, off, _ = Corridor(path).project(np.column_stack([i,j])*res+origin)
    ax[4].scatter(st, off, s=3, c="r")
    m = np.isin(cloud["classification"],[13,14])
    ax[4].scatter(s[m], cross[m], c="g", s=1)
    ax[4].set(xlim=(0,30), ylim=(-3,3), title="v8 hard cells and PLY 13/14")
    ax[4].set_xticks(np.arange(0,31)); ax[4].grid()
    fig.savefig(OUT / "figures" / f"diagnostic_{name}.jpg", dpi=110)


if __name__ == "__main__":
    main(sys.argv[1])
