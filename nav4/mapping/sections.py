"""Cross-sections perpendicular to the route (task 2 old labels, task 5 new labels)."""
import numpy as np

from nav4.mapping.analysis import ROBOT_W, lookup

CLASS_COLORS = {1: '#4c72b0', 4: '#dd8452', 5: '#55a868', 6: '#c44e52', 7: '#8172b3', 12: '#937860',
                13: '#da8bc3', 14: '#8c8c8c', 23: '#ccb974', 24: '#64b5cd'}


class PointSlab:
    """Holds x,y,z,cls,nz arrays (RAM) with a coarse 1 m bucket index for fast slab queries."""

    def __init__(self, pts):
        self.x = np.asarray(pts['x'])
        self.y = np.asarray(pts['y'])
        self.z = np.asarray(pts['z'])
        self.cls = np.asarray(pts['cls'])
        self.nz = np.asarray(pts['nz']).astype(np.float32)
        self.x0, self.y0 = float(self.x.min()), float(self.y.min())
        bx = np.floor(self.x - self.x0).astype(np.int64)
        by = np.floor(self.y - self.y0).astype(np.int64)
        self.ny = int(by.max()) + 1
        key = bx * self.ny + by
        self.order = np.argsort(key, kind='stable')
        self.key = key[self.order]

    def query_box(self, xmin, xmax, ymin, ymax):
        bx0, bx1 = int(np.floor(xmin - self.x0)), int(np.floor(xmax - self.x0))
        by0, by1 = int(np.floor(ymin - self.y0)), int(np.floor(ymax - self.y0))
        idx = []
        for bx in range(bx0, bx1 + 1):
            a = np.searchsorted(self.key, bx * self.ny + by0)
            b = np.searchsorted(self.key, bx * self.ny + by1, side='right')
            idx.append(self.order[a:b])
        return np.concatenate(idx) if idx else np.zeros(0, np.int64)

    def slab(self, centre, yaw, half_lateral=4., half_along=.15):
        r = np.hypot(half_lateral, half_along) + .1
        idx = self.query_box(centre[0] - r, centre[0] + r, centre[1] - r, centre[1] + r)
        dx, dy = self.x[idx] - centre[0], self.y[idx] - centre[1]
        along = dx * np.cos(yaw) + dy * np.sin(yaw)
        lat = -dx * np.sin(yaw) + dy * np.cos(yaw)
        m = (np.abs(along) <= half_along) & (np.abs(lat) <= half_lateral)
        idx = idx[m]
        return idx, lat[m]


def cut_profile(grid2d, origin, centre, yaw, offsets, fill=np.nan):
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    pts = centre[None] + offsets[:, None] * normal[None]
    return lookup(grid2d, origin, pts, fill)


def draw_section(ax_hag, ax_cls, slab, idx, lat, ground_cut, offsets, title, labels=None, zlim=None,
                 walked_offsets=None, path_offset=None):
    """ax_hag: points coloured by height above the given ground; ax_cls: by classification.
    labels: dict name -> (bool array on offsets, color) drawn as strips under the ground."""
    z = slab.z[idx]
    g_at = np.interp(lat, offsets, np.nan_to_num(ground_cut, nan=np.nanmin(ground_cut) if np.isfinite(ground_cut).any() else 0))
    hag = z - g_at
    for ax in (ax_hag, ax_cls):
        ax.plot(offsets, ground_cut, 'k-', lw=1.2, label='map ground')
        ax.axvspan(-ROBOT_W / 2, ROBOT_W / 2, color='0.85', zorder=0)
        if walked_offsets is not None:
            for w in walked_offsets:
                ax.axvline(w, color='g', lw=.8, ls=':')
        if path_offset is not None:
            ax.axvline(path_offset, color='m', lw=.8, ls='--')
    sc = ax_hag.scatter(lat, z, c=np.clip(hag, -.3, 2.), s=1.5, cmap='turbo', vmin=-.3, vmax=2., linewidths=0)
    for c, col in CLASS_COLORS.items():
        m = slab.cls[idx] == c
        if m.any():
            ax_cls.scatter(lat[m], z[m], s=1.5, color=col, label=f'cls {c} ({m.mean():.0%})', linewidths=0)
    ax_cls.legend(fontsize=6, markerscale=4, loc='upper right', ncol=2)
    base = np.nanmin(ground_cut) if np.isfinite(ground_cut).any() else z.min()
    if labels:
        for k, (name, (mask, col)) in enumerate(labels.items()):
            y0 = base - .25 - .12 * k
            for ax in (ax_hag, ax_cls):
                ax.scatter(offsets[mask], np.full(mask.sum(), y0), s=4, marker='s', color=col, linewidths=0)
                ax.text(offsets[0], y0, name, fontsize=6, va='center', ha='left')
    ax_hag.set_title(title, fontsize=8)
    for ax in (ax_hag, ax_cls):
        ax.set_xlabel('lateral offset (m, +left)')
        ax.set_ylabel('z (m)')
        if zlim:
            ax.set_ylim(*zlim)
        else:
            ax.set_ylim(base - .25 - .12 * (len(labels or {}) + 1), base + 3.)
    return sc
