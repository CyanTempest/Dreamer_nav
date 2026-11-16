"""Segment-A terrain model: ground, support (locomotion capability), hard/soft obstacles.

Pure functions on per-cell arrays and the (cell, z)-sorted point index; the
pipeline that calls them is build_trail_map.py.  Parameters: trail_map_config.json.
"""
import numpy as np
from scipy import ndimage as ndi

from nav4.mapping.common import RES
from nav4.mapping.ground import connected_from_seeds, connected_two_layer, lowest_layer, nan_median


# ---------------------------------------------------------------- ground
def cone_lower_envelope(z, valid, radius, slope):
    """L(c) = min_n z(n) + slope*|c-n| over valid n within radius (grey erosion with a cone)."""
    r = int(np.ceil(radius / RES))
    big = np.where(valid, z, np.inf).astype(np.float32)
    out = big.copy()
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            d = np.hypot(a, b) * RES
            if (a == 0 and b == 0) or d > radius + 1e-9:
                continue
            shifted = np.full_like(big, np.inf)
            sa = slice(max(0, a), big.shape[0] + min(0, a))
            ta = slice(max(0, -a), big.shape[0] + min(0, -a))
            sb = slice(max(0, b), big.shape[1] + min(0, b))
            tb = slice(max(0, -b), big.shape[1] + min(0, -b))
            shifted[ta, tb] = big[sa, sb] + slope * d
            np.minimum(out, shifted, out=out)
    return out


def lowest_ground_layer(cells, cfg):
    g = cfg['ground']
    return lowest_layer(cells, np.ones(len(cells.z), bool), g['layer_thickness_m'], g['layer_min_points'])


def ground_surface(cells, cfg, z0, n0, walked_mask, prior_z):
    """Returns dict with ground (filled everywhere), known, observed, flags."""
    g = cfg['ground']
    has = np.isfinite(z0)
    # concentration of returns at the lowest layer vs. the slab above it (bare ground ~1, vegetation << 1)
    zl = z0.reshape(-1)[cells.flat]
    rel = cells.z - zl
    in_layer = np.isfinite(rel) & (rel >= -1e-4) & (rel <= g['layer_thickness_m'])
    n_layer = cells.count(in_layer)
    n_slab = cells.count(np.isfinite(rel) & (rel >= -1e-4) & (rel <= g['vegetation_slab_m']))
    # ground value = mean of the lowest-layer returns (the minimum is biased low by
    # 0.5*cell*tan(slope), e.g. -2.9 cm on a 30 deg slope)
    with np.errstate(invalid='ignore', divide='ignore'):
        zmean = (z0 + (cells.sum(np.nan_to_num(rel), in_layer) / np.maximum(n_layer, 1))).astype(np.float32)
    z0 = np.where(has, zmean, np.nan).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        conc = np.where(n_slab > 0, n_layer / np.maximum(n_slab, 1), np.nan).astype(np.float32)
    # 1) pit / spike test (old proxy rule, 5x5 median, 0.25 m)
    med = nan_median(z0, g['median_window'])
    spike = has & (np.abs(z0 - med) >= g['median_tolerance_m'])
    obs = has & ~spike
    # 2) vegetation / object tops: well above a slope-limited lower envelope AND diffuse returns
    env = cone_lower_envelope(z0, obs, g['envelope_radius_m'], np.tan(np.deg2rad(g['envelope_slope_deg'])))
    above = z0 - env
    veg_top = obs & (above >= g['vegetation_top_min_m']) & ~(conc >= g['bare_concentration_min'])
    obs &= ~veg_top
    # 3) connectivity from the walked path.  Seeds: walked cells whose layer is within
    #    prior_tolerance of (scanner z - device height).  A second, higher layer (>= gap above
    #    the lowest one, bare/concentrated returns) is used instead of the lowest layer only
    #    where the lowest is NOT connected to the walked surface but the higher one is and the
    #    lowest lies >= sub_surface_min_drop below it: sub-surface returns (reflections on
    #    wet/polished floors, under-deck points) beneath a walked surface.
    zraw = zl
    cand2 = np.isfinite(rel) & (rel > g['second_layer_gap_m'])
    z1raw, _ = lowest_layer(cells, cand2, g['layer_thickness_m'], g['layer_min_points'])
    zl1 = z1raw.reshape(-1)[cells.flat]
    rel1 = cells.z - zl1
    in1 = np.isfinite(rel1) & (rel1 >= -1e-4) & (rel1 <= g['layer_thickness_m'])
    n1 = cells.count(in1)
    n1slab = cells.count(np.isfinite(rel1) & (rel1 >= -1e-4) & (rel1 <= g['vegetation_slab_m']))
    with np.errstate(invalid='ignore', divide='ignore'):
        z1 = np.where(np.isfinite(z1raw), z1raw + cells.sum(np.nan_to_num(rel1), in1) / np.maximum(n1, 1), np.nan).astype(np.float32)
        conc1 = n1 / np.maximum(n1slab, 1)
    v1 = np.isfinite(z1) & (conc1 >= g['bare_concentration_min'])
    tol = g['prior_tolerance_m']
    seeds0 = walked_mask & obs & (np.abs(z0 - prior_z) <= tol)
    seeds1 = walked_mask & v1 & (np.abs(z1 - prior_z) <= tol)
    conn0, conn1 = connected_two_layer(z0, obs, z1, v1, seeds0, seeds1, g['connect_max_dz_m'])
    sub_surface = ~conn0 & conn1 & has & (z1 - z0 >= g['sub_surface_min_drop_m'])
    z0 = np.where(sub_surface, z1, z0).astype(np.float32)
    obs = (obs & ~sub_surface) | sub_surface
    comp_ok = conn0 | sub_surface
    lab, nlab = ndi.label(obs & ~comp_ok, structure=np.ones((3, 3)))
    if nlab:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        big = np.zeros(nlab + 1, bool)
        big[1:] = sizes * RES ** 2 >= g['min_component_area_m2']
        comp_ok |= big[lab]
    small_island = obs & ~comp_ok
    obs &= comp_ok
    connected = conn0 | sub_surface
    # 4) gap fill (old proxy: <= 0.2 m)
    dist, ind = ndi.distance_transform_edt(~obs, return_indices=True)
    ground = z0[tuple(ind)] if obs.any() else np.zeros_like(z0)
    known = dist * RES <= g['interpolate_radius_m']
    # 5) walked-path prior where the path cell is still unknown
    prior_fill = walked_mask & ~known & np.isfinite(prior_z)
    ground = np.where(prior_fill, prior_z, ground).astype(np.float32)
    known |= prior_fill
    return dict(ground=ground, known=known, observed=obs, z0=z0, n0=n0, conc=conc, spike=spike,
                veg_top=veg_top, small_island=small_island, prior_fill=prior_fill, above_env=above.astype(np.float32),
                sub_surface=sub_surface, connected=connected)


# ---------------------------------------------------------------- terrain derivatives
def plane_gradient(ground, known, radius):
    """Least-squares plane over a disc of `radius` (known cells only): (gx, gy, n)."""
    r = int(np.ceil(radius / RES))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx ** 2 + yy ** 2) * RES ** 2 <= radius ** 2 + 1e-9).astype(np.float64)
    w = known.astype(np.float64)
    z = np.where(known, ground, 0.).astype(np.float64)
    # kernel offsets: axis 0 of the map is world x, axis 1 is world y
    kx, ky = (yy * RES) * disc, (xx * RES) * disc
    corr = lambda img, k: ndi.correlate(img, k, mode='constant', cval=0.)
    n = corr(w, disc)
    sx = corr(w, kx); sy = corr(w, ky)
    sxx = corr(w, kx * kx); syy = corr(w, ky * ky); sxy = corr(w, kx * ky)
    sz = corr(z, disc); sxz = corr(z, kx); syz = corr(z, ky)
    # centre the normal equations
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my, mz = sx / n, sy / n, sz / n
        cxx = sxx - n * mx * mx; cyy = syy - n * my * my; cxy = sxy - n * mx * my
        cxz = sxz - n * mx * mz; cyz = syz - n * my * mz
        det = cxx * cyy - cxy * cxy
        gx = (cyy * cxz - cxy * cyz) / det
        gy = (cxx * cyz - cxy * cxz) / det
    bad = (n < 0.5 * disc.sum()) | ~np.isfinite(gx) | ~np.isfinite(gy)
    gx = np.where(bad, np.nan, gx)
    gy = np.where(bad, np.nan, gy)
    return gx.astype(np.float32), gy.astype(np.float32), n.astype(np.float32)


def max_step(ground, known, radius=None):
    """Local discontinuity height per cell: max of |g(n)-g(c)| over the 8 neighbours and of the
    axis-aligned central differences |g(c+d)-g(c-d)|, |d| = 0.1 m (0.2 m baseline), over known
    cells.  Only cells on/straddling a discontinuity are flagged (a riser smeared over two cells
    is still measured at full height); a 35 deg slope gives 0.07 / 0.14 m."""
    H, W = ground.shape
    out = np.zeros(ground.shape, np.float32)

    def shifted(a, b):
        sa = slice(max(0, a), H + min(0, a)); ta = slice(max(0, -a), H + min(0, -a))
        sb = slice(max(0, b), W + min(0, b)); tb = slice(max(0, -b), W + min(0, -b))
        return sa, sb, ta, tb

    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            sa, sb, ta, tb = shifted(a, b)
            d = np.abs(ground[sa, sb] - ground[ta, tb])
            ok = known[sa, sb] & known[ta, tb]
            np.maximum(out[ta, tb], np.where(ok, d, 0), out=out[ta, tb])
    for a, b in ((1, 0), (0, 1)):
        # central difference at c: g(c+d) - g(c-d)
        gp = np.full(ground.shape, np.nan, np.float32); gm = np.full(ground.shape, np.nan, np.float32)
        sa, sb, ta, tb = shifted(a, b)
        gp[ta, tb] = np.where(known[sa, sb], ground[sa, sb], np.nan)
        sa, sb, ta, tb = shifted(-a, -b)
        gm[ta, tb] = np.where(known[sa, sb], ground[sa, sb], np.nan)
        d = np.abs(gp - gm)
        np.maximum(out, np.where(np.isfinite(d), d, 0), out=out)
    return out


# ---------------------------------------------------------------- obstacle features
def _voxel_pca(cflat, x, y, z, size, min_points):
    """Covariance eigen-features per voxel (cflat = voxel id)."""
    bc = lambda w: np.bincount(cflat, weights=w, minlength=size)
    n = np.bincount(cflat, minlength=size).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my, mz = bc(x) / n, bc(y) / n, bc(z) / n
        cov = np.zeros((size, 3, 3))
        cov[:, 0, 0] = bc(x * x) / n - mx * mx
        cov[:, 1, 1] = bc(y * y) / n - my * my
        cov[:, 2, 2] = bc(z * z) / n - mz * mz
        cov[:, 0, 1] = cov[:, 1, 0] = bc(x * y) / n - mx * my
        cov[:, 0, 2] = cov[:, 2, 0] = bc(x * z) / n - mx * mz
        cov[:, 1, 2] = cov[:, 2, 1] = bc(y * z) / n - my * mz
    good = n >= min_points
    lam = np.full((size, 3), np.nan)
    vec = np.full((size, 3, 3), np.nan)
    if good.any():
        w, v = np.linalg.eigh(cov[good])
        lam[good] = np.maximum(w[:, ::-1], 0)
        vec[good] = v[:, :, ::-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        lin = (lam[:, 0] - lam[:, 1]) / lam[:, 0]
        pla = (lam[:, 1] - lam[:, 2]) / lam[:, 0]
        sca = lam[:, 2] / lam[:, 0]
    return n, lin, pla, sca, np.abs(vec[:, 2, 0]), np.abs(vec[:, 2, 2])


def column_features(cells, relative, absnz, cls, cfg, shape, px, py, rel_terrain=None, origin=(0., 0.)):
    """Per fine cell counts and per 0.3 m voxel (2 voxels over the body band) shape features.
    px, py: point x, y in the cells' sorted order (float).
    rel_terrain: height above the HIGHEST known ground in the 3x3 neighbourhood; the lower
    body-band bound uses it so that riser / ledge faces (terrain, judged by the support
    step test) are not counted as obstacles.  The upper bound uses the cell's own ground."""
    o = cfg['obstacles']
    lo, hi = o['body_band_m']
    rt = relative if rel_terrain is None else rel_terrain
    body = (rt > lo) & (relative < hi)
    n_body = cells.count(body)
    n_vert = cells.count(body & (absnz < o['vertical_normal_max_abs_nz']))
    n_above = cells.count((relative >= hi) & (relative < o['continuation_top_m']))
    nbins = int(round((hi - lo) / o['subbin_m']))
    occ = np.zeros(shape, np.int8)
    for k in range(nbins):
        m = (relative > lo + k * o['subbin_m']) & (relative <= lo + (k + 1) * o['subbin_m'])
        occ += (cells.count(m) > 0).astype(np.int8)
    rail = cells.count(body & np.isin(cls, o['linear_structure_classes']))
    f = int(o['pca_block_cells'])
    flat = cells.flat[body]
    i, j = flat // shape[1], flat % shape[1]
    ci, cj = i // f, j // f
    cshape = ((shape[0] + f - 1) // f, (shape[1] + f - 1) // f)
    mid = (lo + hi) / 2
    rel = relative[body].astype(np.float64)
    vk = (rel >= mid).astype(np.int64)
    vflat = (ci * cshape[1] + cj) * 2 + vk
    size = cshape[0] * cshape[1] * 2
    # coordinates relative to the voxel's own corner (covariance is translation invariant)
    x = px[body].astype(np.float64) - (origin[0] + (ci * f - .5) * RES)
    y = py[body].astype(np.float64) - (origin[1] + (cj * f - .5) * RES)
    n, lin, pla, sca, e1z, e3z = _voxel_pca(vflat, x, y, rel, size, o['pca_min_points'])
    v = lambda a: a.reshape(cshape[0], cshape[1], 2)
    up = lambda a: np.repeat(np.repeat(a, f, 0), f, 1)[:shape[0], :shape[1]].astype(np.float32)
    feats = dict(n_body=n_body, n_vert=n_vert, n_above=n_above, occ_bins=occ, n_bins=nbins, n_rail=rail)
    for k, name in ((0, 'lo'), (1, 'hi')):
        feats[f'vox_{name}_n'] = up(v(n)[..., k])
        feats[f'vox_{name}_lin'] = up(v(lin)[..., k])
        feats[f'vox_{name}_pla'] = up(v(pla)[..., k])
        feats[f'vox_{name}_sca'] = up(v(sca)[..., k])
        feats[f'vox_{name}_e1z'] = up(v(e1z)[..., k])
        feats[f'vox_{name}_e3z'] = up(v(e3z)[..., k])
    return feats


def classify_obstacles(feat, known, cfg):
    """obstacle = >= body_min_points in body band on known ground; HARD unless positive soft evidence.

    Hard evidence (any voxel of the 0.3 m column, or the 0.1 m column):
      vertical line (post/stem/trunk edge), vertical plane (wall/rock face/trunk),
      any well-defined surface (planar, not scattered: rock top, bench, ledge),
      continuous upright column (all 6 sub-bins + continuation + vertical normals),
      linear-structure classification codes (posts/rails).
    Soft = obstacle cell with no hard evidence, every populated voxel volumetric
    (scattering >= soft_min_scattering with enough points), nothing directly above
    the robot height (0.8-1.2 m), and not adjacent to a hard-evidence cell.
    """
    o = cfg['obstacles']
    obstacle = (feat['n_body'] >= o['body_min_points']) & known
    with np.errstate(invalid='ignore', divide='ignore'):
        frac_vert = feat['n_vert'] / np.maximum(feat['n_body'], 1)
    vline = np.zeros(known.shape, bool)
    vplane = np.zeros(known.shape, bool)
    surface = np.zeros(known.shape, bool)
    scattered_all = np.ones(known.shape, bool)
    evidence = np.zeros(known.shape, bool)
    for k in ('lo', 'hi'):
        n = feat[f'vox_{k}_n']
        enough = n >= o['pca_min_points']
        lin, pla, sca = feat[f'vox_{k}_lin'], feat[f'vox_{k}_pla'], feat[f'vox_{k}_sca']
        vline |= enough & (lin >= o['linear_min']) & (feat[f'vox_{k}_e1z'] >= o['vertical_axis_min_abs_z'])
        vplane |= enough & (pla >= o['planar_min']) & (feat[f'vox_{k}_e3z'] <= o['vertical_plane_max_normal_z'])
        surface |= enough & (pla >= o['planar_min']) & (sca < o['soft_min_scattering'])
        # voxels with too few returns give no evidence either way; at least one voxel must
        # carry enough returns and every such voxel must be volumetric (scattered)
        scattered_all &= ~enough | (sca >= o['soft_min_scattering'])
        evidence |= enough
    diffuse = evidence & scattered_all
    continuous = (feat['occ_bins'] >= feat['n_bins']) & (feat['n_above'] >= o['continuation_min_points']) & \
        (frac_vert >= o['continuous_column_min_vertical_fraction'])
    railing = feat['n_rail'] >= o['body_min_points']
    hard_evidence = vline | vplane | surface | continuous | railing
    soft = obstacle & ~hard_evidence & diffuse
    if o['soft_requires_no_body_top_continuation']:
        soft &= feat['n_above'] < o['continuation_min_points']
    soft &= ~ndi.binary_dilation(obstacle & hard_evidence, structure=np.ones((3, 3)))
    hard = obstacle & ~soft
    return dict(obstacle=obstacle, hard=hard, soft=soft, frac_vert=frac_vert.astype(np.float32),
                vertical_line=vline, vertical_plane=vplane, surface=surface, continuous=continuous,
                railing=railing, diffuse=diffuse, hard_evidence=hard_evidence)
