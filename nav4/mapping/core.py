"""Core terrain model on an indexed point set (used by build_trail_map and the unit tests)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import RES, signed_distance
from nav4.mapping.grid import band_atlas
from nav4.mapping.ground import connected_from_seeds, polyline_distance_mask, walked_cells
from nav4.mapping.model import classify_obstacles, column_features, ground_surface, max_step, plane_gradient


def run_model(cells, px, py, absnz, cls, origin, shape, coverage, z0, n0, walked_moving, walked_all_xy,
              lateral_yaw, h_dev, cfg, log=lambda msg: None):
    """cells: CellIndex; px, py, absnz, cls in cells' sorted order; walked_moving: (N,3+) xyz of
    moving poses (prior + trail zone); walked_all_xy: (M,2) all walked poses (corridor prior);
    lateral_yaw: per-cell route heading (for cross-track grade)."""
    g = cfg['ground']
    walked = walked_moving
    wmask = walked_cells(origin, shape, walked[:, 0:2], g['walked_cell_radius_m'])
    prior_z = np.full(shape, np.nan, np.float32)
    if g['prior_fill_under_path']:
        tree = cKDTree(walked[:, 0:2])
        ci = np.argwhere(wmask)
        _, k = tree.query(origin + ci * RES)
        prior_z[ci[:, 0], ci[:, 1]] = walked[k, 2] - h_dev
    gr = ground_surface(cells, cfg, z0, n0, wmask, prior_z)
    ground, known = gr['ground'], gr['known'] & coverage
    log('ground done')

    atlas, relative = band_atlas(cells, ground, known)
    outside = ~coverage
    atlas[outside, 1:19] = 0
    atlas[outside, 2:19:3] = -1
    atlas[..., 19] = known
    atlas[..., 0] = ground
    log('atlas done')

    gmax = ndi.maximum_filter(np.where(known, ground, -np.inf).astype(np.float32), size=3)
    gmax = np.where(np.isfinite(gmax), gmax, ground)
    rel_terrain = cells.z - gmax.reshape(-1)[cells.flat]
    feat = column_features(cells, relative, absnz, cls, cfg, shape, px, py, rel_terrain, origin)
    del rel_terrain
    obs = classify_obstacles(feat, known, cfg)
    log('obstacles done')
    del relative

    t = cfg['terrain']
    gx, gy, _ = plane_gradient(ground, known, t['grade_radius_m'])
    grade = np.degrees(np.arctan(np.hypot(gx, gy)))
    step = max_step(ground, known, t['step_radius_m'])
    cap_ok = np.nan_to_num(grade, nan=90.) <= t['max_grade_deg']
    cap_ok &= step <= t['max_step_m']
    nx, ny = -np.sin(lateral_yaw), np.cos(lateral_yaw)
    cross = np.degrees(np.arctan(np.abs(np.nan_to_num(gx) * nx + np.nan_to_num(gy) * ny)))
    along = np.degrees(np.arctan(np.abs(np.nan_to_num(gx) * np.cos(lateral_yaw) + np.nan_to_num(gy) * np.sin(lateral_yaw))))
    zone, dist_walk = polyline_distance_mask(origin, shape, walked[:, 0:2], t['trail_zone_halfwidth_m'])
    zone_conn = connected_from_seeds(ground, known & zone, wmask & known, t['trail_zone_connect_max_step_m'])
    exempt = zone_conn & (cross <= t['trail_zone_max_cross_grade_deg']) if t['trail_terrain_exemption'] else np.zeros(shape, bool)
    support_cap = known & coverage & cap_ok
    support = known & coverage & (cap_ok | exempt)
    for m in (support, support_cap):
        m[[0, 1, -2, -1], :] = False
        m[:, [0, 1, -2, -1]] = False
    log('support done')

    wc = cfg['walked_corridor']
    corridor, _ = polyline_distance_mask(origin, shape, walked_all_xy, wc['halfwidth_m'])
    hard, soft = obs['hard'], obs['soft']
    collide_b = hard                       # (b) no walked-corridor prior
    collide_c = hard & ~corridor           # (c) with prior
    collide_all = obs['obstacle']          # soft treated as hard, no prior
    collide_all_c = obs['obstacle'] & ~corridor
    use = collide_c if wc['apply_in_saved_map'] else collide_b
    flag = np.zeros(shape, np.uint8)
    flag[obs['obstacle']] = 2
    flag[use] = 1
    sdf = dict(support=signed_distance(support), support_cap=signed_distance(support_cap),
               obstacle_b=signed_distance(collide_b), obstacle_c=signed_distance(collide_c),
               obstacle_all=signed_distance(collide_all), obstacle_all_c=signed_distance(collide_all_c))
    log('sdf done')

    return dict(ground=ground, known=known, atlas=atlas, gr=gr, feat=feat, obs=obs, grade=grade, step=step,
                cap_ok=cap_ok, cross=cross, along=along, zone_conn=zone_conn, dist_walk=dist_walk, exempt=exempt,
                support=support, support_cap=support_cap, corridor=corridor, hard=hard, soft=soft, flag=flag,
                sdf=sdf, wmask=wmask, collide_c=collide_c, collide_b=collide_b)
