"""Core terrain model on an indexed point set (used by build_trail_map and the unit tests).

Fixes on top of the v5 model (see trail_map_config.json for every threshold):
  A  obstacle cells without hard evidence and with sub-solid return density -> 'sparse'
     (visible, non-colliding), classify_obstacles(dist=...)
  B  trail-zone exemption: robot-scale cross-track grade (oriented footprint plane fit) and
     zone connectivity up to the measured maximum stair riser
  C  decks over voids under the walked path (deck_over_void), applied before the atlas and
     the obstacle features so the deck is the ground for both
The v5-rule layers (A off: hard_default, B off: support_v5_exemption) are returned for the report.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import RES, signed_distance
from nav4.mapping.grid import band_atlas
from nav4.mapping.ground import connected_from_seeds, polyline_distance_mask, walked_cells
from nav4.mapping.model import (classify_obstacles, column_features, deck_over_void, ground_surface, max_step,
                                oriented_cross_grade, plane_gradient)


def _edge_off(m):
    m[[0, 1, -2, -1], :] = False
    m[:, [0, 1, -2, -1]] = False
    return m


def run_model(cells, px, py, absnz, cls, origin, shape, coverage, z0, n0, walked_moving, walked_all_xy,
              lateral_yaw, h_dev, cfg, log=lambda msg: None):
    """cells: CellIndex; px, py, absnz, cls in cells' sorted order; walked_moving: (N,3+) xyz of
    moving poses (prior + trail zone); walked_all_xy: (M,2) all walked poses (corridor prior);
    lateral_yaw: per-cell route heading (for cross-track grade)."""
    g = cfg['ground']
    t = cfg['terrain']
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
    zone, dist_walk = polyline_distance_mask(origin, shape, walked[:, 0:2], t['trail_zone_halfwidth_m'])
    # fix C: decks over voids under the walked path (before atlas / obstacle features)
    dk = deck_over_void(cells, absnz, ground, known, wmask, prior_z, dist_walk, cfg)
    ground_v5 = ground.copy()
    deck = dk['deck'] & coverage
    ground = np.where(deck, dk['deck_z'], ground).astype(np.float32)
    known = known | deck
    gr['observed'] = gr['observed'] | deck
    log(f'ground done (deck cells {int(deck.sum())}, void walked cells {int(dk["void"].sum())}, '
        f'without deck {int(dk["void_without_deck"].sum())})')

    atlas, relative = band_atlas(cells, ground, known)
    n_ground_layer = cells.count(np.abs(relative) <= g['layer_thickness_m']).astype(np.int32)   # fix A diagnostics
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
    obs = classify_obstacles(feat, known, cfg, dist=dist_walk)
    log('obstacles done')
    del relative

    gx, gy, _ = plane_gradient(ground, known, t['grade_radius_m'])
    grade = np.degrees(np.arctan(np.hypot(gx, gy)))
    step = max_step(ground, known, t['step_radius_m'])
    cap_ok = np.nan_to_num(grade, nan=90.) <= t['max_grade_deg']
    cap_ok &= step <= t['max_step_m']
    nx, ny = -np.sin(lateral_yaw), np.cos(lateral_yaw)
    cross_disc = np.degrees(np.arctan(np.abs(np.nan_to_num(gx) * nx + np.nan_to_num(gy) * ny)))
    along = np.degrees(np.arctan(np.abs(np.nan_to_num(gx) * np.cos(lateral_yaw) + np.nan_to_num(gy) * np.sin(lateral_yaw))))
    # v5 exemption (B off): disc cross grade, 0.20 m connectivity
    zone_conn_v5 = connected_from_seeds(ground, known & zone, wmask & known, t['max_step_m'])
    exempt_v5 = zone_conn_v5 & (cross_disc <= t['trail_zone_max_cross_grade_deg'])
    # fix B: robot-scale cross grade inside the trail zone, connectivity up to the measured max riser
    cross = oriented_cross_grade(ground, known, zone & known, lateral_yaw, t['cross_grade_half_width_m'],
                                 t['cross_grade_half_length_m'])
    zone_conn = connected_from_seeds(ground, known & zone, wmask & known, t['trail_zone_connect_max_step_m'])
    exempt = zone_conn & (np.nan_to_num(cross, nan=90.) <= t['trail_zone_max_cross_grade_deg'])
    if not t['trail_terrain_exemption']:
        exempt = np.zeros(shape, bool)
        exempt_v5 = np.zeros(shape, bool)
    support_cap = _edge_off(known & coverage & cap_ok)
    support = _edge_off(known & coverage & (cap_ok | exempt))
    support_v5_exemption = _edge_off(known & coverage & (cap_ok | exempt_v5))
    log('support done')

    wc = cfg['walked_corridor']
    corridor, _ = polyline_distance_mask(origin, shape, walked_all_xy, wc['halfwidth_m'])
    hard, soft, sparse = obs['hard'], obs['soft'], obs['sparse']
    collide_b = hard                       # (b) no walked-corridor prior
    collide_c = hard & ~corridor           # (c) with prior
    collide_all = obs['obstacle']          # soft and sparse treated as hard, no prior
    collide_all_c = obs['obstacle'] & ~corridor
    collide_default_c = obs['hard_default'] & ~corridor     # fix A off (v5 rule: sparse is hard)
    use = collide_c if wc['apply_in_saved_map'] else collide_b
    flag = np.zeros(shape, np.uint8)
    flag[obs['obstacle']] = 2
    flag[use] = 1
    sdf = dict(support=signed_distance(support), support_cap=signed_distance(support_cap),
               support_v5_exemption=signed_distance(support_v5_exemption),
               obstacle_b=signed_distance(collide_b), obstacle_c=signed_distance(collide_c),
               obstacle_all=signed_distance(collide_all), obstacle_all_c=signed_distance(collide_all_c),
               obstacle_default_c=signed_distance(collide_default_c))
    log('sdf done')

    return dict(ground=ground, known=known, atlas=atlas, gr=gr, feat=feat, obs=obs, grade=grade, step=step,
                cap_ok=cap_ok, cross=cross, cross_disc=cross_disc, along=along, zone=zone, zone_conn=zone_conn,
                zone_conn_v5=zone_conn_v5, dist_walk=dist_walk, exempt=exempt, exempt_v5=exempt_v5,
                support=support, support_cap=support_cap, support_v5_exemption=support_v5_exemption,
                corridor=corridor, hard=hard, soft=soft, sparse=sparse, flag=flag, sdf=sdf, wmask=wmask,
                collide_c=collide_c, collide_b=collide_b, deck=deck, deck_info=dk, ground_v5=ground_v5,
                prior_z=prior_z, n_ground_layer=n_ground_layer)
