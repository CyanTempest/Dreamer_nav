"""Extract the video-matched forest route, keeping measured stairs and elevation.

The capture trajectory proposes XY positions; vertical rays recover ground.
No low-road altitude cutoff, artificial lane detour, or terrain flattening.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
import trimesh

from _bootstrap import ROOT
from build_atec_route import project_ground


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def refine_clearance(guide, reference):
    """Choose small lateral offsets inside measured treads, without changing paths."""
    manifest = json.loads((reference/'manifest.json').read_text())
    transform = np.asarray(manifest['T_scene_native'])
    points = guide@transform[:3,:3].T+transform[:3,3]
    with np.load(reference/'terrain.npz') as grid:
        observed = RegularGridInterpolator((grid['ys'],grid['xs']),grid['observed'].astype(float),
                                           method='nearest',bounds_error=False,fill_value=0)
        ground = RegularGridInterpolator((grid['ys'],grid['xs']),grid['heights'],
                                         bounds_error=False,fill_value=-100.)
    route = np.loadtxt(reference/'centerline.csv',delimiter=',',skiprows=1)
    distance,index = cKDTree(route[:,1:3]).query(points[:,:2])
    last = np.flatnonzero((distance<.10)&(index<len(route)-15))[-1]
    sample = np.unique(np.r_[np.arange(0,last,5),last])
    centers = points[sample,:2]
    tangent = np.gradient(centers,axis=0)
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    normals = np.column_stack([-tangent[:,1],tangent[:,0]])
    offsets = np.linspace(-.30,.30,31)
    footprint = np.linspace(-.28,.28,15)
    xy = centers[:,None,None,:]+normals[:,None,None,:]*(offsets[None,:,None,None]+footprint[None,None,:,None])
    observed_fraction = observed(xy[...,[1,0]]).mean(axis=2)
    heights = ground(xy[...,[1,0]])
    cross_slope = np.ptp(heights,axis=2)
    cost = 10000*(1-observed_fraction)+100*np.maximum(0,cross_slope-.18)**2+15*offsets[None,:]**2
    # Keep reset and final candidates attached to the visually established guide.
    cost[0,np.abs(offsets)>.001] += 10000
    cost[-1,np.abs(offsets)>.001] += 10000
    transition = 400*(offsets[:,None]-offsets[None,:])**2
    transition[np.abs(offsets[:,None]-offsets[None,:])>.061] = 1e6
    accumulated=cost[0].copy(); previous=[]
    for row in cost[1:]:
        candidate=accumulated[:,None]+transition
        back=np.argmin(candidate,axis=0);previous.append(back)
        accumulated=row+candidate[back,np.arange(len(offsets))]
    path=[int(np.argmin(accumulated))]
    for back in previous[::-1]:path.append(int(back[path[-1]]))
    chosen=offsets[np.array(path[::-1])]
    lateral=PchipInterpolator(sample,chosen)(np.arange(last+1))
    directions=np.column_stack([np.interp(np.arange(last+1),sample,normals[:,j]) for j in range(2)])
    points[:last+1,:2]+=lateral[:,None]*directions
    native=(points-transform[:3,3])@transform[:3,:3]
    return native, {'reference':str(reference.resolve()),'max_lateral_offset_m':float(np.max(np.abs(lateral))),
                    'method':'dynamic_programming_within_0.30m_of_video_matched_capture_guide',
                    'footprint_halfwidth_m':.28,'height_and_observation_costs':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/home/pc/Datasets/atec-opendataset'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-seconds', type=float, default=135.)
    parser.add_argument('--end-seconds', type=float, default=400.)
    parser.add_argument('--length', type=float, default=100.)
    parser.add_argument('--resolution', type=float, default=.04)
    parser.add_argument('--dense-lidar', type=Path, required=True,
                        help='Native XYZ+normal ROI extracted from plannar_map.ply')
    parser.add_argument('--clearance-reference', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps({'status':'building','navigation_complete':False}))
    traj_path = args.dataset/'03_Trajectory/developer_data/poses.csv'
    poses = np.loadtxt(traj_path)
    times = poses[:, 0]-poses[0, 0]
    use = (times >= args.start_seconds) & (times <= args.end_seconds)
    guide = savgol_filter(poses[use, 1:4], 11, 2, axis=0)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(guide[:, :2], axis=0), axis=1))]
    query_arc = np.arange(0, arc[-1], .04)
    guide = np.column_stack([np.interp(query_arc, arc, guide[:, j]) for j in range(3)])
    capture_time = np.interp(query_arc, arc, times[use])
    clearance = None
    if args.clearance_reference:
        guide, clearance = refine_clearance(guide, args.clearance_reference)
    tree = cKDTree(guide[:, :2])
    chunks, sources = [], []
    for path in sorted((args.dataset/'02_3DGS_map/render2/data/mesh').glob('*.ply')):
        mesh = trimesh.load(path, process=False)
        dist, idx = tree.query(mesh.triangles_center[:, :2])
        height = mesh.triangles_center[:, 2]-guide[idx, 2]
        keep = (dist < 6) & (height > -5) & (height < 8)
        if keep.any():
            chunks.append(mesh.submesh([np.flatnonzero(keep)], append=True))
            sources.append({'path': str(path), 'sha256': sha256(path)})
    mesh = trimesh.util.concatenate(chunks)
    ground = project_ground(mesh, guide[:, :2], top=guide[:, 2]-.6)
    if not np.isfinite(ground).all():
        raise ValueError('Unobserved centerline: cannot fabricate ground')
    if np.any(guide[:, 2]-ground > 2.5):
        raise ValueError('Ground more than 2.5 m below capture: review correspondence')
    native_route = np.column_stack([guide[:, :2], ground])
    ground_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(native_route, axis=0), axis=1))]
    if ground_arc[-1] < args.length:
        raise ValueError('Capture interval does not cover the requested length')
    # Keep all original measured samples, adding only the exact final cut point.
    n = np.searchsorted(ground_arc, args.length)
    endpoint = [np.interp(args.length, ground_arc, native_route[:, j]) for j in range(3)]
    native_route = np.vstack([native_route[:n], endpoint])
    capture_time = np.r_[capture_time[:n], np.interp(args.length, ground_arc, capture_time)]
    route_arc = np.r_[ground_arc[:n], args.length]
    native_start = native_route[0]
    delta = native_route[min(25, len(native_route)-1), :2]-native_start[:2]
    yaw = np.arctan2(delta[1], delta[0])
    rot = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    transform = np.eye(4); transform[:3, :3] = rot; transform[:3, 3] = -rot@native_start
    route = (native_route-native_start)@rot.T
    mesh.apply_transform(transform)
    mesh.export(out/'visual.obj')
    np.savetxt(out/'centerline.csv', np.column_stack([route_arc, route]), delimiter=',',
               header='s_m,x_m,y_m,z_m', comments='', fmt='%.9f')
    np.savetxt(out/'native_correspondence.csv', np.column_stack([route_arc, capture_time, native_route]),
               delimiter=',', header='s_m,capture_elapsed_s,native_x,native_y,ground_z', comments='', fmt='%.9f')
    tree = cKDTree(route[:, :2])
    resolution = args.resolution
    lo = np.floor((route[:, :2].min(axis=0)-2)/resolution)*resolution
    hi = np.ceil((route[:, :2].max(axis=0)+2)/resolution)*resolution
    xs, ys = [np.arange(a,b+resolution/2,resolution) for a,b in zip(lo,hi)]
    xx, yy = np.meshgrid(xs,ys)
    xy = np.column_stack([xx.ravel(),yy.ravel()])
    dist, idx = tree.query(xy)
    selected = dist <= 1.5
    height = np.full(len(xy), np.nan)
    print(f'Projecting {selected.sum()} ground cells at {resolution:g} m', flush=True)
    height[selected] = project_ground(mesh, xy[selected], top=route[idx[selected],2]+.35)
    observed = np.isfinite(height)
    drive = dist <= .32
    lidar = np.load(args.dense_lidar, mmap_mode='r')
    lidar_tree = cKDTree(lidar[:,:3])
    lidar_distance, _ = lidar_tree.query(native_route)
    del lidar_tree
    # Repair actual mesh holes from dense measured LiDAR, never across a riser.
    local_lidar = (lidar[:,:3]-native_start)@rot.T
    ld, li = tree.query(local_lidar[:,:2])
    support = (ld <= 1.7) & (np.abs(lidar[:,5]) > .65) & (np.abs(local_lidar[:,2]-route[li,2]) < .45)
    local_lidar = local_lidar[support]
    lidar_xy = cKDTree(local_lidar[:,:2])
    repairs = np.zeros(len(xy), bool)
    for i in np.flatnonzero(selected & ~observed):
        distance, near = lidar_xy.query(xy[i],k=32)
        p = local_lidar[near[distance < .10]]
        if len(p) < 5:
            continue
        # Select one measured height layer, avoiding mixed upper/lower step planes.
        p = p[np.abs(p[:,2]-p[0,2]) < .04]
        if len(p) < 5:
            continue
        design = np.column_stack([p[:,:2]-xy[i],np.ones(len(p))])
        coefficients,_,rank,_ = np.linalg.lstsq(design,p[:,2],rcond=None)
        if rank == 3 and np.max(np.abs(design@coefficients-p[:,2])) < .025 and np.linalg.norm(coefficients[:2]) < 1.5:
            height[i] = coefficients[2];observed[i] = True;repairs[i] = True
    # Missing cells stay unknown. Do not bridge steps or extrapolate banks.
    bottom = float(route[:,2].min()-3)
    height[~observed] = bottom
    np.savez_compressed(out/'terrain.npz', heights=height.reshape(xx.shape), observed=observed.reshape(xx.shape),
                        drive=drive.reshape(xx.shape), lidar_repair=repairs.reshape(xx.shape), xs=xs, ys=ys)
    np.savez_compressed(out/'surface_validation.npz', lidar_distance_m=lidar_distance,
                        unsupported_drive_xy=xy[drive&~observed])
    manifest = {
        'status':'measured_competition_geometry_physics_pending', 'scene_id':'atec_competition_forest_100m_'+out.name,
        'dataset_root':str(args.dataset), 'sources':sources, 'trajectory_sha256':sha256(traj_path),
        'source_video':str(ROOT/'路线视频.mp4'), 'video_opening_s':0,
        'route_selection':'video_matched_green_railing_entrance_then_continuous_forest_path',
        'alignment_review':str(ROOT/'outputs/route_alignment/2026-11-17/alignment_review.json'),
        'trajectory_candidate_seconds':[args.start_seconds,float(capture_time[-1])],
        'capture_path_is_ground_route':False, 'route_length_m':float(route_arc[-1]),
        'clearance_refinement':clearance,
        'native_start_xyz_m':native_start.tolist(), 'T_scene_native':transform.tolist(),
        'units':'meters', 'up_axis':'Z', 'ground_projection':'highest_source_mesh_below_capture_height_minus_0.6m',
        'collision':'measured_heightfield_no_flattening', 'resolution_m':resolution,
        'drive_corridor_halfwidth_m':.32, 'observed_drive_fraction':float(observed[drive].mean()),
        'unsupported_drive_cells':int((drive&~observed).sum()), 'unknown_cells_below_scene':True,
        'lidar_repaired_cells':int(repairs.sum()),
        'lidar_repair_rule':'5_points_within_0.10m_same_height_layer_0.04m_plane_residual_below_0.025m_slope_below_1.5',
        'visual_faces':len(mesh.faces), 'raw_ground_range_m':[float(route[:,2].min()),float(route[:,2].max())],
        'net_elevation_gain_m':float(route[-1,2]-route[0,2]),
        'lidar_surface_distance_p50_m':float(np.median(lidar_distance)),
        'lidar_surface_distance_p95_m':float(np.percentile(lidar_distance,95)),
        'lidar_roi':str(args.dense_lidar.resolve()), 'lidar_roi_sha256':sha256(args.dense_lidar),
        'physics_validated':False,'navigation_complete':False,
        'limitations':['video_pose_not_precisely_calibrated','capture_XY_proposal_not_optimized_for_robot_clearance',
                       'heightfield_resolves_stairs_at_grid_resolution','obstacle_collision_pending','locomotion_unverified_on_stairs']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    scatter=axs[0].scatter(xy[observed,0],xy[observed,1],c=height[observed],s=.6,cmap='terrain')
    axs[0].plot(route[:,0],route[:,1],'r-',lw=1);axs[0].axis('equal')
    axs[0].set(title='Video-matched forest route | measured surface',xlabel='Scene x (m)',ylabel='Scene y (m)')
    fig.colorbar(scatter,ax=axs[0],label='Ground z (m)')
    axs[1].plot(route_arc,route[:,2]);axs[1].set(xlabel='Ground route arc length (m)',ylabel='Ground z (m)',title='Stairs preserved: no terrain flattening');axs[1].grid(alpha=.3)
    fig.savefig(out/'route_overview.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources','T_scene_native']},indent=2),flush=True)


if __name__ == '__main__':
    main()
