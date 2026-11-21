"""Audit primary frozen-policy evidence; leave final visual review explicit.

Unlike the early rolling-baseline verifier, this understands reset settling,
policy input timing, partial final physics steps on failure, and both declared
waypoint observation contracts. It does not change any original result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from evaluate_base_checkpoint import STAGES
from atec_nav.rl.spatial_waypoint_observation import SpatialWaypointObservation
from atec_nav.rl.waypoint_observation import waypoint_goal

TERRAIN_DIGEST = '7071987edb9b0a038c4765f2a510e502aa834e278925dabc000b545e789f0cc7'
TERRAIN_FILES = {
    'manifest.json': TERRAIN_DIGEST,
    'centerline.csv': 'df674e817b42264004e7dd89996b327ab296c56dd74726a227ff464797cbeeef',
    'terrain.npz': '32c82831b580838e3987ffa15906c3527f6b2f7b2e86d7bffb2ad34c623bb061',
    'obstacles.npz': 'e8d1e156c6119e9d705c99c7fab19f4921e6b10c5046743a6fd3df9aa4e0cb6e',
    'visual.obj': '3b4c180e4e8d440c9ea78cc65e76b31630e678901c786e8c5d5243bee6aedda2',
    'splats.npz': '27c5a2b7bb166e7d9615509f77ba39e2a052d73fd0d7214cb1d185091c5433c4',
}
ENTRANCE_REFINEMENT_FILES = {
    **TERRAIN_FILES,
    'manifest.json': 'c934d80c5f8d6741cce9189f995060890e4322f57a107ba7fdd815f6158ca3e5',
    'obstacles.npz': '37f147e3ea218837a7337761f33768ecfbcb3b262634951a89e2e31feb4e1678',
    'refinement_audit.npz': '9e4af9a33222f5537e194f69ffaeea529f2113c88bdcf0eaca3107b63559286f',
}
EXIT_BEND_REFINEMENT_FILES = {
    **ENTRANCE_REFINEMENT_FILES,
    'manifest.json': '5ec08c9b55aea5966b9bb7c1b54c684b7e12ccb3d8cbb1624d20c62627da449c',
    'obstacles.npz': '0add259e1f839255702ae8ead26b9b7aea152e827863df216394361a91f7a3d6',
    'refinement_audit.npz': '1cddf19ffaa82485ff96ea7fe907cc555612723721aaff67b411485391dbb48b',
}
COMPETITION_TERRAINS = {files['manifest.json']: files for files in
                        (TERRAIN_FILES, ENTRANCE_REFINEMENT_FILES, EXIT_BEND_REFINEMENT_FILES)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def audit_surface_refinement(terrain, manifest, parent_files):
    from scipy.spatial import cKDTree
    from atec_nav.sim.surface_voxels import child_cells, conservative_surface_mask

    revision = manifest['surface_collision_refinement']
    parent = Path(revision['parent'])
    require(revision['parent_manifest_sha256'] == parent_files['manifest.json'] == sha(parent/'manifest.json'),
            'Refinement parent is not the expected measured course')
    require(revision['parent_obstacles_sha256'] == parent_files['obstacles.npz'] == sha(parent/'obstacles.npz'),
            'Refinement parent obstacles changed')
    if 'refinement_audit.npz' in parent_files:
        require(revision['parent_refinement_audit_sha256'] == parent_files['refinement_audit.npz'] ==
                sha(parent/'refinement_audit.npz'), 'Parent refinement evidence changed')
    require(revision['visual_sha256'] == parent_files['visual.obj'] and
            revision['lidar_sha256'] == manifest['lidar_roi_sha256'], 'Refinement used different surface evidence')
    route = np.loadtxt(terrain/'centerline.csv', delimiter=',', skiprows=1)
    with np.load(parent/'obstacles.npz') as original, np.load(terrain/'obstacles.npz') as derived, \
            np.load(terrain/'refinement_audit.npz') as source:
        boxes = original['boxes']
        result = derived['boxes']
        selected = source['selected_parent_indices']
        lo, hi = revision['route_interval_m']
        stations = route[cKDTree(route[:, 1:3]).query(boxes[:, :2])[1], 0]
        np.testing.assert_array_equal(selected, np.flatnonzero((stations >= lo)&(stations <= hi)))
        unaffected = np.ones(len(boxes), dtype=bool)
        unaffected[selected] = False
        kept_count = int(unaffected.sum())
        np.testing.assert_array_equal(result[:kept_count], boxes[unaffected])
        resolution = revision['fine_resolution_m']
        children = child_cells(boxes[selected], resolution)
        np.testing.assert_array_equal(children, source['child_indices'])
        retained = conservative_surface_mask(np.minimum(source['distance_to_mesh_m'], source['distance_to_lidar_m']), resolution)
        np.testing.assert_array_equal(retained, source['retained'])
        np.testing.assert_array_equal(child_cells(result[kept_count:], resolution), children[retained])
    return dict(parent_manifest_sha256=parent_files['manifest.json'], interval_m=revision['route_interval_m'],
                unchanged_obstacle_boxes=kept_count, measured_surface_cells=int(retained.sum()))


def audit_assets(config):
    env=config['environment']
    require(env['physics_hz']==500 and env['control_hz']==5 and env['locomotion_controller']=='matrix_stair',
            'Physical/controller timing contract changed')
    speed=env.get('speed_transport')
    if speed is not None:
        from audit_speed_transport import validate_transport_settings
        validate_transport_settings(speed)
    require(config['actions']==dict(vx_mps=.5 if speed else .45,vy_mps=0.,wz_radps=.3,slew_rates=[.5,1.]),
            'Undeclared action contract changed')
    terrain=ROOT/env['terrain_path']
    terrain_digest = sha(terrain/'manifest.json')
    expected_terrain = COMPETITION_TERRAINS.get(terrain_digest)
    require(expected_terrain is not None, 'Wrong competition scene manifest')
    for name,expected in expected_terrain.items():require(sha(terrain/name)==expected,'Wrong competition scene artifact: '+name)
    manifest=read(terrain/'manifest.json')
    require(Path(manifest['dataset_root'])==Path('/home/pc/Datasets/atec-opendataset'),'Wrong source dataset')
    for source in manifest['sources']:require(sha(source['path'])==source['sha256'],'Original 3DGS source changed')
    require(sha(manifest['lidar_roi'])==manifest['lidar_roi_sha256'],'Original LiDAR region changed')
    entrance = exit_bend = None
    if expected_terrain is EXIT_BEND_REFINEMENT_FILES:
        exit_bend = audit_surface_refinement(terrain, manifest, ENTRANCE_REFINEMENT_FILES)
        parent = Path(manifest['surface_collision_refinement']['parent'])
        entrance = audit_surface_refinement(parent, read(parent/'manifest.json'), TERRAIN_FILES)
    elif expected_terrain is ENTRANCE_REFINEMENT_FILES:
        entrance = audit_surface_refinement(terrain, manifest, TERRAIN_FILES)
    robot=ROOT/env['robot_path']
    robot_files={'manifest.json':'fe72277fa2bcac5ec0fe317d00cc8a5f80737f836e7516c730c157b995b726ef',
                 'robot.xml':'cc64dd778b7bdd2f971a75ad548fad773645e9257df59482c9d16e3ac311e8a6'}
    for name,expected in robot_files.items():require(sha(robot/name)==expected,'Physical M1 asset changed')
    robot_manifest=read(robot/'manifest.json')
    require(sha(ROOT/robot_manifest['source_urdf'])==robot_manifest['source_sha256'],'Source M1 description changed')
    return dict(terrain_artifacts_sha256=expected_terrain, entrance_refinement=entrance, exit_bend_refinement=exit_bend,
                source_3dgs_files_verified=len(manifest['sources']),
                dataset_root=manifest['dataset_root'],source_lidar_sha256=manifest['lidar_roi_sha256'],
                robot_artifacts_sha256=robot_files,source_robot_sha256=robot_manifest['source_sha256'],
                simulation_only=True)


def audit_trial(run, config, checkpoint, start, length, seed, mode, noise):
    run=Path(run);manifest=read(run/'run_manifest.json');metrics=read(run/'metrics.json')
    require(manifest['status']=='evaluated' and manifest['policy']=='DreamerV3_checkpoint', 'Not a completed Dreamer trial')
    require(manifest['checkpoint_sha256']==checkpoint, 'Trial checkpoint mismatch')
    require(manifest['source_sha256']==sha(ROOT/'scripts/evaluate_dreamer.py'), 'Trial evaluator source mismatch')
    require(manifest['seed']==manifest['policy_seed']==seed and manifest['action_mode']==mode, 'Trial seed/mode mismatch')
    require(manifest['teacher_used'] is False and manifest['base_pose_teleported_after_reset'] is False,
            'Teacher or pose override in trial')
    require(manifest['checkpoint_reloaded'] and manifest['policy_actions_finite'], 'Invalid checkpoint/action evidence')
    require(manifest['depth_noise']==noise, 'Trial noise mismatch')
    for evidence in (manifest,metrics):
        require(evidence['evaluation_completed'] and evidence['eligible_for_navigation_gate'], 'Trial aborted')
        require(evidence['start_s']==start, 'Trial start mismatch')
    require(metrics['route_length_m']==metrics['target_length_m']==length, 'Trial route length mismatch')
    require(metrics['navigation_policy']=='agent_actions' and metrics['locomotion_controller']=='matrix_stair',
            'Wrong navigation or locomotion controller')
    require(metrics['localization']=='simulation_ground_truth','Unexpected localization contract')
    require(Path(manifest['terrain']).resolve()==(ROOT/config['environment']['terrain_path']).resolve(), 'Wrong terrain path')
    require(Path(manifest['robot']).resolve()==(ROOT/config['environment']['robot_path']).resolve(), 'Wrong robot path')
    with np.load(run/'physical_states.npz') as f:states={key:f[key] for key in f.files}
    with np.load(run/'policy_inputs.npz') as f:inputs={key:f[key] for key in f.files}
    count=len(states['time']);require(count==metrics['steps'] and count>0, 'Empty or miscounted trial')
    for array in (*states.values(),*inputs.values()):
        require(len(array)==count and np.isfinite(array).all(), 'Invalid saved state/input array')
    poses=states['qpos'];times=states['time'];progress=states['progress_m']
    require(poses.shape==(count,23) and states['qvel'].shape==(count,22), 'Wrong physical M1 state shape')
    np.testing.assert_allclose(np.linalg.norm(poses[:,3:7],axis=1),1,atol=1e-6,rtol=0)
    require(states['action'].shape==(count,2) and np.max(abs(states['action']))<=1, 'Invalid action trace')
    limits=np.array([config['actions']['vx_mps'],config['actions']['wz_radps']])
    require((abs(states['executed_command'])<=limits+1e-6).all(), 'Executed command exceeded configured limits')
    hz=config['environment']['physics_hz'];control=config['environment']['control_hz'];substeps=round(hz/control)
    require(manifest['fps']==control, 'Saved state rate mismatch')
    last_steps=metrics['physics_steps']-(count-1)*substeps
    require(0<last_steps<=substeps, 'Invalid final physics step count')
    if count>1:
        np.testing.assert_allclose(np.diff(times)[:-1],1/control,atol=1e-7,rtol=0)
        np.testing.assert_allclose(times[-1]-times[-2],last_steps/hz,atol=1e-7,rtol=0)
    reset_clock=times[0]-(1/control if count>1 else last_steps/hz)
    np.testing.assert_allclose(times[-1]-reset_clock,metrics['sim_time_s'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(metrics['physics_steps']/hz,metrics['sim_time_s'],atol=1e-7,rtol=0)
    require(bool(inputs['is_first'][0]) and not inputs['is_first'][1:].any() and not inputs['is_last'].any(),
            'Unexpected reset/terminal input inside a physical episode')
    np.testing.assert_array_equal(inputs['reward'][1:],states['reward'][:-1])
    with np.load(run/'observations.npz') as f:
        np.testing.assert_array_equal(f['depth'][:-1],inputs['depth'][1:])
    route=np.loadtxt(ROOT/config['environment']['terrain_path']/'centerline.csv',delimiter=',',skiprows=1)
    spatial=config['environment'].get('factory')=='atec_nav.sim.spatial_mujoco_env:make_env'
    if spatial:
        require(metrics['route_projection']=='metric_heading_v1', 'Spatial route projection mismatch')
        require(metrics['spatial_waypoint_reference']==config['observations']['spatial_waypoint_reference'], 'Spatial contract mismatch')
        require(metrics['spatial_observation_source_sha256']==sha(ROOT/'src/atec_nav/rl/spatial_waypoint_observation.py'),
                'Spatial observation source mismatch')
        observe=SpatialWaypointObservation(route,config)
    else:
        require(metrics.get('route_projection','legacy_ordered')=='legacy_ordered', 'Legacy projection mismatch')
        observe=lambda xy,yaw,s,goal:waypoint_goal(route,xy,yaw,s,goal,config)
    rebuilt=[]
    for pose,s in zip(poses[:-1],progress[:-1]):
        w,x,y,z=pose[3:7];yaw=np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z))
        rebuilt.append(observe(pose[:2],yaw,s,start+length))
    if rebuilt:np.testing.assert_allclose(rebuilt,inputs['goal'][1:],atol=2e-6,rtol=1e-6)
    endpoint=np.array([np.interp(start+length,route[:,0],route[:,j]) for j in (1,2)])
    distance=float(np.linalg.norm(poses[-1,:2]-endpoint))
    np.testing.assert_allclose(distance,metrics['goal_distance_m'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(progress[-1]-start,metrics['final_progress_m'],atol=1e-8,rtol=0)
    require(metrics['success']==(metrics['reason']=='success'), 'Inconsistent success/reason')
    if metrics['success']:
        require(progress[-1]>=start+length-.15 and distance<.25 and not metrics['collision_contacts'], 'Invalid arrival')
        require(last_steps==substeps, 'Success ended within an incomplete control interval')
    steps=np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1)
    transport=None
    if config['environment'].get('speed_transport'):
        from audit_speed_transport import audit_policy_transport
        transport=audit_policy_transport(run,config)
    return dict(run=str(run.resolve()),seed=seed,mode=mode,success=metrics['success'],reason=metrics['reason'],
                calibrated_transport=transport,
                progress_m=metrics['final_progress_m'],endpoint_distance_m=distance,physics_steps=metrics['physics_steps'],
                saved_states=count,verified_goal_inputs=len(rebuilt),initial_reset_input_not_reconstructed=True,
                maximum_goal_error=float(np.max(abs(np.asarray(rebuilt)-inputs['goal'][1:]))) if rebuilt else None,
                reset_settle_clock_s=float(reset_clock),sim_time_s=metrics['sim_time_s'],
                last_control_interval_physics_steps=last_steps,
                maximum_saved_position_step_m=float(steps.max()) if len(steps) else 0.,
                saved_body_3d_path_length_m=float(steps.sum()),
                evidence_sha256={name:sha(run/name) for name in ['metrics.json','run_manifest.json',
                                                               'physical_states.npz','policy_inputs.npz','observations.npz']})


def audit_video(run, video, output):
    import cv2
    run,video,output=map(Path,(run,video,output))
    require(not output.exists(),'Choose a new video audit output');output.mkdir(parents=True)
    recording=read(video.with_suffix('.recording.json'));verification=read(video.with_suffix('.verified.json'))
    manifest=read(run/'run_manifest.json');metrics=read(run/'metrics.json')
    require(recording['appearance']=='native_3DGS' and recording['is_full_run'], 'Incomplete or non-native recording')
    require(recording['source']=='saved_actual_physics_states' and verification['source']=='saved_actual_physical_evaluation_states',
            'Video source is not the actual physical trace')
    require(recording['renderer_sha256']==sha(ROOT/'scripts/record_gaussian_navigation.py'), 'Renderer source mismatch')
    require(verification['sha256']==sha(video) and Path(verification['source_run']).resolve()==run.resolve(), 'Video identity mismatch')
    require(verification['navigation_success'] and metrics['success'] and verification['teacher_used'] is False,
            'Video is not a successful independent policy trial')
    require(recording['navigation_success'] and recording['navigation_result']==verification['navigation_result']=='success'
            and verification['route_length_m']==metrics['route_length_m']==100., 'Video result/route mismatch')
    require(recording['video']==video.name and Path(verification['video']).resolve()==video.resolve(), 'Video path mismatch')
    with np.load(run/'physical_states.npz') as states:
        times=states['time'];progress=states['progress_m']
    count=len(times)
    frames=np.loadtxt(video.with_suffix('.frames.csv'),delimiter=',',skiprows=1,ndmin=2)
    require(frames.shape==(count,4), 'Video/physical frame count mismatch')
    np.testing.assert_array_equal(frames[:,0],np.arange(count));np.testing.assert_array_equal(frames[:,1],np.arange(count))
    np.testing.assert_allclose(frames[:,2],times,atol=1e-9,rtol=0)
    np.testing.assert_allclose(frames[:,3],progress-manifest['start_s'],atol=1e-9,rtol=0)
    capture=cv2.VideoCapture(str(video));require(capture.isOpened(),'Video could not be opened')
    fps=capture.get(cv2.CAP_PROP_FPS);resolution=[int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    chosen={0,count//2,count-1,*[int(np.argmin(abs(progress-s))) for s in (8.9,15.5,45.2,70.)]}
    decoded=0;previews=[]
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            require(list(frame.shape[:2][::-1])==resolution,'Decoded frame resolution changed')
            if decoded in chosen:
                path=output/f'frame_{decoded:06d}.jpg';require(cv2.imwrite(str(path),frame),'Preview write failed')
                previews.append(dict(frame=decoded,progress_m=float(progress[decoded]),path=str(path.resolve())))
            decoded+=1
    finally:capture.release()
    require(decoded==count==recording['frame_count']==verification['decoded_frames'] and count>0,'Incomplete video decode')
    require(resolution==recording['resolution']==verification['resolution']==[1280,720],'Video resolution mismatch')
    require(abs(fps-manifest['fps'])<1e-6 and abs(fps-recording['fps'])<1e-6 and abs(fps-verification['fps'])<1e-6,'Video timing mismatch')
    return dict(video=str(video.resolve()),sha256=verification['sha256'],decoded_frames=decoded,
                fps=fps,resolution=resolution,all_saved_states_rendered_once=True,
                previews=previews,visual_review_complete=False)


def audit_recording_overlay(run, video, decoded, physical_states_sha256):
    import cv2
    from audit_navigation_video_hud import audit as audit_hud

    cv2.setNumThreads(1)
    overlay = audit_hud(Path(run), Path(video))
    require(overlay['video_sha256'] == decoded['sha256'], 'Video changed between decode and overlay verification')
    require(overlay['physical_states_sha256'] == physical_states_sha256, 'Overlay used different physical states')
    require(overlay['decoded_overlay_frames'] == decoded['decoded_frames'], 'Overlay frame count mismatch')
    return overlay


def require_complete_suite(job):
    require(job['status']=='all_stages_passed' and job.get('learned_100m_success') and job.get('video'),
            'Full suite and verified video are not complete')
    require(len(job['stages'])==len(STAGES) and {r['stage'] for r in job['stages']}==set(STAGES), 'All ten stages required')
    require(job['evaluation_seeds']==list(range(6001,6011)), 'Original formal seed cohort required')
    for row in job['stages']:
        start,length,required=STAGES[row['stage']]
        require(row['passed'] and row['evaluation_complete'] and row['start_s']==start and row['length_m']==length
                and row['required_successes']==required, 'Invalid stage gate')
        for mode in ('mean','sample'):
            trials=[r for r in row['evaluations'] if r['mode']==mode]
            require(sorted(r['seed'] for r in trials)==job['evaluation_seeds'] and len(trials)==10, 'Missing formal trials')
            require(sum(r['success'] for r in trials)>=required, 'Insufficient stage successes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();job=read(args.job);require_complete_suite(job)
    output=args.output.resolve();require(not output.exists(),'Choose a new final audit output')
    training=Path(job['training_run']);metadata=read(training/'integration.json');config=metadata['config']
    checkpoint=sha(training/'final/agent.pkl');require(checkpoint==job['checkpoint_sha256'],'Frozen checkpoint changed')
    require(config['observations']['schema']=='physical_v2','Expected physical_v2 observations')
    assets=audit_assets(config)
    spatial=config['environment'].get('factory')=='atec_nav.sim.spatial_mujoco_env:make_env'
    if spatial:
        from evaluate_spatial_checkpoint import evaluation_identity
        from evaluate_spatial_checkpoint_suite import validate_batch
        require(job['identity']==evaluation_identity(training),'Frozen spatial inputs changed')
    rows=[]
    for stage in job['stages']:
        name=stage['stage'];start,length,required=STAGES[name]
        noise={**config.get('noise',{})}
        if name=='noisy100':noise['enabled']=True
        for mode in ('mean','sample'):
            if spatial:validate_batch(args.job.parent/name/mode,job['identity'],job['evaluation_seeds'],mode,name,config,True)
            for trial in [r for r in stage['evaluations'] if r['mode']==mode]:
                require(read(Path(trial['output'])/'run_manifest.json')['trained_updates']==metadata['counters']['updates'],
                        'Trial model update counter mismatch')
                result=audit_trial(trial['output'],config,checkpoint,start,length,trial['seed'],mode,noise)
                require(result['success']==trial['success'] and result['reason']==trial['reason'], 'Job/primary result mismatch')
                require(abs(result['progress_m']-trial['progress_m'])<1e-8,'Job/primary progress mismatch')
                result['stage']=name;rows.append(result)
        print(json.dumps(dict(audited_stage=name,total_trials=len(rows))),flush=True)
    video=job['video'];source=Path(video['source_run']).resolve()
    source_trial = next((r for r in rows if r['stage']=='noisy100' and r['mode']=='mean'
                         and r['success'] and Path(r['run'])==source), None)
    require(source_trial is not None, 'Final recording must use a successful noisy100 mean trial')
    output.mkdir(parents=True)
    decoded=audit_video(source,video['video'],output/'video_frames')
    decoded['overlay_audit'] = audit_recording_overlay(
        source, video['video'], decoded, source_trial['evidence_sha256']['physical_states.npz'])
    require(sha(training/'final/agent.pkl')==checkpoint,'Checkpoint changed during audit')
    report=dict(status='technical_evidence_verified_visual_review_pending',formal_acceptance_complete=False,
                source_job=str(args.job.resolve()),source_job_sha256=sha(args.job),checkpoint_sha256=checkpoint,
                original_scene_manifest_sha256=TERRAIN_DIGEST,
                evaluation_terrain_manifest_sha256=assets['terrain_artifacts_sha256']['manifest.json'],
                assets=assets,stages=len(STAGES),trials=len(rows),
                stage_thresholds_unchanged=True,initial_reset_inputs_not_reconstructed=True,
                saved_physics_state_rate_hz=config['environment']['control_hz'],
                physics_contact_checks_rate_hz=config['environment']['physics_hz'],
                physics_contact_evidence='Evaluator and environment execute strict checks per physical substep; saved poses are control-rate samples.',
                results=rows,video=decoded)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':main()
