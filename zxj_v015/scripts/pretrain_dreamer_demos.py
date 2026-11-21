"""Learn a Dreamer actor/world model from physical recordings with teacher actions.

This run contains no new physical interaction and is not navigation validation.
The saved standard Dreamer checkpoint is evaluated separately in real MuJoCo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from _bootstrap import ROOT
from atec_nav.config import load_config
from atec_nav.rl.demo_replay import seed_correction_windows, seed_recovery_windows, seed_replay
from atec_nav.rl.demo_split import split_teacher_episodes
from atec_nav.rl.recorded_reset import install_recorded_reset_metadata
from atec_nav.rl.offline_imitation import install_teacher_loss


SOURCE_FILES = ('scripts/pretrain_dreamer_demos.py', 'src/atec_nav/rl/offline_imitation.py',
                'src/atec_nav/rl/demo_replay.py', 'src/atec_nav/rl/demo_split.py',
                'src/atec_nav/rl/recorded_reset.py','src/atec_nav/rl/stagnation_observation.py')
# Preserve the implementation loaded by this process even if a later experiment
# edits the workspace while this training run is still active.
LOADED_SOURCES = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}


def replay_terrain_contract(terrain, allow_parent, ancestors=()):
    target_bytes = (terrain/'manifest.json').read_bytes()
    target_digest = hashlib.sha256(target_bytes).hexdigest()
    if not allow_parent:
        if ancestors:
            raise ValueError('Ancestor recordings require explicit source-surface replay transfer')
        return target_digest, None

    def asset_digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').digest()

    fields = ('dataset_root', 'sources', 'T_scene_native', 'route_length_m', 'lidar_roi_sha256')
    names = ('centerline.csv', 'terrain.npz', 'visual.obj', 'splats.npz')
    assets = {name: asset_digest(terrain/name) for name in names}
    target = child = json.loads(target_bytes)
    visited = {terrain.resolve()}
    verified = []
    for expected in (None, *ancestors):
        refinement = child.get('surface_collision_refinement', {})
        parent_path = child.get('parent_terrain')
        if not parent_path or refinement.get('parent') != parent_path:
            raise ValueError('Target terrain requires a documented source-surface refinement')
        parent = Path(parent_path).resolve()
        if parent in visited:
            raise ValueError('Terrain refinement lineage contains a cycle')
        visited.add(parent)
        parent_bytes = (parent/'manifest.json').read_bytes()
        digest = hashlib.sha256(parent_bytes).hexdigest()
        if digest != refinement.get('parent_manifest_sha256') or (expected is not None and digest != expected):
            raise ValueError('Parent terrain manifest differs from the recorded refinement')
        child = json.loads(parent_bytes)
        if any(target.get(field) != child.get(field) for field in fields):
            raise ValueError('Replay transfer requires the same measured route and coordinate frame')
        for name in names:
            if asset_digest(parent/name) != assets[name]:
                raise ValueError(f'Replay transfer changed the original course asset {name}')
        verified.append(digest)
    return target_digest, verified[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resolved-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--weight',type=float,default=1.)
    p.add_argument('--burnin',type=int,default=8)
    p.add_argument('--from-checkpoint',type=Path,help='Optional matching-schema Dreamer initialization')
    p.add_argument('--batch-length',type=int,help='Temporal context for new offline updates')
    p.add_argument('--recovery-weight',type=float,default=1.,help='Weight of teacher reverse/reapproach action labels')
    p.add_argument('--recovery-onset-weight',type=float,default=1.,help='Additional weight of the first recorded reverse command; does not change actions or observations')
    p.add_argument('--reset-independent-carry',action='store_true',help='Reset latent carry before each independent sampled sequence')
    p.add_argument('--allow-waypoint-migration',action='store_true',help='Explicitly initialize from the same tensors with a different planned waypoint reference')
    p.add_argument('--allow-stagnation-migration',action='store_true',help='Explicitly initialize the reserved lateral-command slot with a measured stagnation-age observation')
    p.add_argument('--allow-action-scale-migration',action='store_true',help='Explicitly initialize from earlier vx normalization; recorded commands and previous-command observations must already be migrated')
    p.add_argument('--representation-grad',action='store_true',help='Let action supervision train encoder/recurrent features as well as the actor')
    p.add_argument('--goal-reconstruction-weight',type=float,default=1.,help='Multiplier for original goal reconstruction loss')
    p.add_argument('--stagnation-reconstruction-weight',type=float,default=1.,help='Weight for the measured stagnation-age scalar; requires its observation contract')
    p.add_argument('--stagnation-augmentation-weight',type=float,default=0.,help='Auxiliary independently varied clock reconstruction only; no synthetic action or dynamics targets')
    p.add_argument('--waypoint-augmentation',type=float,default=0.,help='Counterfactual waypoint bearing range in radians; auxiliary labels only, never replaces recorded actions')
    p.add_argument('--waypoint-temporal-jitter',type=float,default=0.,help='Per-frame auxiliary bearing jitter to train responses to current goals')
    p.add_argument('--waypoint-cruise-speed',type=float,default=.315,help='Nominal m/s for auxiliary waypoint labels only; never a runtime speed override')
    p.add_argument('--waypoint-preserve-recorded-speed',action='store_true',help='Keep recorded slow/fast speed regimes in auxiliary turning targets')
    p.add_argument('--speed-demo-repeat',type=int,default=1,help='Training-only sampling repetitions for new speed teacher stages; validation stays unique')
    p.add_argument('--bend-demo-repeat',type=int,default=1,
                   help='Training-only sampling repetitions for verified 25 m and 50 m bend demonstrations')
    p.add_argument('--demo-stage-repeat', action='append', nargs=2, default=[], metavar=('STAGE', 'COUNT'),
                   help='Set training-only sampling repetitions for a recorded teacher stage')
    p.add_argument('--allow-parent-terrain-replay', action='store_true',
                   help='Keep explicitly attributed parent-scene recordings in replay after an obstacle surface refinement')
    p.add_argument('--recovery-phase-weight',type=float,default=0.,
                   help='Separate imitation losses for the recorded reverse and reapproach phases')
    p.add_argument('--recovery-window-stage', action='append', default=[],
                   help='Training stage supplying real reverse-onset sequences, repeat to cycle across stages')
    p.add_argument('--correction-window-stage', action='append', default=[],
                   help='Training stage supplying windows around recorded correction anchors')
    p.add_argument('--teacher-window-every', '--recovery-window-every', dest='teacher_window_every', type=int,
                   help='Use a batch of focused teacher windows at this update interval')
    a=p.parse_args();nav=load_config(a.config);out=a.output.resolve()
    if not 1<=a.speed_demo_repeat<=64:p.error('speed-demo-repeat must be in [1,64]')
    if not 1<=a.bend_demo_repeat<=64:p.error('bend-demo-repeat must be in [1,64]')
    sampling={name:a.speed_demo_repeat for name in ('flat_speed05','curve_speed05','full_speed05')} if a.speed_demo_repeat>1 else {}
    explicit_repeats = {}
    for stage, count in a.demo_stage_repeat:
        if stage in explicit_repeats:
            p.error(f'Repeated --demo-stage-repeat assignment for {stage}')
        try:
            repetitions = int(count)
        except ValueError:
            p.error('--demo-stage-repeat COUNT must be an integer')
        if not 1 <= repetitions <= 64:
            p.error('--demo-stage-repeat COUNT must be in [1,64]')
        explicit_repeats[stage] = repetitions
    if a.updates<=0 or nav['observations'].get('schema')!='physical_v2':p.error('Expected physical_v2 and positive updates')
    source=Path(nav['dreamer']['demo_path']);manifest=json.loads((source/'manifest.json').read_text())
    ancestors = manifest.get('replay_ancestor_terrain_manifest_sha256', [])
    if not isinstance(ancestors, list) or any(not isinstance(digest, str) for digest in ancestors):
        p.error('Replay ancestors must be ordered manifest hashes beyond the direct parent')
    terrain_digest, parent_digest = replay_terrain_contract(
        ROOT/nav['environment']['terrain_path'], a.allow_parent_terrain_replay, ancestors)
    if manifest.get('terrain_manifest_sha256') != terrain_digest or manifest.get('replay_parent_terrain_manifest_sha256') != parent_digest:
        p.error('Replay terrain manifest and declared parent must match the configured terrain contract')
    successful=[r for r in manifest['results'] if r['success']]
    include_teacher_action_mask = any('teacher_start_frame' in row for row in successful)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('JAX_PLATFORMS','cuda')
    sys.path.insert(0,str(ROOT/nav['dreamer']['upstream_path']))
    import elements,ruamel.yaml,jax
    from dreamerv3 import main as upstream
    from dreamerv3.agent import Agent
    install_recorded_reset_metadata(Agent, include_recovery_phase=bool(a.recovery_phase_weight),
                                    include_teacher_action_mask=include_teacher_action_mask)
    if a.waypoint_augmentation and (nav['actions']['vx_mps'] not in (.45,.5) or nav['actions']['wz_radps']!=.3):
        p.error('Counterfactual tracking labels require the declared M1 .45 or .5 m/s action contract')
    if (a.stagnation_reconstruction_weight>1. or a.stagnation_augmentation_weight) and not nav['observations'].get('stagnation_age'):
        p.error('Stagnation reconstruction requires the explicit measured-age observation contract')
    install_teacher_loss(Agent,a.weight,a.burnin,a.recovery_weight,a.representation_grad,a.goal_reconstruction_weight,
                         a.waypoint_augmentation,a.waypoint_temporal_jitter,a.recovery_onset_weight,a.stagnation_reconstruction_weight,
                         a.waypoint_cruise_speed,a.stagnation_augmentation_weight,
                         nav['observations'].get('stagnation_age',{}).get('scale_s',10.),
                         nav['observations'].get('stagnation_age',{}).get('clip_s',20.),
                         nav['actions']['vx_mps'],nav['actions']['wz_radps'],a.waypoint_preserve_recorded_speed,
                         recovery_phase_weight=a.recovery_phase_weight)
    config=elements.Config(ruamel.yaml.YAML(typ='safe').load(a.resolved_config.read_text()))
    config=elements.Config({**config,'logdir':str(out),'seed':31,
                            'jax':{**config.jax,'precompile':False,'profiler':False}})
    if a.batch_length is not None:
        if a.batch_length<=a.burnin+1:p.error('Batch length must leave aligned labels after burn-in')
        config=config.update({'batch_length':a.batch_length,'report_length':a.batch_length})
        nav['dreamer']['batch_length']=a.batch_length
    if a.reset_independent_carry and (config.consec_train!=1 or config.replay_context):
        p.error('Independent carry reset requires consec_train=1 and replay_context=0')
    focus_order = a.recovery_window_stage + a.correction_window_stage
    if bool(focus_order) != (a.teacher_window_every is not None):
        p.error('Teacher window stages and the update interval must be set together')
    if focus_order and (a.teacher_window_every < 2 or not a.reset_independent_carry or not include_teacher_action_mask):
        p.error('Teacher windows require an interval of at least two, independent carry, and recorded supervision masks')
    if len(set(focus_order)) != len(focus_order):
        p.error('Teacher window stages must be distinct')
    if a.bend_demo_repeat>1:
        bend_stages={'turn26_speed05','turn50_speed05'} & {row['stage'] for row in successful}
        sampling.update({stage:a.bend_demo_repeat for stage in sorted(bend_stages)})
    unknown = (set(explicit_repeats) | set(focus_order)) - {row['stage'] for row in successful}
    if unknown:
        p.error(f'Unrecorded teacher stage: {", ".join(sorted(unknown))}')
    sampling.update(explicit_repeats)
    out.mkdir(parents=True,exist_ok=False)
    total_transitions=0;unique_transitions=0
    for row in successful:
        with np.load(source/row['replay_file']) as episode:
            unique_transitions+=len(episode['reward'])
            total_transitions+=len(episode['reward'])*sampling.get(row['stage'],1)
    # Larger full-course datasets must not silently evict the short, earlier
    # demonstrations while replay is seeded. Replay storage is allocated lazily.
    capacity=max(config.replay.size,total_transitions)
    config=config.update({'replay.size':capacity})
    nav['dreamer']['replay_size']=capacity
    config.save(elements.Path(out/'config.yaml'))
    # Hold out complete teacher episodes; no validation frames enter training.
    split_rows,split_audit=split_teacher_episodes(manifest)
    terrain_contract = dict(terrain_manifest_sha256=terrain_digest, parent_terrain_manifest_sha256=parent_digest,
                            verified_terrain_ancestors=ancestors)
    datasets={};selection={}
    for split in ('train','validation'):
        folder=out/(split+'_data');folder.mkdir()
        rows=split_rows[split]
        subset={k:v for k,v in manifest.items() if k not in ('results','replay_sha256')}
        subset.update(results=rows,replay_sha256={})
        for row in rows:
            name=row['replay_file'];os.link(source/name,folder/name);subset['replay_sha256'][name]=manifest['replay_sha256'][name]
        (folder/'manifest.json').write_text(json.dumps(subset,indent=2)+'\n')
        replay=upstream.make_replay(config,split+'_replay',mode='train')
        selection[split]=seed_replay(replay,folder,nav['reward'],observations=nav['observations'],
            preserve_physical_reset=True,stage_repeats=sampling if split=='train' else None,
            recovery_control_hz=nav['environment']['control_hz'] if a.recovery_phase_weight else None,
            include_teacher_action_mask=include_teacher_action_mask,**terrain_contract)
        datasets[split]=replay
    focused = {}
    if focus_order:
        focused_config = config.update({'replay.chunksize':config.batch_length, 'replay.online':False})
        for index, stage in enumerate(focus_order):
            key = f'teacher_window_{index}'
            datasets[key] = upstream.make_replay(focused_config, key+'_replay', mode='train')
            seeder = seed_correction_windows if stage in a.correction_window_stage else seed_recovery_windows
            focused[stage] = seeder(datasets[key], out/'train_data', stage,
                config.batch_length, a.burnin, reward_config=nav['reward'], observations=nav['observations'],
                recovery_control_hz=nav['environment']['control_hz'] if a.recovery_phase_weight else None,
                **terrain_contract)
    class RecordedSpaces:
        obs_space={'depth':elements.Space(np.uint8,(64,64,3),0,255),
                   'proprio':elements.Space(np.float32,(64,)),'goal':elements.Space(np.float32,(3,)),
                   'reward':elements.Space(np.float32),'is_first':elements.Space(bool),
                   'is_last':elements.Space(bool),'is_terminal':elements.Space(bool)}
        act_space={'action':elements.Space(np.float32,(2,),-1,1),'reset':elements.Space(bool)}
        def close(self):pass
    # Upstream creates this solely to discover tensor spaces; data is real
    # recorded physics, with no contract environment used to generate rewards.
    upstream.make_env=lambda *args,**kwargs:RecordedSpaces()
    agent=upstream.make_agent(config)
    initialization=None
    if a.from_checkpoint:
        checkpoint=a.from_checkpoint.resolve()
        prior=json.loads((checkpoint.parent/'integration.json').read_text())
        old_actions=prior['config']['actions'];new_actions=nav['actions']
        if old_actions!=new_actions:
            if (not a.allow_action_scale_migration or old_actions['vx_mps']!=.45 or new_actions['vx_mps']!=.5 or
                {k:v for k,v in old_actions.items() if k!='vx_mps'}!={k:v for k,v in new_actions.items() if k!='vx_mps'}):
                raise ValueError('Action contract migration must be explicit and limited to .45 -> .5 m/s')
            if manifest.get('action_contract')!=new_actions or not manifest.get('recorded_action_scale_migrated'):
                raise ValueError('Action scale migration requires a verified, migrated demonstration manifest')
        old_obs=prior['config']['observations'];new_obs=nav['observations']
        migrated=old_obs!=new_obs
        allowed=set()
        if a.allow_waypoint_migration:allowed.add('waypoint_reference')
        if a.allow_stagnation_migration:allowed.add('stagnation_age')
        if migrated and (not allowed or
                {k:v for k,v in old_obs.items() if k not in allowed}!=
                {k:v for k,v in new_obs.items() if k not in allowed}):
            raise ValueError('Offline initialization observation contract differs')
        elements.checkpoint.load(checkpoint,{'agent':agent.load})
        initialization={'checkpoint':str(checkpoint),'sha256':hashlib.sha256((checkpoint/'agent.pkl').read_bytes()).hexdigest(),
                        'action_scale_migration':{'from':old_actions,'to':new_actions} if old_actions!=new_actions else None,
                        'updates':prior['counters']['updates'],
                        'waypoint_observation_migration':{'from':old_obs.get('waypoint_reference'),
                            'to':new_obs.get('waypoint_reference')} if old_obs.get('waypoint_reference')!=new_obs.get('waypoint_reference') else None,
                        'observation_migration':{key:{'from':old_obs.get(key),'to':new_obs.get(key)}
                            for key in allowed if old_obs.get(key)!=new_obs.get(key)}}
    streams={name:iter(agent.stream(upstream.make_stream(config,replay,'train')))
             for name,replay in datasets.items()}
    carry=agent.init_train(config.batch_size);report_carry=agent.init_report(config.batch_size)
    metrics_path=out/'offline_updates.jsonl';checks=[]
    previous_focus_stage = None
    focus_counts = dict.fromkeys(focused, 0)
    for update in range(a.updates):
        if a.reset_independent_carry:carry=agent.init_train(config.batch_size)
        focus_stage, stream = None, 'train'
        if focused and (update+1)%a.teacher_window_every==0:
            index = ((update+1)//a.teacher_window_every-1)%len(focused)
            focus_stage = focus_order[index]
            focus_counts[focus_stage] += 1
            stream = f'teacher_window_{index}'
        carry,_,metrics=agent.train(carry,next(streams[stream]))
        scalars={k:float(v) for k,v in metrics.items() if np.asarray(v).ndim==0}
        if scalars:
            if not all(np.isfinite(v) for v in scalars.values()):raise RuntimeError('Nonfinite offline training')
            if focused:
                # The upstream JAX wrapper returns the previous call's metrics.
                scalars['metrics_update'] = update
                scalars['teacher_window_stage'] = previous_focus_stage
            with metrics_path.open('a') as stream:stream.write(json.dumps({'update':update+1,**scalars})+'\n')
        previous_focus_stage = focus_stage
        if (update+1)%250==0 or update==a.updates-1:
            _,values=agent.report(report_carry,next(streams['validation']))
            check={'update':update+1,**{k:float(v) for k,v in values.items() if k.startswith('offline_bc/')}}
            checks.append(check);print(json.dumps(check),flush=True)
    saved=agent.save()
    if not all(np.isfinite(v).all() for v in saved['params'].values()):raise RuntimeError('Nonfinite parameters')
    elements.checkpoint.save(out/'final',{'agent':lambda:saved})
    snapshot=out/'source_snapshot';hashes={}
    for relative,content in LOADED_SOURCES.items():
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        hashes[relative]=hashlib.sha256(content).hexdigest()
    # Kept compatible with the regular, unmodified evaluation entry point.
    nav['dreamer']['logdir']=str(out);nav['seed']=31
    audit={'config':nav,'counters':saved['counters'],'all_parameters_finite':True,
           'jax_devices':[str(d) for d in jax.devices()],'final_checkpoint':str(out/'final'),
           'source_hashes':hashes,'teacher_only_offline_initialization':not include_teacher_action_mask,
           'teacher_only_action_imitation':True,'online_physics_interactions':0,
           'initialization':initialization,
           'offline_imitation':{'new_updates':a.updates,'weight':a.weight,'burnin':a.burnin,
               'recovery_weight':a.recovery_weight,'reset_independent_carry':a.reset_independent_carry,
               'recovery_onset_weight':a.recovery_onset_weight,
               'recovery_phase_weight':a.recovery_phase_weight,
               'representation_grad':a.representation_grad,'goal_reconstruction_weight':a.goal_reconstruction_weight,
               'stagnation_reconstruction_weight':a.stagnation_reconstruction_weight,
               'stagnation_augmentation_weight':a.stagnation_augmentation_weight,
               'stagnation_augmented_action_or_dynamics_targets':False,
               'waypoint_augmentation_rad':a.waypoint_augmentation,
               'waypoint_temporal_jitter_rad':a.waypoint_temporal_jitter,
               'waypoint_cruise_speed_mps':a.waypoint_cruise_speed,
               'waypoint_preserve_recorded_speed':a.waypoint_preserve_recorded_speed,
               'counterfactual_labels_are_physical_rollouts':False,
               'preserve_physical_episode_resets':True,
               'recorded_teacher_action_mask':include_teacher_action_mask,
               'replay_parent_terrain_manifest_sha256':parent_digest,
               'replay_capacity':capacity,'dataset_transitions':total_transitions,
               'unique_dataset_transitions':unique_transitions,'training_stage_repeats':sampling,
               'splits':selection,**split_audit,'validation':checks},
           'claim':'offline_teacher_initialization_requires_independent_physical_evaluation'}
    if ancestors:
        audit['offline_imitation']['replay_ancestor_terrain_manifest_sha256'] = ancestors
    if focused:
        audit['offline_imitation']['teacher_window_schedule'] = dict(
            sources=focused, stage_order=focus_order, every_updates=a.teacher_window_every,
            focused_batches=focus_counts, training_api_report_lag_updates=1)
    (out/'integration.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'offline_pretraining_complete','updates':saved['counters']['updates'],'not_navigation_success':True}),flush=True)


if __name__=='__main__':main()
