"""Load explicitly labelled teacher transitions with Driver-compatible alignment."""
import json
import hashlib
from pathlib import Path
import numpy as np

def recorded_reapproach(actions, control_hz):
    """Label high forward requests within the teacher's 7.2 s recovery window."""
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('A positive teacher control frequency is required')
    reverse = actions[:, 0] < -.05
    starts = np.flatnonzero(reverse & ~np.r_[False, reverse[:-1]])
    in_recovery = np.zeros(len(actions), dtype=bool)
    duration = round(7.2 * control_hz)
    for start in starts:
        in_recovery[start + 1:start + duration] = True
    return in_recovery & ~reverse


def seed_recovery_windows(replay, directory, stage, length, burnin, **contract):
    """Seed genuine subsequences with a supervised reverse onset after warm-up."""
    return _seed_teacher_windows(replay, directory, stage, length, burnin, False, contract)


def seed_correction_windows(replay, directory, stage, length, burnin, **contract):
    """Seed genuine subsequences at the verified correction frames in the manifest."""
    return _seed_teacher_windows(replay, directory, stage, length, burnin, True, contract)


def _seed_teacher_windows(replay, directory, stage, length, burnin, corrections, contract):
    if not 0 <= burnin < length-1:
        raise ValueError('Teacher windows must fit burn-in and an aligned action')
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    sources = [row for row in manifest['results'] if row.get('success') and row.get('stage') == stage]
    windows = []

    class Windows:
        def __init__(self):
            self.episode = []

        def add(self, row, worker):
            self.episode.append(row)
            if not row['is_last']:
                return
            episode, self.episode = self.episode, []
            source = sources[-1000-worker]
            supervised = np.array([step['teacher_action_supervised'] for step in episode])
            if corrections:
                anchors = source.get('teacher_focus_frames')
                if 'teacher_focus_frames' not in source and source.get('eligible_for_training') is True:
                    intervention = source.get('teacher_intervention')
                    if isinstance(intervention, dict) and intervention.get('frame') == source.get('teacher_start_frame'):
                        anchors = [intervention.get('frame')]
                if (not isinstance(anchors, list) or not anchors
                        or any(type(frame) is not int or not 0 <= frame < len(episode)-1 or not supervised[frame]
                               for frame in anchors)
                        or len(set(anchors)) != len(anchors)):
                    raise ValueError('Correction anchors must be distinct supervised nonterminal frames')
            else:
                reverse = np.array([step['action'][0] < -.05 for step in episode])
                anchors = np.flatnonzero(reverse & ~np.r_[False, reverse[:-1]] & supervised).tolist()
            if len(episode) < length:
                return
            for anchor in anchors:
                lower = max(0, anchor-(length-2))
                upper = min(len(episode)-length, anchor-burnin)
                if lower > upper:
                    if lower != 0 or not episode[0]['teacher_is_first']:
                        continue
                    upper = 0
                center = burnin+(length-2-burnin)//2
                start = int(np.clip(anchor-center, lower, upper))
                for transition in episode[start:start+length]:
                    replay.add(transition, worker=-1000-len(windows))
                coordinate = 'anchor_frame' if corrections else 'onset_frame'
                windows.append(dict(source_worker=worker, recording=source['replay_file'], start_frame=start,
                                    **{coordinate: anchor}))

    sink = Windows()
    source = seed_replay(sink, directory, stages=[stage], preserve_physical_reset=True,
                         include_teacher_action_mask=True, **contract)
    if not windows:
        raise ValueError(f'No supervised anchor fits a replay window for {stage}')
    return dict(stage=stage, window_length=length, burnin=burnin, windows=windows,
                source_episodes=source['unique_source_episodes'], actual_physical_resets_preserved=True)


def seed_replay(replay, directory, reward_config=None, stages=None, observations=None, terrain_manifest_sha256=None,
                preserve_physical_reset=False,stage_repeats=None,recovery_control_hz=None,
                include_teacher_action_mask=False, parent_terrain_manifest_sha256=None,
                verified_terrain_ancestors=()):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if observations and observations.get('schema', 'legacy_v1') != 'legacy_v1':
        if manifest.get('observations') != observations:
            raise ValueError('Teacher observation schema differs from training')
        if not terrain_manifest_sha256 or manifest.get('terrain_manifest_sha256') != terrain_manifest_sha256:
            raise ValueError('Teacher physical terrain differs from training')
    if stages is not None and (not stages or isinstance(stages, str)):
        raise ValueError('Teacher stages must be a nonempty list of stage names')
    allowed_stages = None if stages is None else set(stages)
    selected = [row for row in manifest['results'] if row.get('success')
                and (allowed_stages is None or row.get('stage') in allowed_stages)]
    declared_parent = manifest.get('replay_parent_terrain_manifest_sha256')
    if declared_parent != parent_terrain_manifest_sha256:
        raise ValueError('Parent terrain replay requires an explicitly verified source')
    declared_ancestors = manifest.get('replay_ancestor_terrain_manifest_sha256', [])
    if (not isinstance(declared_ancestors, list) or tuple(declared_ancestors) != tuple(verified_terrain_ancestors)
            or (declared_ancestors and not parent_terrain_manifest_sha256)):
        raise ValueError('Ancestor terrain replay requires explicitly verified refinement lineage')
    physical_field = 'source_terrain_manifest_sha256'
    if parent_terrain_manifest_sha256:
        if not terrain_manifest_sha256 or not observations or observations.get('schema') != 'physical_v2':
            raise ValueError('Parent terrain replay requires the target and physical observation contracts')
        allowed_sources = {terrain_manifest_sha256, parent_terrain_manifest_sha256, *verified_terrain_ancestors}
        if any(row.get(physical_field) not in allowed_sources for row in selected):
            raise ValueError('Physical recording is missing verified terrain provenance')
    elif terrain_manifest_sha256 and any(row.get(physical_field, terrain_manifest_sha256) != terrain_manifest_sha256
                                         for row in selected):
        raise ValueError('Teacher physical terrain differs from training')
    if reward_config and reward_config.get('goal_bounded_progress'):
        if manifest.get('reward_config') != reward_config:
            raise ValueError('Teacher rewards do not match the goal-bounded training configuration')
    stage_repeats=stage_repeats or {}
    if any(not isinstance(v,int) or isinstance(v,bool) or not 1<=v<=64 for v in stage_repeats.values()):
        raise ValueError('Teacher sampling repetitions must be integers in [1,64]')
    count=0;episodes=0;unique_episodes=0;unique_transitions=0
    unsupervised_prefix_transitions = 0
    for result in selected:
        path=directory/result['replay_file']
        expected=manifest.get('replay_sha256',{}).get(result['replay_file'])
        if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Demo hash mismatch: {path}')
        with np.load(path) as archive:
            # NpzFile decompresses a whole member on every __getitem__. Keep
            # one episode in memory so a 100 m depth stream is decoded once,
            # rather than once for every frame submitted to replay.
            data={key:archive[key] for key in archive.files}
            length=len(data['reward'])
            if not bool(data['is_first'][0]) or not bool(data['is_last'][-1]):
                raise ValueError('Demo must contain one complete episode')
            if any(len(data[k])!=length for k in data):raise ValueError('Unaligned demo arrays')
            if not np.isfinite(data['action']).all() or np.abs(data['action']).max()>1:raise ValueError('Invalid demo action')
            teacher_start = result.get('teacher_start_frame', 0)
            if 'teacher_start_frame' in result:
                if type(teacher_start) is not int or not 0 <= teacher_start < length - 1:
                    raise ValueError('Teacher supervision must begin on a recorded nonterminal frame')
                if teacher_start and not include_teacher_action_mask:
                    raise ValueError('Policy approach actions need a teacher supervision mask')
            if include_teacher_action_mask:
                data['teacher_action_supervised'] = np.arange(length) >= teacher_start
            if recovery_control_hz is not None:
                data['teacher_reapproach'] = recorded_reapproach(data['action'], recovery_control_hz)
            if observations and observations.get('schema') == 'physical_v2':
                expected = {'depth':(observations['height'], observations['width'], 3), 'proprio':(64,), 'goal':(3,)}
                for key, shape in expected.items():
                    if key not in data or data[key].shape[1:] != shape or not np.isfinite(data[key]).all():
                        raise ValueError('Teacher observation array mismatch: ' + key)
            for _ in range(stage_repeats.get(result.get('stage'),1)):
                for i in range(length):
                    transition={k:data[k][i] for k in data}
                    if preserve_physical_reset:
                        transition['teacher_is_first']=np.asarray(data['is_first'][i],dtype=bool).copy()
                    replay.add(transition,worker=-1000-episodes)
                count+=length;episodes+=1
                unsupervised_prefix_transitions += teacher_start
            unique_episodes+=1;unique_transitions+=length
    if not episodes:raise ValueError('No successful teacher episodes')
    return {'teacher_episodes':episodes,'teacher_transitions':count,'teacher_directory':str(directory),
            'teacher_stages':sorted(allowed_stages) if allowed_stages is not None else None,
            'unique_source_episodes':unique_episodes,'unique_source_transitions':unique_transitions,
            'stage_sampling_repetitions':stage_repeats,'repetitions_are_new_physical_trials':False,
            'policy_prefix_transitions_without_imitation':unsupervised_prefix_transitions}
