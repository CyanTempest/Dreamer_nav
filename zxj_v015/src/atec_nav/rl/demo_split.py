"""Keep complete physical recordings and their goal prefixes in one split."""


def split_teacher_episodes(manifest):
    rows=[row for row in manifest['results'] if row.get('success')]
    if not rows:raise ValueError('No successful teacher episodes to split')
    stages={row['stage'] for row in rows}
    heldout={stage:max(row['seed'] for row in rows if row['stage']==stage) for stage in stages}
    def recording(row):
        # Goal-relabelled prefixes carry the hash of the original full replay.
        # Their own NPZ hash differs, despite sharing physical observations.
        digest=(row.get('physical_recording_sha256') or row.get('source_sha256') or
                manifest.get('replay_sha256',{}).get(row['replay_file']))
        if not digest:raise ValueError('Missing physical recording provenance')
        return digest
    validation_sources={recording(row) for row in rows if row['seed']==heldout[row['stage']]}
    # A growing cohort must not silently move an earlier held-out recording
    # into training just because a new seed becomes the largest in its stage.
    retained=set(manifest.get('heldout_physical_recordings',[]))
    if retained-{recording(row) for row in rows}:
        raise ValueError('Retained held-out recording is missing from this dataset')
    validation_sources.update(retained)
    splits={name:[row for row in rows if (recording(row) in validation_sources)==(name=='validation')]
            for name in ('train','validation')}
    for name,subset in splits.items():
        missing=stages-{row['stage'] for row in subset}
        if missing:raise ValueError(f'{name} lacks independent recordings for stages: {sorted(missing)}')
    return splits,{'heldout_seed_by_stage':heldout,'heldout_physical_recordings':sorted(validation_sources),
                   'retained_heldout_physical_recordings':sorted(retained)}
