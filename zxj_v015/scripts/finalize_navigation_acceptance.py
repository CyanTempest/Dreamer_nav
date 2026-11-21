"""Bind full physical-suite verification to the manual review of its exact video."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from audit_frozen_navigation_suite import ROOT, STAGES, read, require, require_complete_suite, sha
from evaluate_spatial_checkpoint import evaluation_identity


def validate_visual_review(report_video, review_path):
    import cv2

    video = Path(report_video['video']).resolve()
    review = read(review_path)
    require(review['status']=='route_video_visual_comparison_complete', 'Manual route comparison is missing')
    require(Path(review['rendered_video']).resolve()==video and review['rendered_video_sha256']==report_video['sha256']==sha(video),
            'Manual review describes a different video')
    require(Path(review['source_original_video']).resolve()==ROOT/'路线视频.mp4' and
            sha(review['source_original_video'])==review['source_original_video_sha256'], 'Original footage changed')
    earlier_audit = Path(review['technical_review'])
    require(sha(earlier_audit)==review['technical_review_sha256'], 'Technical review used for visual comparison changed')
    earlier_video = read(earlier_audit)['video']
    for field in ('sha256','resolution','fps','decoded_frames'):
        require(earlier_video[field]==report_video[field], f'Conflicting video evidence: {field}')
    references = {float(item['requested_time_s']):item for item in review['original_references']}
    require(references and len(references)==len(review['original_references']), 'Original-video landmarks were not sampled')
    for item in references.values():
        require(math.isfinite(item['decoded_time_s']) and item['decoded_time_s']>=0 and
                sha(item['frame'])==item['sha256'], 'Original-video reference image changed')

    previews = {item['frame']:item for item in report_video['previews']}
    frames = {item['frame']:item for item in review['reviewed_frames']}
    require(len(frames)==len(review['reviewed_frames']) and frames.keys()==previews.keys(),
            'Manual review does not cover the audited preview selection')
    require(0 in frames and report_video['decoded_frames']-1 in frames, 'The start and finish must be reviewed')
    require(any(item['original_reference_approx_seconds'] for item in frames.values()),
            'The manual review must identify corresponding original-video landmarks')
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(),'Could not verify the reviewed video frames')
    try:
        for index,item in sorted(frames.items()):
            require(item['label'].strip() and item['visual_observation'].strip(), 'Visual observation is missing')
            require(set(item['original_reference_approx_seconds'])<=references.keys(), 'A visual landmark lacks its original frame')
            require(math.isclose(item['progress_m'],previews[index]['progress_m'],abs_tol=1e-8) and
                    math.isclose(item['playback_seconds'],index/report_video['fps'],abs_tol=1e-8),
                    'Visual review does not match physical frame alignment')
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image = capture.read()
            require(ok and list(image.shape[1::-1])==report_video['resolution'], 'Reviewed frame could not be decoded')
            encoded,jpeg = cv2.imencode('.jpg',image)
            require(encoded and hashlib.sha256(jpeg).hexdigest()==item['sha256']==sha(item['path']),
                    'Reviewed image differs from its decoded video frame')
    finally:
        capture.release()
    return dict(path=str(review_path.resolve()),sha256=sha(review_path),reviewed_frames=len(frames),
                original_reference_frames=len(references),video_sha256=report_video['sha256'])


def verify_evidence(job_path, technical_path, review_path):
    job = read(job_path)
    require_complete_suite(job)
    technical = read(technical_path)
    require(technical['status']=='technical_evidence_verified_visual_review_pending' and
            Path(technical['source_job']).resolve()==job_path.resolve() and technical['source_job_sha256']==sha(job_path),
            'Could not bind the technical audit to the full suite')
    identity = evaluation_identity(Path(job['training_run']))
    require(job['identity']==identity and job['checkpoint_sha256']==technical['checkpoint_sha256']==identity['checkpoint_sha256'],
            'Navigation checkpoint or evaluation identity changed')
    require(technical['stages']==len(STAGES) and technical['trials']==len(technical['results'])==200 and
            technical['stage_thresholds_unchanged'] and technical['saved_physics_state_rate_hz']==5 and
            technical['physics_contact_checks_rate_hz']==500, 'Full physical audit is incomplete')
    assets = technical['assets']
    require(Path(assets['dataset_root'])==Path('/home/pc/Datasets/atec-opendataset') and assets['source_3dgs_files_verified']==28 and
            assets['simulation_only'] and technical['evaluation_terrain_manifest_sha256']==identity['terrain_sha256']['manifest.json'],
            'Competition route asset audit does not match the selected physical run')

    expected = {(stage['stage'],trial['mode'],trial['seed']):trial
                for stage in job['stages'] for trial in stage['evaluations']}
    audited = {(row['stage'],row['mode'],row['seed']):row for row in technical['results']}
    require(expected.keys()==audited.keys() and len(expected)==200, 'Audited trial cohort differs from the frozen suite')
    for key,row in audited.items():
        primary = expected[key]
        physical_run = Path(row['run']).resolve()
        require(physical_run==Path(primary['output']).resolve() and row['success']==primary['success'] and
                row['reason']==primary['reason'] and math.isclose(row['progress_m'],primary['progress_m'],abs_tol=1e-8) and
                row['physics_steps']>0, 'Audited physical trial differs from the suite')
        files = row['evidence_sha256']
        require(set(files)=={'metrics.json','run_manifest.json','physical_states.npz','policy_inputs.npz','observations.npz'} and
                all(sha(physical_run/name)==digest for name,digest in files.items()), f'Physical evidence changed in {physical_run}')

    video = technical['video']
    original = job['video']
    source_run = Path(original['source_run']).resolve()
    chosen = next((row for row in audited.values() if row['stage']=='noisy100' and row['mode']=='mean' and
                   row['success'] and Path(row['run']).resolve()==source_run),None)
    require(chosen and original['teacher_used'] is False and video['video']==original['video'] and
            video['all_saved_states_rendered_once'] and video['fps']==5 and video['resolution']==[1280,720] and
            video['decoded_frames']==original['decoded_frames']==chosen['saved_states'], 'Wrong submitted video or source trajectory')
    overlay = video['overlay_audit']
    require(overlay['decoded_overlay_frames']==video['decoded_frames'] and overlay['video_sha256']==video['sha256'] and
            overlay['physical_states_sha256']==chosen['evidence_sha256']['physical_states.npz'], 'HUD evidence does not cover every physical state')
    visual = validate_visual_review(video,review_path)
    summary = {stage['stage']:{mode:sum(row['success'] for row in stage['evaluations'] if row['mode']==mode)
                              for mode in ('mean','sample')} for stage in job['stages']}
    return dict(status='formal_acceptance_verified',formal_acceptance_complete=True,visual_review_complete=True,
                source_job=str(job_path.resolve()),source_job_sha256=sha(job_path),technical_audit=str(technical_path.resolve()),
                technical_audit_sha256=sha(technical_path),visual_review=visual,checkpoint_sha256=identity['checkpoint_sha256'],
                terrain_asset_sha256=identity['terrain_sha256']['manifest.json'],stages=len(STAGES),trials=len(audited),
                stage_successes=summary,physics_steps=sum(row['physics_steps'] for row in audited.values()),
                video_path=video['video'],video_sha256=video['sha256'],video_frames=video['decoded_frames'],
                video_fps=video['fps'],video_resolution=video['resolution'],simulation_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('technical_audit',type=Path)
    parser.add_argument('visual_review',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(),'Choose a new acceptance record')
    proof = verify_evidence(args.job,args.technical_audit,args.visual_review)
    with tempfile.TemporaryDirectory(dir=args.output.parent,prefix='.acceptance-') as scratch:
        temporary = Path(scratch)/args.output.name
        temporary.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
        os.link(temporary,args.output)
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':
    main()
