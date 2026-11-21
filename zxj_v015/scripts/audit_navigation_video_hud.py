"""Check decoded navigation video overlays against the actual saved physics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import ROOT
from audit_frozen_navigation_suite import read, require, sha


def expected_overlay(width, time, progress, length, speed, finished):
    canvas = np.empty((76, width, 3), np.uint8)
    canvas[:] = (22, 28, 30)
    status = 'goal reached' if finished else 'running'
    cv2.putText(canvas, f'ATEC | DreamerV3 + MATRiX | checkpoint evaluation | {status}',
                (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 244, 245), 2, cv2.LINE_AA)
    measurements = f't={time:.1f}s   Route {progress:.2f}/{length:.1f} m   v={speed:.2f} m/s'
    cv2.putText(canvas, measurements, (18, 59), cv2.FONT_HERSHEY_SIMPLEX, .62,
                (110, 230, 245), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)


def core_disagreements(actual, expected):
    # Compare the distinct text and background cores. Antialiasing and lossy
    # H.264 can move the luminance of pixels along glyph edges.
    missing_text = (expected >= 145) & (actual < 100)
    unexpected_text = (expected <= 68) & (actual >= 100)
    return int(np.count_nonzero(missing_text | unexpected_text))


def audit(run, video):
    metrics = read(run/'metrics.json')
    manifest = read(run/'run_manifest.json')
    recording = read(video.with_suffix('.recording.json'))
    verification = read(video.with_suffix('.verified.json'))
    require(metrics['success'] and manifest['policy'] == 'DreamerV3_checkpoint'
            and manifest['evaluation_completed'] and manifest['teacher_used'] is False
            and manifest['base_pose_teleported_after_reset'] is False, 'Expected successful Dreamer physics')
    require(Path(verification['source_run']).resolve() == run.resolve() and verification['sha256'] == sha(video),
            'Recording does not match the verified source run')
    require(recording['renderer_sha256'] == sha(ROOT/'scripts/record_gaussian_navigation.py'),
            'Overlay renderer changed')
    with np.load(run/'physical_states.npz') as trace:
        times = trace['time']
        progress = trace['progress_m'] - manifest['start_s']
        speed = np.linalg.norm(trace['qvel'][:, :2], axis=1)
    count = len(times)
    require(count == verification['decoded_frames'] and count > 0, 'Physical sample count differs from the video')
    width, height = verification['resolution']
    max_errors, preview_scores = 0, []
    preview_indices = {0, count // 2, count - 1}
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened(), 'Video could not be decoded')
    try:
        for i, (time, distance, velocity) in enumerate(zip(times, progress, speed, strict=True)):
            ok, frame = capture.read()
            require(ok and frame.shape[:2] == (height, width), f'Invalid decoded frame {i}')
            overlay = cv2.cvtColor(frame[:76], cv2.COLOR_BGR2GRAY)
            reference = expected_overlay(width, time, distance, metrics['route_length_m'], velocity, i == count - 1)
            disagreements = core_disagreements(overlay, reference)
            require(disagreements <= 2, f'Video overlay does not match physical state {i}: {disagreements} pixels')
            max_errors = max(disagreements, max_errors)
            if i in preview_indices:
                path = video.with_name(video.stem + f'_{i:06d}.png')
                lossless = cv2.imread(str(path))
                require(lossless is not None and lossless.shape == frame.shape, f'Invalid renderer reference {path}')
                psnr = cv2.PSNR(frame[77:], lossless[77:])
                preview_scores.append(dict(frame=i, reference_sha256=sha(path), scene_psnr_db=psnr))
        require(not capture.read()[0], 'Video contains extra frames')
    finally:
        capture.release()
    return dict(video=str(video.resolve()), video_sha256=verification['sha256'],
                physical_states_sha256=sha(run/'physical_states.npz'),
                audit_script_sha256=sha(Path(__file__)),
                decoded_overlay_frames=count, maximum_core_pixel_disagreements=max_errors,
                maximum_allowed_core_pixel_disagreements=2,
                checked_overlay_values=['simulation time', 'route progress', 'horizontal speed', 'arrival status'],
                lossless_renderer_frame_comparisons=preview_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report = audit(args.run, args.video)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
