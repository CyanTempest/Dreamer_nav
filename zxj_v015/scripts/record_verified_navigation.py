"""Render native 3DGS video, then decode every frame before reporting success."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--name', default='dreamer_navigation_3dgs.mp4')
    p.add_argument('--allow-failure', action='store_true',
                   help='Export a diagnostic failure replay; does not mark navigation successful')
    a = p.parse_args()
    run = a.run.resolve()
    metrics = json.loads((run / 'metrics.json').read_text())
    if not metrics['success'] and not a.allow_failure:
        raise ValueError('Success recording requires an actually successful physical evaluation')
    subprocess.run([sys.executable, str(ROOT / 'scripts/record_gaussian_navigation.py'),
                    str(run), '--width', str(a.width), '--height', str(a.height),
                    '--name', a.name, '--camera-distance', '3.5',
                    '--camera-elevation', '-20', '--camera-azimuth-offset', '0'], check=True)
    video = run / a.name
    with np.load(run / 'physical_states.npz') as states:
        expected = len(states['time'])
    manifest = json.loads((run / 'run_manifest.json').read_text())
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (a.height, a.width):
                raise ValueError('Video resolution does not match the requested export')
            count += 1
    finally:
        capture.release()
    if count != expected or count == 0 or not np.isclose(fps, manifest['fps']):
        raise ValueError(f'Incomplete video: {count}/{expected} frames, {fps} fps')
    digest = hashlib.sha256()
    with video.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    report = {'video': str(video), 'decoded_frames': count, 'fps': fps,
              'resolution': [a.width, a.height], 'sha256': digest.hexdigest(),
              'source': 'saved_actual_physical_evaluation_states',
              'navigation_success': bool(metrics['success']),
              'navigation_result': metrics.get('reason', metrics.get('result', 'unknown')),
              'source_run': str(run), 'teacher_used': manifest.get('teacher_used'),
              'route_length_m': metrics['route_length_m']}
    video.with_suffix('.verified.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
