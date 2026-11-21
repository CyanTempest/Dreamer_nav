"""Evaluate an explicit frozen spatial-observation checkpoint with provenance.

Use through run_matrix_isolated.sh --script. The shared physical evaluator and
the default environment remain unchanged. A trial is never a suite-level pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    'scripts/evaluate_spatial_checkpoint.py', 'scripts/evaluate_dreamer.py',
    'scripts/evaluate_spatial_checkpoint_suite.py', 'scripts/evaluate_base_checkpoint.py',
    'scripts/run_matrix_isolated.sh', 'scripts/record_verified_navigation.py',
    'src/atec_nav/sim/spatial_mujoco_env.py', 'src/atec_nav/sim/mujoco_env.py',
    'src/atec_nav/rl/spatial_waypoint_observation.py', 'src/atec_nav/rl/waypoint_observation.py',
    'src/atec_nav/robots/spatial_reference.py', 'src/atec_nav/robots/reference_tracking.py',
    'src/atec_nav/sim/metric_route_projection.py', 'src/atec_nav/sim/mujoco_route.py',
    'src/atec_nav/robots/matrix_controller.py', 'src/atec_nav/rl/navigation_signals.py',
    'src/atec_nav/robots/calibrated_matrix_controller.py',
    'src/atec_nav/rl/stagnation_observation.py', 'src/atec_nav/rl/adapter.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_identity(training):
    training = Path(training)
    metadata = json.loads((training/'integration.json').read_text())
    config = metadata['config']
    if (config['environment'].get('factory') != 'atec_nav.sim.spatial_mujoco_env:make_env'
            or config['environment'].get('route_projection') != 'metric_heading_v1'
            or config['observations'].get('spatial_waypoint_reference', {}).get('kind') != 'local_spatial_v1'
            or config['observations'].get('schema') != 'physical_v2'):
        raise ValueError('Explicit physical_v2 spatial observation contract required')
    terrain = ROOT/config['environment']['terrain_path']
    robot = ROOT/config['environment']['robot_path']
    if config['environment'].get('speed_transport'):
        from atec_nav.robots.calibrated_matrix_controller import calibrated_target
        calibrated_target(0., config['environment']['speed_transport'])
        if config['actions']['vx_mps']!=.5:
            raise ValueError('Unvalidated speed/transport contract')
    return {
        'checkpoint_sha256': sha(training/'final/agent.pkl'),
        'integration_sha256': sha(training/'integration.json'),
        'agent_config_sha256': sha(training/'config.yaml'),
        'source_sha256': {name: sha(ROOT/name) for name in CODE_FILES},
        'terrain_sha256': {name: sha(terrain/name) for name in ('manifest.json','centerline.csv','terrain.npz','obstacles.npz')},
        'robot_sha256': {name: sha(robot/name) for name in ('manifest.json','robot.xml')},
        'observation_contract': config['observations'],
        'environment_factory': config['environment']['factory'],
        'route_projection': config['environment']['route_projection'],
        'action_contract': config['actions'],
        'controller_transport': config['environment'].get('speed_transport'),
    }


def main():
    # --script uses the scene interpreter. Re-exec once into the configured
    # Dreamer environment while retaining the private network and shared memory.
    if os.environ.get('ATEC_SPATIAL_CHECKPOINT_INTERPRETER_READY') != '1':
        executable = Path(os.environ.get('ATEC_DREAMER_PYTHON', str(ROOT/'.venvs/dreamer-gpu/bin/python')))
        if not executable.is_file():
            raise ValueError('Configured Dreamer Python does not exist')
        os.environ['ATEC_SPATIAL_CHECKPOINT_INTERPRETER_READY'] = '1'
        os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ.setdefault('JAX_PLATFORMS', os.environ.get('ATEC_JAX_PLATFORM', 'cpu'))
    import _bootstrap  # Adds the project source directory before environment imports.
    import evaluate_dreamer as evaluator
    from atec_nav.sim.spatial_mujoco_env import make_env
    if '--help' in sys.argv or '-h' in sys.argv:
        evaluator.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('training_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    training, out = args.training_run.resolve(), args.output.resolve()
    identity = evaluation_identity(training)
    if out.exists():
        raise ValueError('Choose a new evaluation output')
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.parent/(out.name+'.spatial_evaluation_provenance.json')
    if sidecar.exists():
        raise ValueError('Evaluation provenance already exists')
    report = {'claim':'frozen_spatial_policy_trials_require_separate_full_suite_verification',
              'status':'running', 'optimizer_updates_performed':0, 'identity':identity,
              'arguments':sys.argv[1:]}
    def save(path):
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(path)
    save(sidecar)
    original_factory = evaluator.make_env
    evaluator.make_env = make_env
    try:
        evaluator.main()
        if evaluation_identity(training) != identity:
            raise ValueError('Frozen evaluation inputs changed during the batch')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='aborted',error=repr(error))
        raise
    finally:
        evaluator.make_env = original_factory
        save(sidecar)
        if out.exists():
            save(out/'spatial_evaluation_provenance.json')


if __name__ == '__main__':
    main()
