"""Apply the existing ten-stage gates to one explicit frozen spatial policy.

This entrypoint never changes the default evaluator or the isolated runner.
Completed results additionally require the declared spatial factory and source
identity, so accidentally running the legacy observations cannot pass a gate.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from _bootstrap import ROOT
from evaluate_base_checkpoint import STAGES, parallel_modes_available
from evaluate_spatial_checkpoint import evaluation_identity, sha

DEFAULT_ORDER = ['course100','noisy100','course50','stairs20','stairs15',
                 'stairs10','stairs6','stairs3','curve','flat']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_batch(destination, identity, seeds, mode, stage, config, finished):
    """Validate primary trial evidence, including normally failed physical trials."""
    destination = Path(destination)
    batch = json.loads((destination/'batch_metrics.json').read_text())
    provenance = json.loads((destination.parent/(destination.name+'.spatial_evaluation_provenance.json')).read_text())
    require(provenance['identity'] == identity, 'Frozen spatial evaluation identity mismatch')
    require(provenance['optimizer_updates_performed'] == 0, 'Evaluation performed optimizer updates')
    require(provenance['status'] == 'completed' if finished else provenance['status'] in ('running','completed'),
            'Spatial evaluation did not complete normally')
    start, length, required = STAGES[stage]
    require(batch['checkpoint_sha256'] == identity['checkpoint_sha256'], 'Frozen checkpoint mismatch')
    require(batch['planned_seeds'] == seeds and batch['action_mode'] == mode, 'Seed/mode mismatch')
    require(batch['required_successes'] == required and batch['teacher_used'] is False, 'Gate/teacher mismatch')
    rows = batch['results']
    require([r['seed'] for r in rows] == seeds[:len(rows)] and len(rows) <= len(seeds), 'Invalid trial sequence')
    require(batch['complete'] == (len(rows) == len(seeds)), 'Incorrect completion flag')
    unreachable = sum(not r['success'] for r in rows) > len(seeds)-required
    require(batch['gate_unreachable'] == unreachable, 'Incorrect unreachable flag')
    require(not finished or batch['complete'] or unreachable, 'Batch ended without completing its trials')
    expected_noise = {**config.get('noise', {})}
    if stage == 'noisy100':
        expected_noise['enabled'] = True
    for row in rows:
        run = destination/f"seed_{row['seed']}"
        require(Path(row['output']).resolve() == run.resolve(), 'Trial output mismatch')
        manifest = json.loads((run/'run_manifest.json').read_text())
        metrics = json.loads((run/'metrics.json').read_text())
        require(manifest['status'] == 'evaluated' and manifest['policy'] == 'DreamerV3_checkpoint', 'Invalid policy trial')
        require(manifest['checkpoint_sha256'] == identity['checkpoint_sha256'], 'Trial checkpoint mismatch')
        require(manifest['source_sha256'] == identity['source_sha256']['scripts/evaluate_dreamer.py'], 'Evaluator source mismatch')
        require(manifest['seed'] == manifest['policy_seed'] == row['seed'] and manifest['action_mode'] == mode,
                'Trial seed/mode mismatch')
        require(manifest['teacher_used'] is False and manifest['base_pose_teleported_after_reset'] is False,
                'Teacher/pose override detected')
        require(manifest['checkpoint_reloaded'] and manifest['policy_actions_finite'], 'Invalid policy checkpoint/actions')
        require(manifest['depth_noise'] == expected_noise, 'Depth noise contract mismatch')
        for evidence in (manifest, metrics):
            require(evidence['evaluation_completed'] and evidence['eligible_for_navigation_gate'], 'Aborted evaluation')
            require(evidence['start_s'] == start, 'Start position mismatch')
        require(metrics['target_length_m'] == metrics['route_length_m'] == length, 'Route length mismatch')
        require(metrics['route_projection'] == identity['route_projection'], 'Route projection mismatch')
        require(metrics['spatial_waypoint_reference'] == identity['observation_contract']['spatial_waypoint_reference'],
                'Actual spatial observation contract mismatch')
        require(metrics['spatial_observation_source_sha256'] ==
                identity['source_sha256']['src/atec_nav/rl/spatial_waypoint_observation.py'], 'Observation implementation mismatch')
        require(metrics['navigation_policy'] == 'agent_actions', 'Actions did not come from the agent')
        if identity.get('controller_transport'):
            require(metrics.get('speed_transport',{}).get('settings')==identity['controller_transport'],
                    'Actual calibrated transport contract mismatch')
            require(Path(metrics['speed_transport']['packet_log']).is_file(),'Missing actual AppCmd evidence')
        require(metrics['success'] == row['success'] and metrics['reason'] == row['reason'], 'Trial result mismatch')
        require(math.isfinite(row['progress_m']) and abs(metrics['final_progress_m']-row['progress_m']) < 1e-8,
                'Progress mismatch')
        require(metrics['physics_steps'] > 0 and metrics['steps'] > 0, 'Empty physical evaluation')
        if row['success']:
            require(metrics['reason'] == 'success' and metrics['final_progress_m'] >= length-.15
                    and metrics['goal_distance_m'] < .25 and not metrics['collision_contacts'], 'Invalid arrival claim')
    return batch


def wait_for_memory(save, minimum_gib=10):
    stable = None
    while True:
        available = next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))
        pressure = float(dict(x.split('=') for x in Path('/proc/pressure/memory').read_text().splitlines()[0].split()[1:])['avg10'])
        stable = (stable if stable is not None else time.monotonic()) if available >= minimum_gib and pressure < 1 else None
        save(status='waiting_for_memory', available_gib=available, memory_pressure_avg10=pressure)
        if stable is not None and time.monotonic()-stable >= 30:
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=6001)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=DEFAULT_ORDER)
    parser.add_argument('--parallel-modes', choices=['auto','no'], default='auto')
    args = parser.parse_args()
    if args.seed_start < 0 or len(set(args.stages)) != len(args.stages):
        parser.error('Distinct stages and nonnegative seeds required')
    training, output = args.training_run.resolve(), args.output.resolve()
    identity = evaluation_identity(training)
    config = json.loads((training/'integration.json').read_text())['config']
    full_suite = set(args.stages) == set(STAGES)
    seeds = list(range(args.seed_start,args.seed_start+10))
    output.mkdir(parents=True,exist_ok=False)
    status = dict(pid=os.getpid(),status='starting',started_unix=time.time(),evaluation_only=True,
                  learned_100m_success=False,formal_acceptance_complete=False,stages=[],
                  scope='competition_100m' if full_suite else 'spatial_frozen_policy_stages',
                  training_run=str(training),checkpoint_sha256=identity['checkpoint_sha256'],
                  identity=identity,evaluation_seeds=seeds)
    def save(**changes):
        status.update(changes,updated_unix=time.time())
        temporary=output/'job.json.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(output/'job.json')
    def check_identity():
        require(evaluation_identity(training) == identity, 'Frozen inputs changed during the suite')
    children=[]
    save()
    try:
        for name in args.stages:
            start,length,required=STAGES[name]
            folder=output/name;folder.mkdir()
            row=dict(stage=name,start_s=start,length_m=length,required_successes=required,evaluations=[],
                     depth_noise_enabled=name=='noisy100' or config.get('noise',{}).get('enabled',False))
            status['stages'].append(row);save(active_stage=name)
            wait_for_memory(save);check_identity()
            parallel=parallel_modes_available(args.parallel_modes)
            row['evaluation_parallelism']=2 if parallel else 1
            groups=[('mean','sample')] if parallel else [('mean',),('sample',)]
            for group_index,group in enumerate(groups):
                if group_index:
                    wait_for_memory(save);check_identity()
                with ExitStack() as stack:
                    active={}
                    for mode in group:
                        destination=folder/mode
                        command=['bash',str(ROOT/'scripts/run_matrix_isolated.sh'),'--script',
                                 str(ROOT/'scripts/evaluate_spatial_checkpoint.py'),str(training),
                                 '--output',str(destination),'--start-s',str(start),'--length',str(length),
                                 '--steps',str(int(max(60.,length*7.)*config['environment']['control_hz'])+1),
                                 '--action-mode',mode,'--seeds',*map(str,seeds),'--required-successes',str(required)]
                        if name=='noisy100':command.append('--depth-noise')
                        log=stack.enter_context((folder/(mode+'.log')).open('w'))
                        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                        children.append(child);active[mode]=(child,destination)
                    save(status='evaluating',active_children={mode:p.pid for mode,(p,_) in active.items()})
                    while True:
                        running=False
                        for mode,(child,destination) in active.items():
                            code=child.poll()
                            if code is not None and code!=0:raise RuntimeError(f'{name}/{mode} evaluator exited {code}')
                            if (destination/'batch_metrics.json').exists():
                                batch=validate_batch(destination,identity,seeds,mode,name,config,code==0)
                                row['evaluations']=[r for r in row['evaluations'] if r['mode']!=mode]+[
                                    {'mode':mode,**r} for r in batch['results']]
                                row[mode+'_successes']=sum(r['success'] for r in batch['results'])
                                row[mode+'_trials']=len(batch['results']);row[mode+'_complete']=batch['complete']
                            elif code==0:raise RuntimeError('Evaluator exited without a batch report')
                            running |= code is None
                        save()
                        if not running:break
                        time.sleep(5)
                check_identity()
                if any(not row[mode+'_complete'] or row[mode+'_successes']<required for mode in group):
                    row.update(passed=False,evaluation_complete=all(row.get(m+'_complete',False) for m in ('mean','sample')))
                    save(status='stage_failed_needs_diagnosis');return
            row.update(passed=True,evaluation_complete=True);save()
        if full_suite:
            chosen=next(r for r in status['stages'] if r['stage']=='noisy100')
            trial=next(r for r in chosen['evaluations'] if r['success'] and r['mode']=='mean')
            run=Path(trial['output']);save(status='recording',recording_run=str(run))
            recording=config.get('recording',{});env=os.environ.copy();env.pop('PYTHONPATH',None)
            cuda=recording.get('cuda_home','/home/pc/miniconda3/envs/taska')
            env.update(MUJOCO_GL='egl',CUDA_HOME=cuda,CPLUS_INCLUDE_PATH=cuda+'/targets/x86_64-linux/include',
                       TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
            with (output/'recording.log').open('w') as log:
                subprocess.run([str(ROOT/recording.get('python','.venvs/scene/bin/python')),
                                str(ROOT/'scripts/record_verified_navigation.py'),str(run),'--width','1280','--height','720'],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            video=json.loads((run/'dreamer_navigation_3dgs.verified.json').read_text())
            require(video['navigation_success'] and video['teacher_used'] is False and video['route_length_m']>=100
                    and video['decoded_frames']>0 and video['source_run']==str(run)
                    and video['sha256']==sha(video['video']), 'Invalid verified policy video')
            check_identity();save(video=video,learned_100m_success=True)
        # A separate final visual/evidence review remains necessary even when all gates pass.
        save(status='all_stages_passed',final_review_required=full_suite)
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        save(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
