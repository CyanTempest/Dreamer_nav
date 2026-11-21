"""Independently evaluate a frozen physical_v2 checkpoint on measured stage gates."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time

from _bootstrap import ROOT


STAGES={'flat':(65.,3.,9),'curve':(55.,3.,9),
        **{f'stairs{length}':(0.,float(length),8) for length in (3,6,10,15,20)},
        'course50':(0.,50.,8),'course100':(0.,100.,8),'noisy100':(0.,100.,8)}


def parallel_modes_available(setting):
    if setting!='auto':return setting=='yes'
    try:
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
        # Each current physical_v2 evaluator peaks near 5 GiB. Leave room for
        # the desktop and other jobs; never reserve or stop their resources.
        return available>=14*1024**3
    except (OSError,StopIteration,ValueError):return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-start',type=int,default=3001)
    parser.add_argument('--stages',nargs='+',choices=STAGES,default=['flat','curve'])
    parser.add_argument('--parallel-modes',choices=['auto','yes','no'],default='auto')
    args=parser.parse_args()
    full_suite=set(args.stages)==set(STAGES)
    training=args.training_run.resolve();output=args.output.resolve()
    audit=json.loads((training/'integration.json').read_text())
    if audit['config']['observations'].get('schema')!='physical_v2':
        parser.error('Expected a physical_v2 checkpoint')
    if args.seed_start<0:parser.error('Expected nonnegative seeds')
    if len(set(args.stages))!=len(args.stages):parser.error('Duplicate stages')
    output.mkdir(parents=True,exist_ok=False)
    seeds=list(range(args.seed_start,args.seed_start+10))
    scope=('competition_100m' if full_suite else
           'physical_v2_flat_curve_pretraining' if args.stages==['flat','curve'] else 'physical_v2_frozen_policy_stages')
    status={'pid':os.getpid(),'status':'starting','scope':scope,
            'evaluation_only':True,'started_unix':time.time(),'learned_100m_success':False,
            'evaluation_seeds':seeds,'stages':[],
            'training_run':str(training),'checkpoint_sha256':hashlib.sha256((training/'final/agent.pkl').read_bytes()).hexdigest()}
    def save():
        status['updated_unix']=time.time()
        path=output/'job.json.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(output/'job.json')
    children=[]
    save()
    try:
        for name in args.stages:
            start,length,required=STAGES[name]
            folder=output/name;folder.mkdir()
            row={'stage':name,'start_s':start,'length_m':length,'required_successes':required,
                 'depth_noise_enabled':name=='noisy100' or audit['config'].get('noise',{}).get('enabled',False),
                 'evaluation_training_run':str(training),'evaluations':[]}
            status['stages'].append(row);status.update(status='evaluating',active_stage=name);save()
            parallel=parallel_modes_available(args.parallel_modes)
            row['evaluation_parallelism']=2 if parallel else 1
            groups=[('mean','sample')] if parallel else [('mean',),('sample',)]
            for group in groups:
                with ExitStack() as stack:
                    active={}
                    for mode in group:
                        destination=folder/mode
                        command=['bash',str(ROOT/'scripts/run_matrix_isolated.sh'),'--evaluate',str(training),
                                 '--output',str(destination),'--start-s',str(start),'--length',str(length),
                                 '--steps',str(int(max(60.,length*7.)*audit['config']['environment']['control_hz'])+1),
                                 '--action-mode',mode,'--seeds',*map(str,seeds),'--required-successes',str(required)]
                        if name=='noisy100':command.append('--depth-noise')
                        log=stack.enter_context((folder/(mode+'.log')).open('w'))
                        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        children.append(child);active[mode]=(child,destination)
                    while True:
                        running=False
                        for mode,(child,destination) in active.items():
                            code=child.poll()
                            if code is not None and code!=0:raise RuntimeError(f'{name}/{mode} evaluator exited {code}')
                            report=destination/'batch_metrics.json'
                            if report.exists():
                                batch=json.loads(report.read_text())
                                if batch.get('checkpoint_sha256')!=status['checkpoint_sha256']:
                                    raise ValueError('Evaluation did not use the frozen checkpoint selected for this suite')
                                row['evaluations']=[r for r in row['evaluations'] if r['mode']!=mode]+[
                                    {'mode':mode,**r} for r in batch['results']]
                                row[mode+'_successes']=sum(r['success'] for r in batch['results'])
                                row[mode+'_trials']=len(batch['results'])
                                row[mode+'_complete']=batch['complete']
                            elif code==0:raise RuntimeError('Evaluator exited without its batch report')
                            running |= code is None
                        save()
                        if not running:break
                        time.sleep(5)
                if any(not row[mode+'_complete'] or row[mode+'_successes']<required for mode in group):
                    row['passed']=False
                    row['evaluation_complete']=all(row.get(mode+'_complete',False) for mode in ('mean','sample'))
                    status['status']='stage_failed_needs_diagnosis';save();return
            row.update(passed=True,evaluation_complete=True);save()
        if full_suite:
            # A short-stage suite or a single 100 m pass cannot trigger the
            # complete-task claim. Keep one checkpoint for the entire suite.
            chosen=next(row for row in status['stages'] if row['stage']=='noisy100')
            trial=next(row for row in chosen['evaluations'] if row['success'] and row['mode']=='mean')
            run=Path(trial['output'])
            status.update(status='recording',recording_run=str(run));save()
            recording=audit['config'].get('recording',{})
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            cuda=recording.get('cuda_home','/home/pc/miniconda3/envs/taska')
            env.update(MUJOCO_GL='egl',CUDA_HOME=cuda,CPLUS_INCLUDE_PATH=cuda+'/targets/x86_64-linux/include',
                       TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
            with (output/'recording.log').open('w') as log:
                subprocess.run([str(ROOT/recording.get('python','.venvs/scene/bin/python')),
                    str(ROOT/'scripts/record_verified_navigation.py'),str(run),'--width','1280','--height','720'],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            video=json.loads((run/'dreamer_navigation_3dgs.verified.json').read_text())
            if not (video['navigation_success'] and video['teacher_used'] is False and video['route_length_m']>=100):
                raise ValueError('Video does not certify an independent full-course policy rollout')
            status.update(video=video,learned_100m_success=True)
        status['status']='all_stages_passed'
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        status.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
