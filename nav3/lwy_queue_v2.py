"""Resumable, bounded remote queue; validation alone selects the final models."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

STATE=Path('metrics/lwy_v2_queue.json')

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def main():
    global STATE
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['profiles','parallel-check','runtime-candidate','smoke-collectors','smoke','train','final','all'],default='all')
    p.add_argument('--released',action='store_true',help='Approved runtime2 run; preserve all previous queue artifacts')
    p.add_argument('--profile-start',type=int,default=4608)
    p.add_argument('--retry-failed',action='append',default=[],help='Explicitly diagnosed failed job tag; no automatic retries')
    a=p.parse_args()
    if a.released:STATE=Path('metrics/lwy_runtime2_queue.json')
    import fcntl
    queue_lock=Path('logs')/(STATE.stem+'.lock')
    lock_file=queue_lock.open('a')
    fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    artifact_prefix='lwy_runtime2' if a.released else 'lwy_v2'
    def run_name(method):return method+'_lwy'+('_runtime2' if a.released else '')
    from .lwy_eval_v2 import require_g1,protocol_hash
    from .lwy_comparison_runtime import level as evaluation_level,flags as evaluation_flags,suffix as evaluation_suffix,setup as comparison_setup
    require_g1()
    assert read('metrics/lwy_v2_random_feasibility.json')['accepted_unpassable']==0
    assert read('metrics/lwy_v2_policy_protocol.json')['passed']
    state=read(STATE) if STATE.exists() else dict(jobs={},training={},gpu_job_wall_seconds=0.,gpu_budget_seconds=12*3600,
        preceding_g1_and_diagnostics_allowance_seconds=1800.,created_utc=datetime.now(timezone.utc).isoformat())
    if a.released and not STATE.exists():
        prior=read('metrics/lwy_v2_queue.json');preflight=read('metrics/lwy_runtime2_preflight.json')
        choice=read('metrics/lwy_runtime2_ppo_memory_analysis.json')
        state.update(gpu_job_wall_seconds=prior['gpu_job_wall_seconds']+preflight['wall_seconds'],
            preceding_g1_and_diagnostics_allowance_seconds=prior['preceding_g1_and_diagnostics_allowance_seconds'],
            selected_envs=dict(dreamer=4608,ppo=choice['selected_envs']),training_generation='_runtime2',
            prior_budget_source='metrics/lwy_v2_queue.json',preflight_budget_source='metrics/lwy_runtime2_preflight.json',
            release_source='TASK_LWY3.md',run_suffix='_runtime2')
    previous=Path(f'/proc/{state.get("pid",0)}/cmdline')
    assert not (previous.exists() and b'nav3.lwy_queue_v2' in previous.read_bytes()), 'Another queue process is still active'
    if state.get('error'):
        state.setdefault('stage_failures',[]).append(dict(stage=state.get('stage'),error=state.pop('error'),
            updated_utc=state.get('updated_utc')))
    state.update(pid=os.getpid(),stage=a.stage,status='running');write(STATE,state)
    if Path('configs/lwy_v2_memory.json').exists():
        state['memory_configuration']=read('configs/lwy_v2_memory.json')
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(state['memory_configuration']['jax_pool_fraction'])
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']=state['memory_configuration'].get('jax_allocator','default')
    state['memory_configuration']['evaluation_autotune']=evaluation_level()
    state['memory_configuration']['ppo_training_xla_flags']=evaluation_flags()
    child=[None]
    parallel_children=[]
    def report():
        # Reporting is CPU-only and also runs after a failed GPU job.
        with Path(f'logs/{artifact_prefix}_report_updates.log').open('a') as output:
            subprocess.run(['.venv_lwy/bin/python','-m','nav3.lwy_report_runtime2' if a.released else 'nav3.lwy_report_v2'],
                stdout=output,stderr=subprocess.STDOUT,timeout=180,check=False)
    def interrupted(signum,frame):
        if child[0] is not None and child[0].poll() is None:os.killpg(child[0].pid,signal.SIGTERM)
        for process in parallel_children:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        state.update(status='interrupted',signal=signum);write(STATE,state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def run(tag,arguments,timeout=3000,allow_failure=False):
        if state['jobs'].get(tag,{}).get('returncode')==0:return state['jobs'][tag]
        suffix=''
        if tag in state['jobs'] and tag in a.retry_failed:
            history=state.setdefault('job_attempts',{}).setdefault(tag,[])
            history.append(state['jobs'].pop(tag));suffix=f'_retry{len(history)}';write(STATE,state)
        available=state['gpu_budget_seconds']-state['preceding_g1_and_diagnostics_allowance_seconds']-state['gpu_job_wall_seconds']
        if available<60:raise RuntimeError('GPU wall-clock budget exhausted')
        log=Path(f'logs/lwy_{tag}{suffix}.log');assert not log.exists(),f'Failed job needs diagnosis and an explicit retry: {tag}'
        before=time.monotonic();peak=0;active=0;last=before;timed_out=False
        env={**os.environ,'PYTHONUNBUFFERED':'1'};comparison_setup(environment=env)
        if arguments[0] in ['nav3.lwy_train_dreamer','nav3.train_ppo','nav3.lwy_memory_v2']:
            from .lwy_training_runtime import configure
            method=arguments[arguments.index('--method')+1] if arguments[0]=='nav3.lwy_memory_v2' else ('ppo' if arguments[0]=='nav3.train_ppo' else 'dreamer')
            configure(method,env)
        if arguments[0] in ['nav3.lwy_compare_v2','nav3.lwy_video_v2','nav3.train_ppo']:
            env['XLA_PYTHON_CLIENT_ALLOCATOR']='default';env['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.95'
        command=[sys.executable,'-m',*arguments]
        state.update(active_job=tag,active_log=str(log),active_started_utc=datetime.now(timezone.utc).isoformat());write(STATE,state)
        with log.open('x') as f:
            child[0]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while child[0].poll() is None:
                try:
                    memory,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    peak=max(peak,memory);now=time.monotonic();active+=(now-last)*(util>1);last=now
                except (subprocess.SubprocessError,ValueError):pass
                if time.monotonic()-before>min(timeout,available):
                    timed_out=True
                    os.killpg(child[0].pid,signal.SIGTERM)
                    try:child[0].wait(90)
                    except subprocess.TimeoutExpired:os.killpg(child[0].pid,signal.SIGKILL)
                    break
                time.sleep(1)
            raw_rc=child[0].wait();rc=124 if timed_out else raw_rc;child[0]=None
        result=dict(command=command,returncode=rc,wall_seconds=time.monotonic()-before,peak_used_mib=peak,
            observed_active_gpu_seconds=active,log=str(log),timed_out=timed_out,process_returncode=raw_rc)
        state['jobs'][tag]=result;state['gpu_job_wall_seconds']+=result['wall_seconds'];write(STATE,state)
        if rc and not allow_failure:raise RuntimeError(f'{tag} failed with exit {rc}; see {log}')
        return result
    def run_pair(jobs,timeout=3600):
        """Two independent evaluations share idle GPU intervals; never training.

        Budget accounting is elapsed occupancy of this one physical GPU, not
        the sum of overlapping process lifetimes. Each job retains its time.
        """
        pending=[(tag,args) for tag,args in jobs if state['jobs'].get(tag,{}).get('returncode')!=0]
        if not pending:return
        assert len(pending)<=2
        available=state['gpu_budget_seconds']-state['preceding_g1_and_diagnostics_allowance_seconds']-state['gpu_job_wall_seconds']
        if available<60:raise RuntimeError('GPU wall-clock budget exhausted')
        begin=time.monotonic();processes=[];peak=0;failed=False
        env={**os.environ,'PYTHONUNBUFFERED':'1'};comparison_setup(environment=env)
        env['XLA_PYTHON_CLIENT_ALLOCATOR']='default';env['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.95'
        state.update(active_job='parallel_evaluation',active_jobs=[tag for tag,_ in pending],
            active_logs=[f'logs/lwy_{tag}.log' for tag,_ in pending],active_log=None,
            active_started_utc=datetime.now(timezone.utc).isoformat());write(STATE,state)
        try:
            for tag,args in pending:
                log=Path(f'logs/lwy_{tag}.log');assert not log.exists(),f'Failed job requires diagnosis: {tag}'
                command=[sys.executable,'-m',*args]
                with log.open('x') as output:
                    process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                processes.append((tag,command,log,process,time.monotonic()));parallel_children.append(process)
            while any(p.poll() is None for _,_,_,p,_ in processes):
                peak=max(peak,int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)))
                for tag,command,log,p,start in processes:
                    rc=p.poll()
                    if rc is not None and tag not in state['jobs']:
                        state['jobs'][tag]=dict(command=command,returncode=rc,wall_seconds=time.monotonic()-start,log=str(log),
                            parallel_evaluation=True,shared_gpu_peak_mib=peak)
                        failed|=bool(rc);write(STATE,state)
                if time.monotonic()-begin>min(timeout,available):failed=True
                if failed:break
                time.sleep(1)
        finally:
            for tag,command,log,p,start in processes:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(90)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                rc=p.wait();failed|=bool(rc)
                state['jobs'].setdefault(tag,dict(command=command,returncode=rc,wall_seconds=time.monotonic()-start,
                    log=str(log),parallel_evaluation=True,shared_gpu_peak_mib=peak))
            parallel_children.clear()
            elapsed=time.monotonic()-begin;state['gpu_job_wall_seconds']+=elapsed
            state.setdefault('parallel_evaluation_batches',[]).append(dict(tags=[tag for tag,_ in pending],
                gpu_occupied_wall_seconds=elapsed,shared_peak_mib=peak))
            state['active_jobs']=[];write(STATE,state)
        if failed:raise RuntimeError('Parallel evaluation failed; inspect individual job logs')
    def parallel_check():
        results={};version=evaluation_suffix()
        for method in ['dreamer_s0','ppo_s0']:
            if version:
                serial=f'v2_parallel_protocol_{method}_blind_serial{version}'
                run(serial,['nav3.lwy_compare_v2','--method',method,'--dataset','blind','--episodes','5','--tag',serial])
                baseline=read(f'metrics/lwy_{serial}.json')
            else:baseline=read(f'metrics/lwy_v2_final_{method}_blind_ob0_sf0.json')
            jobs=[]
            for copy in ['a','b']:
                tag=f'v2_parallel_protocol_{method}_blind_{copy}{version}'
                jobs.append((tag,['nav3.lwy_compare_v2','--method',method,'--dataset','blind','--episodes','5','--tag',tag]))
            run_pair(jobs)
            for tag,_ in jobs:
                current=read(f'metrics/lwy_{tag}.json')
                equal=[x['action_pose_sha256']==y['action_pose_sha256'] for x,y in zip(baseline['rows'],current['rows'])]
                assert len(equal)==100 and all(equal),f'Concurrent inference differs for {method}'
            results[method]=dict(episodes_per_copy=100,concurrent_copies=2,exact_trajectory_hashes=200,dataset='blind')
        write(f'metrics/lwy_v2_parallel_protocol{version}.json',dict(passed=True,concurrent_processes=2,methods=results,xla_flags=evaluation_flags()))
        run('v2_video_protocol_probe'+version,['nav3.lwy_video_v2','--protocol-probe'],timeout=600)
    def runtime_candidate():
        """Reviewable alternate-runtime evidence; never activates that protocol."""
        before=state['gpu_job_wall_seconds'];results={}
        deadline=time.monotonic()+3600-state.get('outside_queue_diagnostic_gpu_allowance_seconds',900)
        def limit():
            remaining=deadline-time.monotonic()
            assert remaining>=60,'The one-hour numerical diagnosis budget is exhausted'
            return min(600,remaining)
        for method in ['dreamer_s0','ppo_s0']:
            def job(label,n=100):
                tag=f'v2_runtime_candidate_{method}_{label}'
                return tag,['nav3.lwy_compare_v2','--method',method,'--dataset','original',
                    '--episodes',str(n),'--tag',tag,'--diagnostic-autotune','1']
            tag,args=job('serial');run(tag,args,timeout=limit())
            run_pair([job('parallel_a'),job('parallel_b')],timeout=limit())
            tag,args=job('single',1);run(tag,args,timeout=limit())
            base=read(f'metrics/lwy_v2_runtime_candidate_{method}_serial.json')
            hashes=[r['action_pose_sha256'] for r in base['rows']]
            assert len(hashes)==100
            for label in ['parallel_a','parallel_b']:
                other=read(f'metrics/lwy_v2_runtime_candidate_{method}_{label}.json')
                assert hashes==[r['action_pose_sha256'] for r in other['rows']],(method,label)
            one=read(f'metrics/lwy_v2_runtime_candidate_{method}_single.json')
            assert hashes[0]==one['rows'][0]['action_pose_sha256'],method
            results[method]=dict(cold_process_copies=3,identical_full_trajectory_hashes=300,
                one_vs_100_exact=True,successes=base['successes'],mean_gates=base['mean_gates'],trav_iou=base['trav_iou'])
        qualification={}
        for dataset,n in [('original',10),('blind',1)]:
            tag=f'v2_runtime_candidate_planner_{dataset}'
            run(tag,['nav3.lwy_compare_v2','--method','planner','--dataset',dataset,
                '--episodes',str(n),'--tag',tag,'--diagnostic-autotune','1'],timeout=limit())
            current=read(f'metrics/lwy_{tag}.json')
            reference=read(f'metrics/lwy_v2_round1_{dataset}.json')
            qualification[dataset]=dict(episodes=current['episodes'],successes=current['successes'],
                success_rate=current['success_rate'],same_level0_trajectory_hashes=sum(
                    x['action_pose_sha256']==y['action_pose_sha256'] for x,y in zip(current['rows'],reference['rows'])))
        assert qualification['original']['successes']==10 and qualification['blind']['success_rate']>=.95
        elapsed=state['gpu_job_wall_seconds']-before
        write('metrics/lwy_v2_runtime_candidate_protocol.json',dict(passed=True,adopted=False,
            environment_sha256=protocol_hash(),
            xla_flags='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true',
            independent_prng=True,single_environment_vmap=True,fixed_policy_streams=100,
            methods=results,qualification=qualification,gpu_occupied_wall_seconds=elapsed,
            diagnostic_gpu_upper_bound_seconds=state.get('outside_queue_diagnostic_gpu_allowance_seconds',900)+elapsed))
    def profile(method,n):
        tag=f'v2_memory_{method}_{n}_training_v2'
        path=Path(f'metrics/lwy_{tag}.json')
        fraction=float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION','.95'))
        allocator=os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR','default')
        if method=='ppo':allocator=state.get('memory_configuration',{}).get('ppo_allocator','default')
        if path.exists():
            previous=read(path)
            if previous.get('jax_pool_fraction',.95)!=fraction:tag+=f'_pool{round(fraction*100)}'
            if previous.get('jax_allocator','default')!=allocator:tag+=f'_{allocator}'
            path=Path(f'metrics/lwy_{tag}.json')
        job=run(tag,['nav3.lwy_memory_v2','--method',method,'--envs',str(n),'--output',str(path)],allow_failure=True)
        if not path.exists():write(path,dict(method=method,envs=n,completed=False,finite_loss=False,
            peak_used_mib=job['peak_used_mib'],peak_fraction=job['peak_used_mib']/24564,jax_pool_fraction=fraction,jax_allocator=allocator,
            error=f'Process exit {job["returncode"]}',samples=[]))
        return dict(read(path),artifact=str(path))
    def profiles():
        if 'selected_envs' in state:return
        # Full-bank measurements must not overlap another CUDA compute job.
        for _ in range(60):
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not active:break
            time.sleep(5)
        assert not active, 'Another CUDA job is active; leave it untouched and retry this profile later'
        n=a.profile_start;tested=[];selected=None
        if n!=4608:assert Path('metrics/lwy_v2_memory_dreamer_4608.json').exists(), 'Measure original 4608 first'
        for _ in range(8):
            r=profile('dreamer',n);tested.append(r)
            if r['completed'] and r['finite_loss'] and .85<=r['peak_fraction']<=.90:selected=n;break
            if r['completed'] and r['peak_fraction']<.85:
                if n==4608:selected=n;break
                higher=[x['envs'] for x in tested if x['envs']>n]
                proposed=128*((n+min(higher))//256) if higher else min(4608,n+512)
            else:
                good=[x['envs'] for x in tested if x['completed'] and x['peak_fraction']<.90]
                proposed=128*((max(good,default=0)+n)//256) if good else n-512
            if proposed<=0 or any(x['envs']==proposed for x in tested):break
            n=proposed
        assert selected is not None,'No measured Dreamer collector meets target; manual diagnosis required'
        pr=profile('ppo',128)
        assert pr['completed'] and pr['finite_loss'],'PPO original collector does not fit'
        state['selected_envs']=dict(dreamer=selected,ppo=128);write(STATE,state)
        write('metrics/lwy_v2_memory_selection.json',dict(selected_envs=state['selected_envs'],
            dreamer_profiles=[r['artifact'] for r in tested],
            ppo_profile=pr['artifact'],memory_configuration=state.get('memory_configuration'),
            target_fraction=[.85,.90],all_maps=True,real_agents=True))
    def initialize(method):
        name=method+'_lwy';out=Path('checkpoints')/run_name(method)
        if out.exists():
            assert (out/'latest.pkl').exists() and name in state['training'];return
        out.mkdir();shutil.copy2(f'checkpoints/{method}_s0/latest.pkl',out/'latest.pkl')
        state['training'][name]=dict(smoke_complete=False,rounds=[],best=None,bad_streak=0,stopped=False)
        write(STATE,state)
    def train_job(method,tag,seconds):
        tag+=state.get('training_generation','')
        n=state['selected_envs'][method]
        args=(['nav3.lwy_train_dreamer','--name',run_name(method),'--envs',str(n),'--seconds',str(seconds)] if method=='dreamer' else
              ['nav3.train_ppo','--name',run_name(method),'--lwy-round','--resume','--envs',str(n),'--seconds',str(seconds),'--steps','1000000000000'])
        return run(tag,args,timeout=seconds+1500)
    def snapshot(method,label):
        name=method+'_lwy'
        generation=state.get('training_generation','')
        snapshot=Path('checkpoints/snapshots')/(f'{run_name(method)}_{label}h' if a.released else f'{name}_{label}h{generation}');snapshot.mkdir(exist_ok=True)
        cp=snapshot/'latest.pkl'
        if not cp.exists():shutil.copy2(Path('checkpoints')/run_name(method)/'latest.pkl',cp)
        return cp
    def validate(method,label):
        if Path('metrics/lwy_v2_training_numerics_diagnosis.json').exists():
            assert Path('metrics/lwy_v2_evaluation_runtime_decision.json').exists(), 'Evaluation runtime decision is pending; do not silently change or use the affected protocol'
        name=method+'_lwy';train=state['training'][name]
        assert all(row.get('xla_flags','--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true')==evaluation_flags() for row in train['rounds']), 'Do not mix validation protocols'
        if any(row['label']==label for row in train['rounds']):return
        generation=state.get('training_generation','');cp=snapshot(method,label)
        tag=f'v2_val_{name}_{label}h{generation}{evaluation_suffix()}'
        run(tag,['nav3.lwy_compare_v2','--method',name,'--dataset','val','--episodes','2','--checkpoint',str(cp),'--tag',tag])
        result=read(f'metrics/lwy_{tag}.json')
        row=dict(label=label,checkpoint=str(cp),evaluation=f'metrics/lwy_{tag}.json',success_rate=result['success_rate'],mean_gates=result['mean_gates'],xla_flags=evaluation_flags())
        progress=Path('checkpoints')/run_name(method)/'progress.json'
        if progress.exists():row['training_progress']={k:v for k,v in read(progress).items() if k!='episode_mix'}
        previous=train['rounds'][-1] if train['rounds'] else None
        train['consecutive_declines']=(train.get('consecutive_declines',0)+1
            if previous and row['success_rate']<previous['success_rate'] else 0)
        if not any(r['label']==label for r in train['rounds']):train['rounds'].append(row)
        best=train['best']
        if best is None or (row['success_rate'],row['mean_gates'])>(best['success_rate'],best['mean_gates']):train['best']=row
        if best is not None and row['success_rate']<best['success_rate']-.30:train['bad_streak']+=1
        else:train['bad_streak']=0
        if train['bad_streak']>=2:train.update(stopped=True,stop_reason='two consecutive validation scores >30 percentage points below best')
        elif train['consecutive_declines']>=2:
            train.update(stopped=True,stop_reason='two successive decreases in validation success; continuation request requires diagnosis before further training')
        write(STATE,state);write(f'metrics/{artifact_prefix}_learning_curve.json',state['training'])
        report()
        if train['stopped']:
            run(f'v2_collapse_{name}_{label}h',['nav3.lwy_collapse_v2','--method',name,'--state',str(STATE),'--run-name',run_name(method)],timeout=3600)
    def smoke_preconditions():
        profiles()
        assert read('metrics/lwy_v2_checks.json')['passed']
        assert read('metrics/lwy_v2_sampler_probe.json')['passed']
        if Path('metrics/lwy_v2_gradient_health_alert.json').exists():
            assert read('metrics/lwy_v2_gradient_health_resolution.json')['cleared_for_resume'], 'Resolve gradient health alert before more learning'
    def smoke_collectors():
        """Training startup checks can proceed while an evaluation decision waits."""
        smoke_preconditions()
        for method in ['dreamer','ppo']:
            initialize(method);name=method+'_lwy';train=state['training'][name]
            if train.get('smoke_collectors_complete'):continue
            snapshot(method,'0.00')
            train_job(method,f'v2_smoke_{method}',360)
            progress=train.setdefault('smoke_first_progress',read(f'checkpoints/{name}/progress.json'))
            assert progress['updates']>0;write(STATE,state)
            snapshot(method,'0.10')
            train_job(method,f'v2_smoke_resume_{method}',300)
            resumed=read(f'checkpoints/{name}/progress.json')
            assert resumed['updates']>progress['updates'] and resumed['steps']>progress['steps'], 'Resumed collector made no progress'
            train['resume_verified']=dict(before=progress,after=resumed)
            train['smoke_job_wall_seconds']=sum(state['jobs'][tag+state.get('training_generation','')]['wall_seconds'] for tag in [f'v2_smoke_{method}',f'v2_smoke_resume_{method}'])
            train['smoke_job_wall_seconds']+=sum(attempt['wall_seconds']
                for tag in [f'v2_smoke_{method}',f'v2_smoke_resume_{method}']
                for attempt in state.get('job_attempts',{}).get(tag+state.get('training_generation',''),[]))
            assert train['smoke_job_wall_seconds']<=900, 'Smoke exceeded the 15-minute cap'
            snapshot(method,'0.18')
            train['smoke_collectors_complete']=True;write(STATE,state)
    def smoke():
        if a.released:
            smoke_preconditions()
            assert evaluation_level()==1
            prior=read('metrics/lwy_v2_queue.json')
            for method in ['dreamer','ppo']:
                evidence=prior['training'][method+'_lwy']
                assert evidence['smoke_collectors_complete'] and evidence['smoke_job_wall_seconds']<=900
                assert evidence['resume_verified']['after']['updates']>evidence['resume_verified']['before']['updates']
                initialize(method)
                train=state['training'][method+'_lwy']
                train['smoke_evidence']='metrics/lwy_v2_queue.json'
                train['run_name']=run_name(method)
                validate(method,'0.00')
            return
        smoke_preconditions()
        if Path('metrics/lwy_v2_training_numerics_diagnosis.json').exists():
            assert Path('metrics/lwy_v2_evaluation_runtime_decision.json').exists(), 'Resolve evaluation runtime before validation or formal training'
        if evaluation_level()==1:parallel_check()
        else:run('v2_video_protocol_probe',['nav3.lwy_video_v2','--protocol-probe'],timeout=600)
        smoke_collectors()
        for method in ['dreamer','ppo']:
            train=state['training'][method+'_lwy']
            if train['smoke_complete']:continue
            for label in ['0.00','0.10','0.18']:validate(method,label)
            train['smoke_complete']=True;write(STATE,state)
    def training():
        smoke()
        for method in ['dreamer','ppo']:
            train=state['training'][method+'_lwy']
            for i in range(1,9):
                if train['stopped']:break
                label=f'{i*.5:.2f}'
                if any(r['label']==label for r in train['rounds']):continue
                train_job(method,f'v2_train_{method}_{label}h',1800)
                validate(method,label)
        write(f'metrics/{artifact_prefix}_model_selection.json',dict(rule='validation success rate, then mean gates; exact ties choose earlier snapshot',
            selected={name:x['best'] for name,x in state['training'].items()},training=state['training'],xla_flags=evaluation_flags()))
    def final():
        selected=read(f'metrics/{artifact_prefix}_model_selection.json')['selected']
        if not a.released:parallel_check()
        pending=[]
        for method in ['dreamer_lwy','ppo_lwy','dreamer_s0','ppo_s0','planner']:
            for dataset,episodes in [('original',100),('blind',5),('heldout',5),('mini',30)]:
                # Pair cases with similar episode duration, especially both
                # filtered cases, so a quick failure does not leave a long run alone.
                for filtered in [0,1]:
                    for obstacles in [0,1]:
                        tag=f'{artifact_prefix.removeprefix("lwy_")}_final_{method}_{dataset}_ob{obstacles}_sf{filtered}{evaluation_suffix()}'
                        existing=Path(f'metrics/lwy_{tag}.json')
                        if existing.exists() and tag not in state['jobs']:
                            result=read(existing)
                            assert result['runtime']['policy_batch_width']==100
                            assert result['runtime']['xla_flags']==evaluation_flags()
                            assert result['episodes']==(30 if dataset=='mini' else 100)
                            assert result['method']==method and result['dataset']==dataset
                            state['jobs'][tag]=dict(returncode=0,previously_completed=True,artifact=str(existing))
                            write(STATE,state)
                        args=['nav3.lwy_compare_v2','--method',method,'--dataset',dataset,'--episodes',str(episodes),'--tag',tag]
                        if method in selected:args+=['--checkpoint',selected[method]['checkpoint']]
                        if obstacles:args+=['--random-obstacles']
                        if filtered:args+=['--filter']
                        pending.append((tag,args))
                        if len(pending)==2:run_pair(pending);pending=[]
        if pending:run_pair(pending)
        run(artifact_prefix.removeprefix('lwy_')+'_inference_video'+evaluation_suffix(),['nav3.lwy_video_v2','--selection',f'metrics/{artifact_prefix}_model_selection.json','--result-prefix',artifact_prefix],timeout=3000)
        state['status']='evaluations_complete';write(STATE,state)
        report()
    try:
        if a.stage in ['profiles','all']:profiles()
        if a.stage=='parallel-check':parallel_check()
        if a.stage=='runtime-candidate':runtime_candidate()
        if a.stage=='smoke-collectors':smoke_collectors()
        if a.stage in ['smoke','all']:smoke()
        if a.stage in ['train','all']:training()
        if a.stage in ['final','all']:final()
        state.update(status='complete' if a.stage in ['all','final'] else 'stage_complete',active_job=None)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat();write(STATE,state)
        report()


if __name__=='__main__':main()
