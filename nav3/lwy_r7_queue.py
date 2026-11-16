"""Detached, resumable R7 queue with gate, GPU ledger and bounded jobs."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

STATE=Path('metrics/lwy_r7_queue.json')
PYTHON=os.environ.get('NAV3_PYTHON',sys.executable)


def save(data):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(STATE)


def run(state,label,args,gpu=True,limit=10200):
    old=next((r for r in state['jobs'] if r['label']==label and r.get('returncode')==0),None)
    if old:return
    used=state.get('gate_gpu_seconds_upper_bound',0)+sum(r.get('seconds',0) for r in state['jobs'] if r.get('gpu'))
    assert not gpu or used+limit<=36000,f'GPU budget reserve insufficient: {used}+{limit}'
    log=Path('logs')/f'lwy_r7_{label}.log'
    assert not log.exists(),'Failed jobs require explicit diagnosis and a new label'
    row=dict(label=label,args=args,gpu=gpu,started=time.time(),log=str(log))
    state['active']=label;state['jobs'].append(row);save(state)
    with log.open('x') as f:
        process=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row['pid']=process.pid;save(state)
        try:code=process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    row.update(returncode=code,seconds=time.time()-row['started']);state['active']=None;save(state)
    assert code==0,f'{label} failed ({code}); see {log}'


def evaluate(state,label,method,checkpoint,dataset='val',episodes=4,ob=False,sf=False,margin=0,limit=900):
    output=Path(f'metrics/lwy_r7_{label}.json')
    args=[PYTHON,'-m','nav3.lwy_compare_r7','--method',method,'--dataset',dataset,'--episodes',str(episodes),'--tag','r7_'+label]
    if checkpoint:args+=['--checkpoint',str(checkpoint)]
    if ob:args+=['--random-obstacles']
    if sf:args+=['--filter']
    if margin:args+=['--support-margin',str(margin)]
    run(state,label,args,limit=limit)
    return json.loads(output.read_text())


def score(x):return (x['success_rate'],x['mean_gates'])


def gate():
    original=json.loads(Path('metrics/lwy_r7_gate_original.json').read_text())
    blind=json.loads(Path('metrics/lwy_r7_gate_blind.json').read_text())
    result=dict(passed=original['successes']>=9 and blind['success_rate']>=.85,
        original_successes=original['successes'],original_episodes=original['episodes'],
        blind_successes=blind['successes'],blind_episodes=blind['episodes'],
        obstacle_rule='shoulder_intrusion_004_v1')
    p=Path('metrics/lwy_r7_gate.json')
    if not p.exists():p.write_text(json.dumps(result,indent=2))
    else:assert json.loads(p.read_text())==result
    assert result['passed'],f'R7 gate failed; training forbidden: {result}'


def prep(state):
    gate()
    run(state,'catalog_train_val',[sys.executable,'-m','nav3.lwy_obstacles_r7','--splits','train','val'],gpu=False)
    assert json.loads(Path('metrics/lwy_r7_random_feasibility_200.json').read_text())['accepted_unpassable']==0
    run(state,'memory_dreamer_2048',[PYTHON,'-m','nav3.lwy_memory_r7','--method','dreamer','--envs','2048',
        '--output','metrics/lwy_r7_memory_dreamer_2048.json','--live-output','metrics/lwy_r7_memory_live.json'],limit=900)
    m=json.loads(Path('metrics/lwy_r7_memory_dreamer_2048.json').read_text())
    assert m['completed'] and m['finite_loss']
    baseline=m['samples'][0]['used_mib'];total=m['samples'][0]['total_mib']
    state['memory']=dict(baseline_mib=baseline,peak_mib=m['peak_used_mib'],available_fraction=(m['peak_used_mib']-baseline)/(total-baseline))
    assert state['memory']['available_fraction']<=.90,'Reduce collectors after this measured memory failure'
    state['envs']=2048;save(state)


def diagnosis(state):
    baseline='checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl'
    initial=evaluate(state,'val_initial','dreamer_lwy',baseline)
    state['initial']=dict(checkpoint=baseline,**{k:initial[k] for k in ['success_rate','mean_gates']});save(state)
    candidates=[];skipped=[]
    def used():
        return sum(r.get('seconds',0) for r in state['jobs'] if r.get('gpu') and
            (r['label']=='val_initial' or r['label'].startswith('diag_') or r['label'].startswith('val_diag_')))
    def done(label):return any(r['label']==label and r.get('returncode')==0 for r in state['jobs'])
    for config in ['baseline','low_lr','uniform']:
        name='dreamer_lwy_r7_diag_'+config
        if not done('diag_'+config):
            validations=[r.get('seconds',0) for r in state['jobs'] if r.get('returncode')==0 and
                (r['label']=='val_initial' or r['label'].startswith('val_diag_'))]
            reserve=max(300,max(validations,default=300)*1.1+30)
            train_limit=min(1150,int(3500-used()-reserve))
            if train_limit<900:
                skipped.append(dict(config=config,reason='One-hour diagnosis cap; insufficient time for the same 720-second arm and validation'))
                continue
            run(state,'clone_'+config,[sys.executable,'scripts/lwy_r7_clone.py',name],gpu=False)
            run(state,'diag_'+config,[PYTHON,'-m','nav3.lwy_train_r7','--name',name,'--envs',str(state['envs']),
                '--seconds','720','--config',f'configs/lwy_r7_{config}.json'],limit=train_limit)
        remaining=int(3500-used())
        assert done('val_diag_'+config) or remaining>=180,'Diagnosis GPU cap before validation'
        result=evaluate(state,'val_diag_'+config,'dreamer_lwy',f'checkpoints/{name}/latest.pkl',limit=max(1,min(900,remaining)))
        candidates.append(dict(config=config,**{k:result[k] for k in ['success_rate','mean_gates']}))
    winner=max(candidates,key=score)
    state['diagnosis']=dict(arms=candidates,skipped=skipped,gpu_seconds=used(),selected=winner['config'],note='Single 12-minute interventions; no unique root cause asserted from one noisy validation sample.')
    save(state)


def failures(state):
    for dataset,episodes in [('original',100),('blind',5)]:
        result=evaluate(state,f'final_dreamer_lwy_{dataset}_ob0_sf0','dreamer_lwy',
            'checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl',dataset,episodes)
        old=json.loads(Path(f'metrics/lwy_runtime2_final_dreamer_lwy_{dataset}_ob0_sf0_rt1.json').read_text())
        assert [r['action_pose_sha256'] for r in result['rows']]==[r['action_pose_sha256'] for r in old['rows']], 'R7 ob0 replay must exactly match the unchanged old protocol'


def train(state):
    config=state['diagnosis']['selected'];name='dreamer_lwy_r7'
    run(state,'clone_main',[sys.executable,'scripts/lwy_r7_clone.py',name],gpu=False)
    rounds=state.setdefault('rounds',[])
    if not rounds:
        zero=Path('checkpoints/snapshots/dreamer_lwy_r7_0.00h');assert not zero.exists();zero.mkdir()
        shutil.copy2(state['initial']['checkpoint'],zero/'latest.pkl')
        rounds.append(dict(hours=0.,**{**state['initial'],'checkpoint':str(zero/'latest.pkl')}))
    save(state)
    for i in range(1,11):
        if any(r['hours']==i*.5 for r in rounds):continue
        if state.get('early_stop'):break
        label=f'train_{i:02d}'
        run(state,label,[PYTHON,'-m','nav3.lwy_train_r7','--name',name,'--envs',str(state['envs']),
            '--seconds','1800','--config',f'configs/lwy_r7_{config}.json'],limit=2100)
        snapshot=Path(f'checkpoints/snapshots/{name}_{i*.5:.2f}h')
        assert not snapshot.exists();snapshot.mkdir();shutil.copy2(Path('checkpoints')/name/'latest.pkl',snapshot/'latest.pkl')
        result=evaluate(state,f'val_main_{i:02d}','dreamer_lwy_r7',snapshot/'latest.pkl')
        rounds.append(dict(hours=i*.5,checkpoint=str(snapshot/'latest.pkl'),**{k:result[k] for k in ['success_rate','mean_gates']}))
        best=max(rounds,key=score)
        if len(rounds)>=3:
            if all(r['success_rate']<best['success_rate']-.30 for r in rounds[-2:]):state['early_stop']='two validations >30 percentage points below best'
            elif rounds[-1]['success_rate']<rounds[-2]['success_rate']<rounds[-3]['success_rate']:state['early_stop']='two successive validation decreases; stop and diagnose'
        state['best']=best;save(state)
        run(state,f'report_main_{i:02d}',['.venv_lwy/bin/python','-m','nav3.lwy_r7_report'],gpu=False,limit=180)
    state['best']=max(rounds,key=score);save(state)



def ppo(state):
    name='ppo_lwy_r7';source=Path('checkpoints/snapshots/ppo_lwy_runtime2_3.50h/latest.pkl')
    run(state,'memory_ppo_128',[PYTHON,'-m','nav3.lwy_memory_r7','--method','ppo','--envs','128',
        '--output','metrics/lwy_r7_memory_ppo_128.json','--live-output','metrics/lwy_r7_memory_ppo_live.json'],limit=600)
    measured=json.loads(Path('metrics/lwy_r7_memory_ppo_128.json').read_text())
    assert measured['completed'] and measured['finite_loss']
    base=measured['samples'][0]
    state['ppo_memory']=dict(baseline_mib=base['used_mib'],peak_mib=measured['peak_used_mib'],
        available_fraction=(measured['peak_used_mib']-base['used_mib'])/(base['total_mib']-base['used_mib']))
    save(state)
    assert state['ppo_memory']['available_fraction']<=.90,'PPO memory check exceeds available-memory allowance'
    initial=evaluate(state,'val_ppo_initial','ppo_lwy',source)
    rounds=state.setdefault('ppo_rounds',[dict(hours=0.,checkpoint=str(source),**{k:initial[k] for k in ['success_rate','mean_gates']})])
    out=Path('checkpoints')/name
    if not out.exists():out.mkdir();shutil.copy2(source,out/'latest.pkl')
    for i in range(1,3):
        if len(rounds)>i:continue
        used=sum(r.get('seconds',0) for r in state['jobs'] if r.get('gpu') and ('ppo' in r['label']))
        train_limit=min(1550,int(3600-used-900))
        assert train_limit>=300,'PPO one-hour GPU cap'
        train_seconds=min(1400,train_limit-150)
        run(state,f'ppo_train_{i}',[PYTHON,'-m','nav3.lwy_train_ppo_r7','--lwy-round','--resume','--name',name,
            '--envs','128','--steps','1000000000000','--seconds',str(train_seconds)],limit=train_limit)
        snapshot=Path(f'checkpoints/snapshots/{name}_{i:02d}')
        assert not snapshot.exists();snapshot.mkdir();shutil.copy2(out/'latest.pkl',snapshot/'latest.pkl')
        result=evaluate(state,f'val_ppo_{i}','ppo_lwy',snapshot/'latest.pkl')
        rounds.append(dict(hours=sum(r.get('seconds',0) for r in state['jobs'] if r['label'].startswith('ppo_train_'))/3600,checkpoint=str(snapshot/'latest.pkl'),**{k:result[k] for k in ['success_rate','mean_gates']}));save(state)
    state['ppo_best']=max(rounds,key=score)
    state['ppo_stopped']='single low-cost check exhausted; no additional budget'
    save(state)


def final(state):
    methods={'dreamer_lwy_r7':state['best']['checkpoint'],
        'dreamer_lwy':'checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl',
        'ppo_lwy':state['ppo_best']['checkpoint'],'planner':None}
    for method,checkpoint in methods.items():
        for dataset,episodes in [('original',100),('blind',5),('heldout',5),('mini',30)]:
            for ob in [False,True]:
                for sf in [False,True]:
                    evaluate(state,f'final_{method}_{dataset}_ob{int(ob)}_sf{int(sf)}',method,checkpoint,dataset,episodes,ob,sf)
    for dataset,episodes in [('original',100),('blind',5)]:
        for ob in [False,True]:
            evaluate(state,f'support_{dataset}_ob{int(ob)}','dreamer_lwy_r7',state['best']['checkpoint'],dataset,episodes,ob,True,.08)
    state['status']='evaluation_complete';save(state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prep','diagnosis','failures','train','ppo','final'],required=True);a=p.parse_args()
    import fcntl
    lock=Path('logs/lwy_r7_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs=[],status='running')
    if 'gate_gpu_seconds_upper_bound' not in state:
        begin=Path('metrics/lwy_r7_obstacle_catalog_test_blind.json')
        end=Path('metrics/lwy_r7_gate_blind.json')
        if begin.exists() and end.exists():
            state['gate_gpu_seconds_upper_bound']=max(0.,end.stat().st_mtime-begin.stat().st_mtime)
            state['gate_gpu_accounting']='Wall interval from CPU catalog completion to final gate metric, including both planner GPU process startups.'
    state['status']='running_'+a.stage;state.pop('error',None);save(state)
    try:
        {'prep':prep,'diagnosis':diagnosis,'failures':failures,'train':train,'ppo':ppo,'final':final}[a.stage](state)
        state['status']=a.stage+'_complete';save(state)
    except Exception as e:
        state['status']='stopped';state['error']=repr(e);save(state);raise


if __name__=='__main__':main()
