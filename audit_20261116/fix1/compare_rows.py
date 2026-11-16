"""Compare per-episode rows (and, where present, .npz dumps) of two diagnose outputs exactly."""
import json,sys
from pathlib import Path
import numpy as np
KEYS=['gates','failure','s','ep_return','steps','valid_start','map']

def load(p):return json.loads(Path(p).read_text())

def compare(new,ref,label):
    a,b=load(new),load(ref);ra,rb=a['rows'],b['rows']
    report=dict(label=label,new=new,ref=ref,episodes_new=len(ra),episodes_ref=len(rb),same_episode_ids=sorted(ra)==sorted(rb))
    core=[i for i in rb if any(ra.get(i,{}).get(k,'<missing>')!=rb[i].get(k,'<missing>') for k in KEYS+['truncated'])]
    common=[i for i in rb if i in ra and {k:ra[i][k] for k in set(ra[i])&set(rb[i])}!={k:rb[i][k] for k in set(ra[i])&set(rb[i])}]
    report.update(core_key_mismatches=core,all_common_key_mismatches=common,
        keys_only_in_new=sorted({k for r in ra.values() for k in r}-{k for r in rb.values() for k in r}),
        keys_only_in_ref=sorted({k for r in rb.values() for k in r}-{k for r in ra.values() for k in r}))
    v=lambda r:[x for x in r.values() if x['valid_start']]
    ok=lambda r:sum(1 for x in v(r) if x.get('gates')==8 and x.get('failure')==0)
    report.update(successes_new=f'{ok(ra)}/{len(v(ra))}',successes_ref=f'{ok(rb)}/{len(v(rb))}')
    for k in ['filter_intervention_fraction','valid_starts','episodes','action']:
        if k in a or k in b:report[k]=dict(new=a.get(k,'<missing>'),ref=b.get(k,'<missing>'))
    for suffix in ['.npz','_rollout.npz']:
        pn=Path(new).with_name(Path(new).stem+suffix) if suffix!='.npz' else Path(new).with_suffix('.npz')
        pr=Path(ref).with_name(Path(ref).stem+suffix) if suffix!='.npz' else Path(ref).with_suffix('.npz')
        if pn.exists() and pr.exists():
            with np.load(pn) as x, np.load(pr) as y:
                report['npz'+suffix]=all(k in y and x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in x.files) and sorted(x.files)==sorted(y.files)
    report['PASS']=report['same_episode_ids'] and not core
    print(json.dumps(report));return report

if __name__=='__main__':
    out=sys.argv[1];pairs=sys.argv[2:]
    results=[compare(pairs[i],pairs[i+1],pairs[i+2]) for i in range(0,len(pairs),3)]
    Path(out).write_text(json.dumps(results,indent=2)+'\n')
