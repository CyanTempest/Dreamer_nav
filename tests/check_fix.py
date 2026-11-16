"""Round-five isolation, frozen physics, repeatability and reward regression."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--reward',action='store_true');a=p.parse_args()
    from nav3.provenance import fingerprint
    assert fingerprint()=='a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d'
    protected=json.loads(Path('metrics/fix_protected_hashes.json').read_text())
    # Large map/checkpoint trees are additionally streamed by fix_protect.py.
    for name,h in protected.items():
        if name.startswith(('metrics/','nav3/','configs/')):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    for name in ['dreamer_s0','ppo_s0']:
        x=json.loads(Path(f'metrics/fix_repeat_v1_{name}_a.json').read_text())
        y=json.loads(Path(f'metrics/fix_repeat_v1_{name}_b.json').read_text())
        assert x['rows']==y['rows'] and x['protocol']==y['protocol'] and x['checkpoint_sha256']==y['checkpoint_sha256']
    for name in ['dreamer_s0','ppo_s0','dreamer_s0_ft','ppo_s0_ft']:
        p=Path(f'metrics/fix_repeat100_{name}.json')
        if p.exists():
            x=json.loads(p.read_text());y=json.loads(Path(f'metrics/fix_eval_v1_{name}_test_ob0_sf0.json').read_text())
            assert x['rows']==y['rows'] and x['checkpoint_sha256']==y['checkpoint_sha256']
    if Path('maps_v2/manifest.json').exists():
        old=json.loads(Path('maps/manifest.json').read_text())['maps']
        new=json.loads(Path('maps_v2/manifest.json').read_text())['maps']
        train={r['seed'] for r in new if r['split']=='train'}
        blind={r['seed'] for r in new if r['split']=='blind'}
        forbidden={r['seed'] for r in old}|set(range(30000,30004))|{871002}
        assert len(train)>=200 and len(blind)==20 and not train&blind and not (train|blind)&forbidden
        assert all(not r['mini'] and r['bypass_validation']['connected'] for r in new)
        if Path('maps_mix_v2/manifest.json').exists():
            mixed=json.loads(Path('maps_mix_v2/manifest.json').read_text())['maps']
            assert {r['seed'] for r in mixed}==train|{r['seed'] for r in old if r['split']=='train'}
            assert all(r['split']=='train' and not r['mini'] for r in mixed)
            allowed={r['seed'] for r in mixed}
            for name in ['dreamer_s0_ft','ppo_s0_ft']:
                log=Path(f'metrics/trainmaps_{name}.jsonl')
                if log.exists():
                    with log.open() as f:
                        for line in f:
                            r=json.loads(line)
                            assert r['seed'] in allowed and r['split']=='train' and not r['mini'],r
            configs=[Path(f'metrics/config_{name}.json') for name in ['dreamer_s0_ft','ppo_s0_ft']]
            if all(p.exists() for p in configs):
                x,y=[json.loads(p.read_text()) for p in configs]
                for k in ['map_root','map_names','map_seeds','margin_weight','margin_distance']:
                    assert x[k]==y[k],k
    if a.reward:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from nav3.bank import MapBank
        from nav3.env import FastEnv
        from nav3.fix_training import training_env
        bank=MapBank('test');old=FastEnv(bank,2,True);new=training_env(bank,2,weight=.25)
        state,_,_=old.reset(jax.random.PRNGKey(991),jnp.zeros(2,jnp.int32),jnp.array([163.2,165.4]))
        # Exercise actual production stepping, including first/reset markers.
        active_penalty=False
        for _ in range(10):
            aa=jnp.zeros((2,3));x,ox,ex=old.step(state,aa);y,oy,ey=new.step(state,aa)
            for key in x:
                if key not in ['reward','ep_return']:assert np.array_equal(x[key],y[key]),key
            assert np.all(np.asarray(y['reward'])<=np.asarray(x['reward']))
            active_penalty|=bool(np.any(np.asarray(y['reward'])<np.asarray(x['reward'])))
            state=x
        assert active_penalty,'Regression must exercise an actual obstacle-margin penalty'
    print(json.dumps(dict(passed=True,old_fingerprint_unchanged=True,exact_repeats=True,map_isolation=True,reward_checked=a.reward)))


if __name__=='__main__':main()
