"""Exercise mixed collectors with real maps, catalog placements and episode resets."""
import json
from pathlib import Path
import time
from .lwy_eval_v2 import setup


def main():
    setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .lwy_bank import LwyBank
    from .bank import MapBank
    from .lwy_mixed import MixedEnv,EpisodeMixMonitor
    view=Path('maps_lwy_v2_sampler_probe');view.mkdir(exist_ok=True)
    choices=sorted(Path('maps_lwy_v2_obstacles/lwy').glob('train_*/audit.json'))[:2]
    assert len(choices)==2
    records=[]
    for p in choices:
        d=Path('maps_lwy_v2')/p.parent.name
        link=view/d.name
        if not link.exists():link.symlink_to(d.resolve(),target_is_directory=True)
        records.append(json.loads((d/'metadata.json').read_text()))
    (view/'manifest.json').write_text(json.dumps(dict(maps=records)))
    # Use the existing full old cache; never create a protected limited cache.
    old=MapBank('train');new=LwyBank('train',root=str(view))
    # The partial catalog producer may still be working on legacy maps.
    assert all((Path('maps_lwy_v2_obstacles/legacy')/r['name']/'audit.json').exists() for r in old.records)
    env=MixedEnv(new,old,64,tracker_config={'yaw_tracking_gain':4.})
    monitor=EpisodeMixMonitor(new.count);begin=time.monotonic()
    state,obs,_=env.reset(jax.random.PRNGKey(610990))
    for _ in range(120):
        monitor.observe(state,obs)
        state={**state,'done':jnp.ones(64,bool)}
        state,obs,_=env.step(state,jnp.zeros((64,3)))
    assert .72<monitor.save()['lwy_fraction']<.78
    for _ in range(100):
        monitor.observe(state,obs)
        state,obs,_=env.step(state,jnp.zeros((64,3)))
        assert np.isfinite(np.asarray(state['pos'])).all()
    result=dict(passed=True,episode_mix=monitor.save(),forced_reset_batches=120,natural_step_batches=100,
        count=64,wall_seconds=time.monotonic()-begin,random_obstacles=True,unchanged_legacy_checker=True)
    Path('metrics/lwy_v2_sampler_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='episode_mix'}),flush=True)


if __name__=='__main__':main()
