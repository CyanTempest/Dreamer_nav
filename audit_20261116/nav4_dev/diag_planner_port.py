"""Port check: legacy nav3.lwy_control_v2.planning_result vs nav4 Planner on identical nav4 observations
(holdout maps, nav4 planner drives). Differences expected only from the 20 Hz shared tracker rollout."""
import json
import sys
import jax
import jax.numpy as jnp
import numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.planner import Planner
from nav3.lwy_control_v2 import planning_result

params = json.load(open('configs/lwy_v2_planner.json'))
bank = TrailBank('r9_official_runner/maps_scan_holdout')
env = TrailEnv(bank, bank.count*2)
planner = Planner(env.cfg)
legacy = jax.jit(lambda o: planning_result(o, **params))
state, obs, _ = env.reset(jax.random.PRNGKey(914901), start_s=2.)
same_offset, gap, speed_same, risky, n = 0, [], 0, 0, 0
for t in range(int(sys.argv[1]) if len(sys.argv) > 1 else 250):
    a, info = planner.plan(obs)
    b, linfo = legacy(obs)
    live = ~np.asarray(obs['is_last'])
    same_offset += int(np.sum((np.abs(np.asarray(info['offset'])-np.asarray(linfo['offset'])) < 1e-4) & live))
    speed_same += int(np.sum((np.abs(np.asarray(a[:, 2])-np.asarray(b[:, 2])) < 1e-4) & live))
    risky += int(np.sum((np.asarray(linfo['min_risk']) > 0) & live))
    gap.append(np.abs(np.asarray(a)-np.asarray(b))[live])
    n += int(live.sum())
    state, obs, _ = env.step(state, a)
gap = np.concatenate(gap)
print(json.dumps(dict(decisions=n, same_offset=same_offset/n, same_speed=speed_same/n,
                      legacy_min_risk_positive=risky/n, mean_abs_action_gap=gap.mean(0).tolist(),
                      p99_abs_action_gap=np.percentile(gap, 99, 0).tolist())))
