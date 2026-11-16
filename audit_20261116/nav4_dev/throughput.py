"""C5: env steps/s with a uniform random policy; peak device memory of this process."""
import json
import os
import subprocess
import sys
import threading
import time
import jax
import jax.numpy as jnp
from nav4.bank import TrailBank
from nav4.env import TrailEnv

manifest, n, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = sys.argv[4:] or None
peak_smi = [0]


def sample():
    while True:
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                           '--format=csv,noheader,nounits'], text=True)
            for line in out.splitlines():
                pid, mem = line.split(',')
                if int(pid) == os.getpid():
                    peak_smi[0] = max(peak_smi[0], int(mem))
        except Exception:
            pass
        time.sleep(.2)


threading.Thread(target=sample, daemon=True).start()
bank = TrailBank(manifest, names)
env = TrailEnv(bank, n)
key = jax.random.PRNGKey(0)
act = jax.jit(lambda k: jax.random.uniform(k, (n, 3), minval=-1, maxval=1))
t0 = time.time()
state, obs, _ = env.reset(key)
jax.block_until_ready(obs['bev'])
for i in range(3):
    state, obs, _ = env.step(state, act(jax.random.fold_in(key, i)))
jax.block_until_ready(obs['bev'])
compile_s = time.time()-t0
t = time.time()
for i in range(steps):
    state, obs, _ = env.step(state, act(jax.random.fold_in(key, 100+i)))
jax.block_until_ready(obs['bev'])
dt = time.time()-t
stats = jax.devices()[0].memory_stats() or {}
time.sleep(.5)
print('THROUGHPUT', json.dumps(dict(manifest=manifest, maps=bank.names, envs=n, steps=steps, seconds=dt,
      env_steps_per_s=n*steps/dt, policy_steps_per_s_per_env=steps/dt, compile_and_warmup_s=compile_s,
      jax_peak_bytes_in_use=stats.get('peak_bytes_in_use'), nvidia_smi_peak_mib=peak_smi[0],
      bank_bytes=int(sum(getattr(bank, k).nbytes for k in bank.KEYS)), device=str(jax.devices()[0]),
      shapes={k: list(v.shape) for k, v in obs.items()})), flush=True)
