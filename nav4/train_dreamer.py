"""DreamerV3 training of the localisation-free nav4 policy with the OFFICIAL loop embodied.run.train.train.

  python -m nav4.train_dreamer --logdir checkpoints/<name> --seconds 14400 [--config nav4/config/train.json]
      [--envs 256] [--resume] [--allow-cpu] [--set KEY=JSON ...] [--env KEY=JSON ...]

dreamerv3/embodied/run/train.py runs unmodified. As in r8_p2_runner/r8_p3/train.py, only these objects are
injected: a device-batched Driver over TrailEnv (one batched env step per driver call), a Replay wrapper that
reports no items until every slot has filled one sequence, the official elements.Checkpoint extended with the
driver and runtime state, an agent wrapper (batched policy, update counter) and a JSONL Logger.

--logdir gets config.json, metrics.jsonl, ckpt/ (official checkpoints), replay/ (chunks) and latest.pkl
(agent + payload['nav4']: env overrides and observation keys, read by nav4.evaluate); best.pkl (same format)
and best.json hold the weights of the best periodic evaluation (fewest assists/100 m).
--seconds: training wall time counted from the first optimizer update, summed over resumes.
--resume continues logdir from its latest official checkpoint with its saved config.
GPU guard: refuses to start (exit 3) unless nvidia-smi reports min_free_gpu_mib available (free + this process's
own use); checked again before every checkpoint (if it fails: save, then stop with exit 3). XLA may allocate at
most min_free_gpu_mib unless xla_memory_fraction sets a separate allocator limit. --allow-cpu skips the guard and forces JAX_PLATFORMS=cpu (smoke tests only).
Seeds: env resets PRNGKey(seed); agent init, train/policy seeds and replay sampler from seed; evaluation resets
from eval.seed with the nav4.evaluate policy key. Thread timing of the replay prefetch makes runs not bit-exact.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXIT_GPU = 3
DETERMINISTIC_XLA = {'--xla_gpu_autotune_level': '1', '--xla_gpu_deterministic_ops': 'true'}


class SegmentComplete(Exception):
    pass


def overrides(items):
    return {k: json.loads(v) for k, v in (item.split('=', 1) for item in items or [])}


def set_key(cfg, key, value):
    """Dotted override of an existing train config key."""
    *path, last = key.split('.')
    node = cfg
    for part in path:
        node = node[part]
    if last not in node:
        raise KeyError(f'unknown train config key {key}')
    node[last] = value


def nvidia_smi(args):
    return subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=60, check=True).stdout


def gpu_memory(query=None, pid=None):
    """GPU 0 memory in MiB: available = free + this process's own use (what other users leave us)."""
    query, pid = query or nvidia_smi, str(os.getpid() if pid is None else pid)
    try:
        free, total = (int(x) for x in query(['--query-gpu=memory.free,memory.total',
                                              '--format=csv,noheader,nounits']).splitlines()[0].split(','))
        own = 0
        for line in query(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']).splitlines():
            fields = [x.strip() for x in line.split(',')]
            if len(fields) == 2 and fields[0] == pid and fields[1].isdigit():
                own += int(fields[1])
        return dict(available_mib=free+own, free_mib=free, own_mib=own, total_mib=total)
    except Exception as error:      # no driver, no GPU, unparsable output: treat as unavailable
        return dict(error=repr(error))


def guard_ok(info, min_free):
    return 'error' not in info and info['available_mib'] >= min_free


def deterministic_xla(flags):
    """XLA_FLAGS with autotune level 1 and deterministic ops (autotune 0 corrupts Dreamer numerics here)."""
    kept = [f for f in flags.split() if f.split('=', 1)[0] not in DETERMINISTIC_XLA]
    return ' '.join(kept+[f'{k}={v}' for k, v in DETERMINISTIC_XLA.items()])


def clean(x):
    """JSON-safe: numpy scalars to float/int, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, 'item') and getattr(x, 'ndim', 1) == 0:
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_atomic(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BatchedReplayWriter:
    """Write one synchronous driver tick to uncompressed upstream replay chunks.

    Per-worker Chunk objects keep their native UUIDs, successor links and save
    format. Their arrays are views of a shared (worker, time, ...) allocation.
    A checkpoint closes partial chunks, so the next add opens a new slab.
    """

    def __init__(self, replay):
        if replay.compress or replay.mmap_directory:
            raise ValueError('batched insertion requires uncompressed in-memory replay')
        self.replay = replay
        self.arrays = {}
        self.chunks = []
        self.index = 0

    def _begin(self, batch):
        import numpy as np
        from embodied.core.chunk import Chunk
        replay = self.replay
        count = len(next(iter(batch.values())))
        chunks = []
        for worker in range(count):
            if worker not in replay.current:
                chunk = Chunk(replay.chunksize)
                replay.chunks[chunk.uuid] = chunk
                replay.refs[chunk.uuid] = 1
                replay.current[worker] = (chunk.uuid, 0)
            uuid, index = replay.current[worker]
            chunk = replay.chunks[uuid]
            if index or chunk.length:
                raise ValueError('expected aligned empty chunks after checkpoint restore')
            chunks.append(chunk)
        self.arrays = {k: np.empty((count, replay.chunksize, *v.shape[1:]), v.dtype)
                       for k, v in batch.items()}
        ids = np.empty((count, replay.chunksize, 20), np.uint8)
        ids[:, :, 16:] = np.arange(replay.chunksize, dtype='>u4').view(np.uint8).reshape(-1, 4)
        for worker, chunk in enumerate(chunks):
            ids[worker, :, :16] = np.frombuffer(bytes(chunk.uuid), np.uint8)
            chunk.data = {k: v[worker] for k, v in self.arrays.items()}
            chunk.data['stepid'] = ids[worker]
        self.chunks = chunks
        self.index = 0

    def add(self, batch):
        replay = self.replay
        batch = {k: v for k, v in batch.items() if not k.startswith('log/')}
        with replay.rwlock.reading, replay.refs_lock:
            if not self.chunks or replay.current[0][0] != self.chunks[0].uuid:
                self._begin(batch)
            index = self.index
            for key, value in batch.items():
                self.arrays[key][:, index] = value
            for worker, chunk in enumerate(self.chunks):
                chunk.length = index + 1
                pair = (chunk.uuid, index)
                stream = replay.streams[worker]
                stream.append(pair)
                replay.refs[chunk.uuid] += 1
                if chunk.length == chunk.size:
                    replay._complete(chunk, worker)
                else:
                    replay.current[worker] = (chunk.uuid, chunk.length)
                if len(stream) >= replay.length:
                    replay.metrics['inserts'] += 1
                    oldest = stream.popleft()
                    replay._insert(*oldest)
                    if replay.online and replay.lengths[worker] % replay.length == 0:
                        replay.queue.append(oldest)
                if replay.online:
                    replay.lengths[worker] += 1
            self.index += 1


class Episodes:
    """Finished training episodes: count, return, failure codes (0 = arrived), progress at termination."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.returns, self.progress, self.length, self.codes = 0, 0., 0., 0, Counter()

    def add(self, ep):
        for i in ep['done'].nonzero()[0]:
            self.count += 1
            self.returns += float(ep['ep_return'][i])
            self.progress += float(ep['progress'][i])
            self.length += int(ep['steps'][i])
            self.codes[str(int(ep['failure'][i]))] += 1

    def result(self):
        if not self.count:
            return dict(count=0)
        return dict(count=self.count, mean_return=self.returns/self.count,
                    mean_progress_at_termination=self.progress/self.count, mean_length_steps=self.length/self.count,
                    failure_codes=dict(sorted(self.codes.items())))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(ROOT/'nav4/config/train.json'), help='train config (ignored with --resume)')
    p.add_argument('--logdir', required=True)
    p.add_argument('--seconds', type=float, required=True, help='training budget (s) from the first update, all resumes')
    p.add_argument('--envs', type=int, help='parallel env slots (default: config envs)')
    p.add_argument('--resume', action='store_true', help='continue --logdir from its latest official checkpoint')
    p.add_argument('--init-checkpoint', help='start a fresh replay and run clock from matching nav4 agent weights')
    p.add_argument('--allow-cpu', action='store_true', help='skip the GPU guard and run on CPU (smoke tests only)')
    p.add_argument('--set', nargs='*', metavar='KEY=JSON', help='train config overrides, e.g. eval.traverses=1')
    p.add_argument('--env', nargs='*', metavar='KEY=JSON', help='extra env.json overrides on top of config env')
    a = p.parse_args(argv)
    logdir = (Path(a.logdir) if Path(a.logdir).is_absolute() else Path.cwd()/a.logdir).resolve()
    if a.resume:
        if not (logdir/'ckpt'/'latest').exists():
            p.error(f'--resume: no official checkpoint in {logdir}/ckpt')
        if a.set or a.env:
            p.error('--resume continues with the saved config; --set/--env are not accepted')
        cfg = json.loads((logdir/'config.json').read_text())['train']
        if a.envs and a.envs != cfg['envs']:
            p.error(f"--resume: the run uses {cfg['envs']} envs")
    else:
        if (logdir/'ckpt').exists():
            p.error(f'{logdir} already holds a checkpoint: use --resume or a new --logdir')
        cfg = json.loads(Path(a.config).read_text())
        for key, value in overrides(a.set).items():
            set_key(cfg, key, value)
        cfg['env'] = {**cfg['env'], **overrides(a.env)}
        cfg['envs'] = a.envs or cfg['envs']
    if a.resume and a.init_checkpoint:
        p.error('--resume and --init-checkpoint are mutually exclusive')
    warmup_steps = max(cfg['warmup_steps'], cfg['batch_length'] + 1)
    n, seed = cfg['envs'], cfg['seed']
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir/'metrics.jsonl'

    def append(row):
        with log.open('a') as f:
            f.write(json.dumps(clean(row), allow_nan=False)+'\n')

    fast_xla = cfg.get('xla_default_tuning', False) and not a.allow_cpu
    prealloc = cfg.get('xla_preallocate', False) and not a.allow_cpu
    # GPU guard and runtime flags, before JAX initialises.
    if a.allow_cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        gpu = dict(bypassed='--allow-cpu')
    else:
        gpu = gpu_memory()
        if not guard_ok(gpu, cfg['min_free_gpu_mib']):
            append(dict(event='gpu_guard_refused', unix=time.time(), gpu=gpu, min_free_gpu_mib=cfg['min_free_gpu_mib']))
            print(f"GPU guard: {gpu} < min_free_gpu_mib={cfg['min_free_gpu_mib']}; not starting", file=sys.stderr)
            return EXIT_GPU
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(prealloc).lower()
        fraction = cfg.get('xla_memory_fraction')
        if fraction is None:
            fraction = min(1., cfg['min_free_gpu_mib']/gpu['total_mib'])
        if not 0 < fraction <= 1:
            p.error('xla_memory_fraction must be in (0, 1]')
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{fraction:.4f}'
    if fast_xla:
        os.environ['XLA_FLAGS'] = ' '.join(f for f in os.environ.get('XLA_FLAGS', '').split()
                                           if f.split('=', 1)[0] not in DETERMINISTIC_XLA)
        os.environ.pop('TF_CUDNN_DETERMINISTIC', None)
    else:
        os.environ['XLA_FLAGS'] = deterministic_xla(os.environ.get('XLA_FLAGS', ''))
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    runtime = {k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                                              'XLA_PYTHON_CLIENT_MEM_FRACTION', 'TF_CUDNN_DETERMINISTIC')}
    import jax
    import jax.numpy as jnp
    import numpy as np
    import elements
    from .bank import TrailBank, manifest_path
    from .env import TrailEnv
    from . import evaluate
    from .policy import DreamerPolicy, spec
    import embodied
    if not a.allow_cpu and jax.default_backend() != 'gpu':
        append(dict(event='gpu_guard_refused', unix=time.time(), reason=f'JAX backend is {jax.default_backend()}'))
        print('GPU guard: JAX did not start on the GPU (unset JAX_PLATFORMS, or --allow-cpu)', file=sys.stderr)
        return EXIT_GPU

    manifest = Path(cfg['manifest']) if Path(cfg['manifest']).is_absolute() else ROOT/cfg['manifest']

    def map_hashes():
        records = json.loads(manifest_path(manifest).read_text())['maps']
        return {'manifest': sha256(manifest_path(manifest)), **{
            r['name']: sha256(manifest_path(manifest).parent/r.get('path', f"{r['name']}/map.npz"))
            for r in records if cfg['maps'] is None or r['name'] in cfg['maps']}}
    map_sha256 = map_hashes()
    bank = TrailBank(manifest, cfg['maps'])
    if map_hashes() != map_sha256:          # the map files are rewritten in place by the mapping pipeline
        print(f'{manifest}: map files changed while loading; start again', file=sys.stderr)
        return 2
    env = TrailEnv(bank, n, cfg['env'])
    initial = env.reset(jax.random.PRNGKey(seed))
    policy = DreamerPolicy(seed, logdir, obs=initial[1], model=cfg['model'], batch_size=cfg['batch_size'],
                          batch_length=cfg['batch_length'], jax_overrides=dict(
        autotune=4 if fast_xla else 1, deterministic=not fast_xla, prealloc=prealloc, platform='cpu' if a.allow_cpu else 'cuda'))
    if a.init_checkpoint:
        from .policy import read_checkpoint
        payload = read_checkpoint(a.init_checkpoint)
        meta = payload.get('nav4', {})
        if (meta.get('obs_space') != spec(policy.obs_space) or
                meta.get('train_config', {}).get('model', 'size12m') != cfg['model']):
            p.error('initial weights must match the configured model and observation space')
        policy.agent.load(payload['agent'])
        del payload
    decoder_keys = list(policy.agent.model.dec.obs_space)
    e = cfg['eval']
    eval_env = TrailEnv(bank, e['traverses'], cfg['env'])
    process_begin = time.monotonic()
    context = SimpleNamespace(begin=None, elapsed=0., stop=False, exit_code=0, cp=None, ratio=None, driver=None,
                              updates=0, train_carry=None, pending_carry=None, evals=0,
                              next_eval=e['minutes']*60. if e['minutes'] > 0 else math.inf,
                              episodes_total=0, saved_step=None)
    signal.signal(signal.SIGTERM, lambda *_: setattr(context, 'stop', True))
    signal.signal(signal.SIGINT, lambda *_: setattr(context, 'stop', True))
    episodes = Episodes()

    def train_seconds():
        return context.elapsed+(time.monotonic()-context.begin if context.begin is not None else 0.)

    loop = ROOT/'dreamerv3/embodied/run/train.py'
    if not a.resume:
        write_atomic(logdir/'config.json', (json.dumps(clean(dict(
            train=cfg, env_config=env.cfg, obs_space=spec(policy.obs_space),
            encoder_keys=list(policy.agent.model.enc.obs_space), decoder_keys=decoder_keys,
            seeds=dict(env_reset=seed, agent=seed, policy_key=seed+1234, replay=seed, eval_reset=e['seed'],
                       eval_policy_key=1234),
            agent=dict(opt=dict(policy.cfg.agent.opt), loss_scales=dict(policy.cfg.agent.loss_scales),
                       imag_loss=dict(policy.cfg.agent.imag_loss), retnorm=dict(policy.cfg.agent.retnorm)),
            runtime=runtime, jax=jax.__version__, devices=[str(d) for d in jax.devices()],
            algorithm='embodied.run.train.train (unmodified source), injected batched Driver/Replay/Checkpoint/Logger',
            official_loop_sha256=sha256(loop), agent_sha256=sha256(ROOT/'dreamerv3/dreamerv3/agent.py'),
            maps=list(bank.names), manifest=str(manifest), map_sha256=map_sha256)), indent=1)+'\n').encode())
    saved_maps = json.loads((logdir/'config.json').read_text()).get('map_sha256')
    append(dict(event='resume' if a.resume else 'start', unix=time.time(), envs=n, seed=seed, budget_seconds=a.seconds,
                gpu=gpu, runtime=runtime, backend=jax.default_backend(), argv=sys.argv, map_sha256=map_sha256,
                map_changed_since_start=saved_maps != map_sha256))

    profiling = cfg.get('profile_steps', False)
    phase = Counter()
    sample_work = [0.]
    previous_end = None

    replay = embodied.Replay(length=cfg['batch_length']+1, capacity=cfg['replay']['capacity'],
                             directory=str(logdir/'replay'), chunksize=cfg['replay']['chunksize'],
                             online=cfg['replay']['online'], seed=seed, save_wait=True,
                             compress=cfg['replay']['compress'])

    batch_writer = BatchedReplayWriter(replay) if cfg['replay'].get('batched', False) else None

    def check_losses(metrics):
        for key, value in metrics.items():
            if (key.startswith('loss/') or key in ('opt/loss', 'opt/grad_norm')) and not np.isfinite(value).all():
                raise FloatingPointError(f'non-finite Dreamer metric {key}')

    class Agent:
        def __getattr__(self, key):
            return getattr(policy.agent, key)

        def init_policy(self, batch):
            return policy.initial(batch)

        def policy(self, carry, obs, mode='train'):
            policy.key, key = jax.random.split(policy.key)
            _, (carry, act, outs) = policy._policy(policy.agent.params, carry, obs, mode=mode, seed=key)
            action = jnp.where(obs['is_last'][:, None], 0., act['action'])
            carry = (*carry[:3], {'action': action})
            outs = {k: v.astype(policy.agent.spaces[k].dtype) for k, v in outs.items() if k != 'finite'}
            return carry, {'action': action}, outs

        def train(self, carry, batch):
            if context.pending_carry is not None:
                carry, context.pending_carry = context.pending_carry, None
            if profiling:
                start = time.perf_counter()
            carry, outs, metrics = policy.agent.train(carry, batch)
            if profiling:
                jax.block_until_ready((carry, policy.agent.params))
                phase['train_update'] += time.perf_counter() - start
            check_losses(metrics)
            context.train_carry = carry
            context.updates += 1
            if context.begin is None:
                context.begin = time.monotonic()
                append(dict(event='training_clock_started', unix=time.time(), steps=int(logger.step),
                            updates=context.updates, train_seconds_before=context.elapsed))
            return carry, outs, metrics

        def stream(self, source):
            upstream = policy.agent.stream(source)
            if not profiling:
                return upstream
            def timed():
                iterator = iter(upstream)
                while True:
                    start = time.perf_counter()
                    batch = next(iterator)
                    jax.block_until_ready(batch)
                    phase['dataset_wait'] += time.perf_counter() - start
                    yield batch
            return timed()

        def save(self):
            return policy.agent.save()

        def load(self, data):
            policy.agent.load(data)

    class Replay:
        def __len__(self):
            return len(replay) if int(logger.step) >= warmup_steps*n else 0

        def __getattr__(self, key):
            return getattr(replay, key)

        def update(self, data):
            if profiling:
                start = time.perf_counter()
            replay.update(data)
            if profiling:
                phase['replay_update'] += time.perf_counter() - start

        def save(self):
            return replay.save(detailed=True)

        def load(self, data):
            replay.load(data)

    class Logger:
        def __init__(self):
            self.step, self.values = elements.Counter(), {}

        def add(self, values, prefix=None):
            for key, value in values.items():
                arr = np.asarray(value)
                if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number) and np.isfinite(arr):
                    self.values[f'{prefix}/{key}' if prefix else key] = float(arr)

        def write(self):
            row = dict(event='train', steps=int(self.step), updates=context.updates, train_seconds=train_seconds(),
                       process_seconds=time.monotonic()-process_begin, episodes=episodes.result(),
                       episodes_total=context.episodes_total,
                       decoder_losses={k: self.values.get(f'train/loss/{k}') for k in decoder_keys},
                       replay_items=len(replay), metrics=self.values)
            if prealloc:
                stats = jax.devices()[0].memory_stats() or {}
                row['allocator'] = {k: stats[k] for k in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit',
                                                         'pool_bytes', 'peak_pool_bytes') if k in stats}
            append(row)
            print(json.dumps(clean({k: v for k, v in row.items() if k != 'metrics'})), flush=True)
            self.values = {}
            episodes.reset()

        def close(self):
            self.write()
    logger = Logger()

    class EvalActor:
        """Deterministic (eval.action='mode') or sampled actions with the current parameters; carry, action zeroing
        and per-step key schedule as nav4.evaluate's DreamerPolicy(0) (key PRNGKey(1234))."""

        def __init__(self, slots):
            self.carry, self.key = policy.initial(slots), jax.random.PRNGKey(1234)
            self.fn = policy._mode if e['action'] == 'mode' else policy._policy

        def __call__(self, obs):
            obs = {k: obs[k] for k in policy.obs_space}
            self.key, key = jax.random.split(self.key)
            _, (carry, act, _) = self.fn(policy.agent.params, self.carry, obs, mode='train', seed=key)
            action = jnp.where(obs['is_last'][:, None], 0., act['action'])
            self.carry = (*carry[:3], {'action': action})
            return action

    def run_eval():
        begin = time.time()
        state, obs, _ = eval_env.reset(jax.random.PRNGKey(e['seed']), start_s=eval_env.cfg['task']['route_start'])
        rows, _, loop_steps, budget, seconds = evaluate.drive(eval_env, state, obs, EvalActor(eval_env.n), None,
                                                              e['assist'], e['max_assists'], e['max_steps'])
        summary = evaluate.summarize(rows, eval_env.cfg['timing']['dt'], eval_env.cfg['assist']['limit'])
        append(dict(event='eval', index=context.evals, steps=int(logger.step), updates=context.updates,
                    train_seconds=train_seconds(), eval_seconds=time.time()-begin, loop_steps=loop_steps, budget=budget,
                    action=e['action'], assist=e['assist'], seed=e['seed'],
                    summary={k: v for k, v in summary.items() if k != 'per_map'},
                    rows=[{k: r[k] for k in ('slot', 'success', 'arrived', 'truncated', 'assists', 'final_station',
                                             'steps', 'distance', 'failures')} for r in rows]))
        print('EVAL', json.dumps(clean(dict(index=context.evals, updates=context.updates, **{
            k: summary.get(k) for k in ('strict_completions', 'completions_within_limit', 'assists_per_100m',
                                        'mean_final_station')}))), flush=True)
        # Keep the best evaluated weights (fewest assists per 100 m, then furthest mean final station).
        score = [summary['assists_per_100m'], -summary['mean_final_station']]
        best = logdir/'best.json'
        if not best.exists() or score < json.loads(best.read_text())['score']:
            write_latest('best.pkl')
            write_atomic(best, json.dumps(clean(dict(
                score=score, index=context.evals, updates=context.updates, train_seconds=train_seconds(),
                summary={k: v for k, v in summary.items() if k != 'per_map'}))).encode())
        context.evals += 1
        while context.next_eval <= train_seconds():
            context.next_eval += e['minutes']*60.

    class Driver:
        def __init__(self, *_, **__):
            self.callbacks, self.carry, self.restored, self.acts = [], None, False, None
            self.state, self.obs, self.extra = initial
            context.driver = self

        def on_step(self, fn):
            if batch_writer is None or fn != replay.add:
                self.callbacks.append(fn)

        def reset(self, init_policy=None):
            if not self.restored or self.carry is None:
                self.carry = init_policy(n)
            self.restored = False

        def __call__(self, fn, steps=0, episodes_=0):
            if context.stop or train_seconds() >= a.seconds:
                raise SegmentComplete()
            if context.begin is not None and train_seconds() >= context.next_eval:
                run_eval()
            nonlocal previous_end
            if profiling:
                begin = time.perf_counter()
                steady = context.begin is not None
                gap = begin - previous_end if previous_end is not None else 0.
                phase.clear()
                sample_start = sample_work[0]
            stepped = self.acts is not None
            if stepped:
                self.state, self.obs, self.extra = env.step(self.state, self.acts)
            if profiling:
                jax.block_until_ready((self.state, self.obs, self.extra))
                mark = time.perf_counter()
                phase['env'] = mark - begin
            self.carry, acts, outs = fn(self.carry, {k: self.obs[k] for k in policy.obs_space})
            self.acts = acts['action']
            if profiling:
                jax.block_until_ready((self.carry, acts, outs))
                now = time.perf_counter()
                phase['policy'] = now - mark
                mark = now
            tran, ep = jax.device_get(({**{k: self.obs[k] for k in policy.obs_space}, **acts, **outs}, dict(
                done=self.obs['is_last'], ep_return=self.state['ep_return'], failure=self.state['failure'],
                steps=self.state['steps'], progress=self.extra['progress'])))
            if profiling:
                phase['device_get'] = time.perf_counter() - mark
            if stepped:
                episodes.add(ep)
                context.episodes_total += int(ep['done'].sum())
            if batch_writer is not None:
                if profiling:
                    mark = time.perf_counter()
                batch_writer.add(tran)
                if profiling:
                    phase['replay_add'] += time.perf_counter() - mark
            for i in range(n):
                # Copies for worker 0: the official logfn keeps its images for a whole episode, and row views
                # would keep every slot's batch alive with them.
                row = {k: (np.array(v[i]) if i == 0 else v[i]) for k, v in tran.items()}
                if not profiling:
                    for callback in self.callbacks:
                        callback(row, i)
                    continue
                for idx, callback in enumerate(self.callbacks):
                    mark = time.perf_counter()
                    callback(row, i)
                    if callback == replay.add:
                        phase['replay_add'] += time.perf_counter() - mark
                    elif idx == (2 if batch_writer is not None else 3):
                        phase['episode_log'] += time.perf_counter() - mark
            if profiling:
                end = time.perf_counter()
                phase['driver_other'] = end - begin - sum(phase.values())
                phase['outer_loop_log'] = gap
                if steady:
                    append(dict(event='profile', updates=context.updates, steps=int(logger.step),
                                seconds=dict(phase), background_sample=sample_work[0]-sample_start))
                previous_end = time.perf_counter()

        def save(self):
            return dict(tree=jax.device_get((self.state, self.obs, self.extra, self.carry, self.acts)), envs=n)

        def load(self, data):
            assert data['envs'] == n, (data['envs'], n)
            self.state, self.obs, self.extra, self.carry, self.acts = jax.tree.map(jnp.asarray, data['tree'])
            self.restored = True

    class Runtime:
        def save(self):
            return dict(train_seconds=train_seconds(), updates=context.updates, ratio_prev=context.ratio._prev,
                        policy_key=np.asarray(policy.key), train_carry=jax.device_get(context.train_carry),
                        evals=context.evals, next_eval=context.next_eval, episodes_total=context.episodes_total)

        def load(self, data):
            context.elapsed, context.updates, context.ratio._prev = data['train_seconds'], data['updates'], data['ratio_prev']
            policy.key = jnp.asarray(data['policy_key'])
            context.pending_carry = (jax.tree.map(jnp.asarray, data['train_carry'])
                                     if data['train_carry'] is not None else None)
            context.train_carry = context.pending_carry
            context.evals, context.next_eval = data['evals'], data['next_eval']
            context.episodes_total = data['episodes_total']

    def write_latest(name='latest.pkl'):
        payload = dict(agent=policy.agent.save(), nav4=dict(
            format='nav4.train_dreamer/1', env_overrides=cfg['env'], obs_space=spec(policy.obs_space),
            manifest=cfg['manifest'], maps=list(bank.names), seed=seed, train_config=cfg),
            steps=int(logger.step), updates=context.updates, train_seconds=train_seconds(), envs=n,
            official_checkpoint=str(context.cp.latest()))
        write_atomic(logdir/name, pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))

    original_cp, original_ratio = elements.Checkpoint, elements.when.Ratio

    def ratio_factory(value):
        context.ratio = original_ratio(value)
        return context.ratio

    class Checkpoint(original_cp):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('keep', cfg['keep_checkpoints'])
            super().__init__(*args, **kwargs)
            self.driver, self.runtime = context.driver, Runtime()
            context.cp = self

        def save(self, path=None, keys=None):
            check = gpu if a.allow_cpu else gpu_memory()
            begin = time.time()
            super().save(path, keys)
            write_latest()
            context.saved_step = int(logger.step)
            append(dict(event='checkpoint', steps=context.saved_step, updates=context.updates,
                        train_seconds=train_seconds(), path=str(self.latest()), save_seconds=time.time()-begin,
                        gpu=check, replay_items=len(replay)))
            if not a.allow_cpu and not guard_ok(check, cfg['min_free_gpu_mib']):
                append(dict(event='gpu_guard_stop', gpu=check, min_free_gpu_mib=cfg['min_free_gpu_mib']))
                context.stop, context.exit_code = True, EXIT_GPU

    def stream(buffer, mode):
        while True:
            if profiling:
                start = time.perf_counter()
            batch = buffer.sample(cfg['batch_size'], mode)
            batch['consec'] = np.zeros(batch['is_first'].shape, np.int32)
            if profiling:
                sample_work[0] += time.perf_counter() - start
            yield batch

    args = SimpleNamespace(
        logdir=str(logdir), usage=dict(psutil=True, nvsmi=False), batch_size=cfg['batch_size'],
        batch_length=cfg['batch_length'], train_ratio=cfg['train_ratio'], log_every=cfg['log_seconds'],
        report_every=0, save_every=cfg['save_minutes']*60, envs=n, debug=True, from_checkpoint='',
        from_checkpoint_regex='.*', steps=10**15, consec_report=1, report_batches=1)
    training = importlib.import_module('embodied.run.train')
    finished = False
    try:
        with patch.object(embodied, 'Driver', Driver), patch.object(elements, 'Checkpoint', Checkpoint), \
                patch.object(elements.when, 'Ratio', ratio_factory):
            training.train(lambda: Agent(), lambda: Replay(), lambda _: None, stream, lambda: logger, args)
        finished = True
    except SegmentComplete:
        finished = True
    finally:
        if finished and context.cp is not None:
            if e.get('at_end', False) and not context.stop and context.begin is not None:
                context.elapsed = train_seconds()
                context.begin = None
                # The upstream agent returns metrics one update late.
                last = jax.device_get(policy.agent.pending_mets or {})
                check_losses(last)
                append(dict(event='last_update', updates=context.updates, train_seconds=train_seconds(),
                            metrics={k: float(v) for k, v in last.items()
                                     if k.startswith('loss/') or k in ('opt/loss', 'opt/grad_norm')}))
                run_eval()
            if context.saved_step != int(logger.step) and not cfg.get('benchmark_only', False):
                context.cp.save()
            logger.close()
            append(dict(event='stop', steps=int(logger.step), updates=context.updates, train_seconds=train_seconds(),
                        process_seconds=time.monotonic()-process_begin, exit_code=context.exit_code,
                        reason='gpu_guard' if context.exit_code else ('signal' if context.stop else 'budget')))
    return context.exit_code


if __name__ == '__main__':
    sys.exit(main())
