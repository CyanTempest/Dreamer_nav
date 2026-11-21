"""Reload a saved official DreamerV3 agent and drive the physical M1 environment."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from _bootstrap import ROOT
from atec_nav.rl.adapter import DreamerAdapter
from atec_nav.sim.mujoco_env import make_env

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument('--length',type=float,help='Evaluate a fixed route length, with curriculum disabled')
    parser.add_argument('--start-s',type=float,default=0.)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument("--seeds", type=int, nargs="+", help="Independent trials, reusing the compiled scene")
    parser.add_argument("--required-successes", type=int, help="Stop a batch once this target becomes unreachable")
    parser.add_argument('--action-mode',choices=['sample','mean'],default='sample')
    parser.add_argument('--depth-noise',action='store_true',help='Enable the saved depth-noise model for this evaluation')
    args = parser.parse_args()
    seeds = args.seeds or [args.seed]
    if min(seeds) < 0 or len(set(seeds)) != len(seeds):parser.error('Seeds must be distinct nonnegative integers')
    if args.required_successes is not None and (not args.seeds or not 1 <= args.required_successes <= len(seeds)):
        parser.error('required-successes needs a batch and must be within its size')
    training = args.training_run.resolve()
    audit = json.loads((training / "integration.json").read_text())
    nav = audit["config"]
    nav['seed']=seeds[0]
    nav['environment']['start_s']=args.start_s
    nav.setdefault('curriculum',{})['enabled']=False
    nav['environment'].pop('episode_log',None)
    if args.depth_noise:nav['noise']['enabled']=True
    if args.length is not None:
        if not 0 < args.length <=100:parser.error('length must be in (0,100]')
        nav['environment']['route_length_m']=args.length
    nav['environment']['episode_seconds']=max(60.,nav['environment']['route_length_m']*7.)
    if args.steps<1:parser.error('steps must be positive')
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / nav["dreamer"]["upstream_path"]))
    import elements
    import ruamel.yaml
    from dreamerv3 import main as upstream
    if args.action_mode=='mean':
        import jax
        from dreamerv3 import agent as policy_module
        # Evaluation-only override, before JIT tracing. No upstream file edited.
        policy_module.sample=lambda xs:jax.tree.map(lambda x:x.pred(),xs)
    config = elements.Config(ruamel.yaml.YAML(typ="safe").load((training / "config.yaml").read_text()))
    # The agent owns a separate RNG. Changing only the environment seed would
    # repeat the same policy samples in every supposedly independent trial.
    config = elements.Config({**config, "seed": seeds[0],
                              "jax": {**config.jax, "precompile": False,
                                      "platform": nav['dreamer'].get('platform','cpu')}})
    env = make_env(nav, ROOT)
    adapter = DreamerAdapter(env, seed=seeds[0])
    class ActualSpaces:
        # Agent construction only reads spaces; share the real environment
        # contract without allocating and discarding a second full scene.
        obs_space = adapter.obs_space
        act_space = adapter.act_space
        def close(self):pass
    upstream.make_env = lambda *a, **kw: ActualSpaces()
    try:
        agent = upstream.make_agent(config)
        checkpoint_path=training/'final/agent.pkl'
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        elements.checkpoint.load(training / "final", {"agent": agent.load})
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=checkpoint_sha256:
            raise RuntimeError('Checkpoint changed during evaluation loading')
        restored = agent.save()
        assert restored["counters"]["updates"] == audit["counters"]["updates"] > 0
        assert all(np.isfinite(p).all() for p in restored["params"].values())
        results = []
        for seed in seeds:
            trial = out / f"seed_{seed}" if args.seeds else out
            if args.seeds:trial.mkdir()
            reset_trial(agent, adapter, seed, restored['counters']['actions'])
            metrics = evaluate_trial(agent, adapter, nav, args, training, trial, restored, seed,checkpoint_sha256)
            results.append({'seed': seed, 'success': metrics['success'], 'reason': metrics['reason'],
                            'progress_m': metrics['final_progress_m'], 'output': str(trial)})
            report = batch_status(seeds, results, args.action_mode, args.required_successes)
            report['checkpoint_sha256']=checkpoint_sha256
            if args.seeds:
                temporary = out/'batch_metrics.json.tmp'
                temporary.write_text(json.dumps(report, indent=2)+'\n');temporary.replace(out/'batch_metrics.json')
            if report['gate_unreachable']:break
        assert int(agent.n_updates) == restored['counters']['updates']
    finally:
        adapter.close()


def batch_status(seeds, results, mode, required):
    failures = sum(not row['success'] for row in results)
    return {'planned_seeds': seeds, 'action_mode': mode, 'results': results,
            'complete': len(results)==len(seeds),
            'gate_unreachable': required is not None and failures > len(seeds)-required,
            'required_successes': required, 'teacher_used': False}


def reset_trial(agent, adapter, seed, checkpoint_actions):
    # The outer JAX agent derives every action RNG from config.seed and its
    # action counter. Reset both, as in fresh single-trial checkpoint loading.
    agent.config = agent.config.update({'seed': seed})
    with agent.n_actions.lock:
        agent.n_actions.value = int(checkpoint_actions)
    adapter.seed = seed
    adapter.done = True


def evaluate_trial(agent, adapter, nav, args, training, out, restored, seed,checkpoint_sha256=None):
    env = adapter.env
    observation = adapter.step(dict(reset=True, action=np.zeros(2, np.float32)))
    carry = agent.init_policy(1)
    states, velocities, times, progress, actions, rewards = [], [], [], [], [], []
    depths, executed_commands, policy_inputs = [], [], []
    error = None
    command = None
    try:
        for step in range(args.steps):
            # Preserve exactly what the policy saw, including its history
            # flags. Post-step depth alone cannot diagnose a recovery action.
            policy_input = {key: np.asarray(value).copy() for key, value in observation.items()}
            batch = {key: value[None] for key, value in observation.items()}
            command = None
            carry, action, _ = agent.policy(carry, batch, mode="eval")
            command = np.clip(action["action"][0], -1, 1).astype(np.float32)
            assert np.isfinite(command).all()
            observation = adapter.step(dict(reset=False, action=command))
            policy_inputs.append(policy_input)
            actions.append(command);rewards.append(float(observation["reward"]))
            executed_commands.append(env.executed_command.copy())
            states.append(env.data.qpos.copy());velocities.append(env.data.qvel.copy())
            times.append(float(env.data.time));progress.append(env.progress)
            depths.append(observation["depth"])
            if (step+1)%50==0:
                current = {'evaluation_step':step+1,**adapter.info}
                print(json.dumps(current),flush=True)
                temporary=out/'progress.json.tmp'
                temporary.write_text(json.dumps(current,indent=2)+'\n');temporary.replace(out/'progress.json')
            if observation["is_last"]: break
    except BaseException as exception:
        error = exception
        # A transport failure can interrupt one 500 Hz substep. Preserve that
        # partial state separately; never append it as a completed transition.
        np.savez_compressed(out/'aborted_step.npz', qpos=env.data.qpos.copy(),
            qvel=env.data.qvel.copy(), time=float(env.data.time),
            attempted_action=np.asarray(command) if command is not None else np.empty(0),
            **{'input_'+key:value for key,value in policy_input.items()})
    info = adapter.info
    if error is not None:
        info={**info,'reason':'evaluation_aborted','success':False,'error':repr(error)}
    elif not observation['is_last']:
        info={**info,'reason':'evaluation_step_limit','success':False}
    np.savez_compressed(out / "physical_states.npz", qpos=states, qvel=velocities,
                        time=times, progress_m=progress, action=actions,
                        executed_command=executed_commands, reward=rewards)
    np.savez_compressed(out / "observations.npz", depth=depths)
    if policy_inputs:
        np.savez_compressed(out/'policy_inputs.npz',
                            **{key:np.stack([row[key] for row in policy_inputs]) for key in policy_inputs[0]})
    manifest = {"status": "aborted" if error is not None else "evaluated", "policy": "DreamerV3_checkpoint", "training_run": str(training),
                "robot": str(ROOT / nav["environment"]["robot_path"]),
                "terrain": str(ROOT / nav["environment"]["terrain_path"]),
                "fps": nav["environment"]["control_hz"], "trained_updates": restored["counters"]["updates"],
                "locomotion": nav['environment'].get('locomotion_controller','rolling'),
                "base_pose_teleported_after_reset": False, "teacher_used": False,
                "action_mode":args.action_mode,"seed":seed,"policy_seed":agent.config.seed,
                "start_s":args.start_s,
                "checkpoint_reloaded": True, "policy_actions_finite": True,
                "checkpoint_sha256":checkpoint_sha256,
                "source_sha256": SOURCE_SHA256,
                "depth_noise":nav.get('noise',{}),
                "evaluation_completed":error is None,"eligible_for_navigation_gate":error is None}
    metrics = {**info, "steps": len(actions), "return": float(np.sum(rewards)),
               "route_length_m": env.goal_s-env.start_s, "final_progress_m": (progress[-1] if progress else env.progress)-env.start_s,
               "success": info["success"],
               "course_success":bool(info['success'] and env.goal_s-env.start_s>=100.),
               "evaluation_completed":error is None,"eligible_for_navigation_gate":error is None,
               "claim": "learned_policy_evaluation_no_teacher_actions"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if error is not None:raise error
    return metrics


if __name__ == "__main__": main()
