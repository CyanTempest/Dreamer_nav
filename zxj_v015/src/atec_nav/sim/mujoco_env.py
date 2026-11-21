"""Action-driven Gymnasium M1 environment on the measured ATEC route.

The agent controls forward velocity and yaw rate through real wheel torques.
No route-following teacher, base pose overwrite, or lateral-motion fiction.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import json
from dataclasses import replace
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from atec_nav.config import project_path
from atec_nav.perception.depth import DepthNoise, NoiseConfig, pack_depth, pack_multiscale_depth
from atec_nav.robots.rolling import RollingController
from atec_nav.robots.terrain_rolling import TerrainRollingController
from atec_nav.sim.mujoco_route import load_scene, route_position, initialize_on_terrain
from atec_nav.rl.navigation_signals import signed_route_signal, slew_command, goal_progress_potential
from atec_nav.rl.waypoint_observation import waypoint_goal
from atec_nav.rl.stagnation_observation import validate_stagnation_settings,stagnation_feature,update_progress_clock


def readiness_issues(config, root):
    issues = []
    for key, names in [("robot_path", ["robot.xml", "manifest.json"]),
                       ("terrain_path", ["terrain.npz", "centerline.csv", "visual.obj", "obstacles.npz", "manifest.json"])]:
        folder = project_path(root, config["environment"].get(key, ""))
        issues.extend(f"Missing physical asset: {folder / name}" for name in names if not (folder / name).is_file())
    if config["actions"].get("vy_mps", 0) != 0:
        issues.append("Wheel baseline cannot command lateral velocity")
    if config["environment"]["backend"] != "mujoco_m1":
        issues.append("Expected mujoco_m1 backend")
    if config['environment'].get('locomotion_controller')=='matrix_stair':
        if os.environ.get('ATEC_MATRIX_ISOLATED')!='1':
            issues.append('matrix_stair requires scripts/run_matrix_isolated.sh')
        for name in ['build/matrix/libatec_matrix_shm.so',
                     'third_party/runtimes/MATRiX_Robot_MC-v0.6.6/build/export/mc/bin/mc_ctrl']:
            if not (Path(root)/name).is_file():issues.append('Missing Robot MC runtime: '+name)
    return issues


class M1NavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config, root, index=0):
        issues = readiness_issues(config, root)
        if issues: raise ValueError("\n".join(issues))
        self.config = config
        self.root = Path(root)
        self.curriculum = None
        curriculum=config.get('curriculum',{})
        if curriculum.get('enabled',False):
            from atec_nav.rl.curriculum import NavigationCurriculum
            self.curriculum=NavigationCurriculum(curriculum.get('lengths_m',[]),curriculum.get('successes_required',3),curriculum.get('stages'))
        env, obs = config["environment"], config["observations"]
        self.route_projection = env.get('route_projection', 'legacy_ordered')
        if self.route_projection not in ('legacy_ordered', 'metric_heading_v1'):
            raise ValueError('Unsupported route projection: '+self.route_projection)
        self.observation_schema = obs.get('schema', 'legacy_v1')
        if self.observation_schema not in ('legacy_v1', 'physical_v2'):
            raise ValueError('Unsupported observation schema: ' + self.observation_schema)
        self.physical_observations = self.observation_schema == 'physical_v2'
        self.stagnation_settings=obs.get('stagnation_age')
        validate_stagnation_settings(self.stagnation_settings)
        if self.stagnation_settings is not None and not self.physical_observations:
            raise ValueError('Stagnation age requires the explicit physical_v2 observation contract')
        depth_channels, proprio_size = (3, 64) if self.physical_observations else (2, 12)
        if obs.get('depth_channels', depth_channels) != depth_channels or obs.get('proprio_size', proprio_size) != proprio_size:
            raise ValueError('Observation dimensions do not match the declared schema')
        controller_name=env.get('locomotion_controller','rolling')
        self.matrix_mode=controller_name=='matrix_stair'
        self.model, self.route, self.grid, self.robot_info = load_scene(
            project_path(root, env["robot_path"]), project_path(root, env["terrain_path"]), matrix_imu=self.matrix_mode)
        self.data = mujoco.MjData(self.model)
        self.clearance_probe = None
        self.body_clearance = None
        if config['reward'].get('clearance_weight', 0.):
            from atec_nav.sim.clearance import BodyClearance
            self.clearance_probe = BodyClearance(self.model, config['reward'].get('clearance_margin_m', .04),
                config['reward'].get('clearance_include_foot_obstacles', False))
        if controller_name not in ('rolling','terrain_rolling','matrix_stair'):
            raise ValueError('Unsupported training locomotion controller: '+controller_name)
        controller_type=TerrainRollingController if controller_name=='terrain_rolling' else RollingController
        if self.matrix_mode:
            from atec_nav.robots.matrix_controller import MatrixController
            self.controller=MatrixController(self.model,self.robot_info,root)
        else:self.controller = controller_type(self.model, self.robot_info)
        self.dt = 1 / env["control_hz"]
        self.substeps = round(self.dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.dt):
            raise ValueError("Control period must be an integer number of physics steps")
        self.height, self.width = obs["height"], obs["width"]
        self.renderer = None
        self.option = mujoco.MjvOption()
        self.option.geomgroup[2:4] = 0  # Do not feed navigation markers to the agent.
        self.joint_contact_state = None
        if self.physical_observations:
            from atec_nav.perception.robot_state import JointContactState
            self.joint_contact_state = JointContactState(self.model)
            # Same surfaces as dynamics: physical geoms in group 3, no visual
            # proxy in group 1, no route markers in group 2. This is simulated
            # collision-aligned sensing, not a calibrated real depth camera.
            self.option.geomgroup[:] = 0
            self.option.geomgroup[3] = 1
        self.noise = DepthNoise(NoiseConfig(**config["noise"]), seed=config["seed"]+index, max_m=obs["max_depth_m"])
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(0, 255, (self.height, self.width, depth_channels), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (proprio_size,), dtype=np.float32),
            "goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.last_action = np.zeros(2, np.float32)
        self.executed_command=np.zeros(2,dtype=float)
        self.done = True

    def _renderer(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width,
                max_geom=max(10000, self.model.ngeom + 1024) if self.physical_observations else 10000)
            mujoco.mjr_uploadHField(self.model, self.renderer._mjr_context, 0)
        return self.renderer

    def _obs(self):
        renderer = self._renderer()
        renderer.update_scene(self.data, camera="navigation", scene_option=self.option)
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        cfg = self.config["observations"]
        noisy_depth = self.noise(depth)
        packed = (pack_multiscale_depth(noisy_depth, cfg['min_depth_m'], cfg.get('near_max_depth_m', 2.), cfg['max_depth_m'])
                  if self.physical_observations else pack_depth(noisy_depth, cfg['min_depth_m'], cfg['max_depth_m']))
        rotation = self.data.xmat[1].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        proprio = np.concatenate([rotation.T @ self.data.qvel[:3], self.data.qvel[3:6],
                                        rotation.T @ [0., 0., -1.],
                                        [self.executed_command[0]/self.config['actions']['vx_mps'], 0.,
                                         self.executed_command[1]/self.config['actions']['wz_radps']]]).astype(np.float32)
        if self.joint_contact_state is not None:
            proprio = np.concatenate([proprio, self.joint_contact_state.observe(self.model, self.data)])
        if self.stagnation_settings is not None:
            proprio[10]=stagnation_feature(self.elapsed,self.last_progress_time,self.stagnation_settings)
        return {
            "depth": packed,
            "proprio": proprio,
            "goal": waypoint_goal(self.route,self.data.qpos[:2],yaw,self.progress,self.goal_s,self.config),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        env = self.config["environment"]
        self.start_s = float(options.get("start_s", self.curriculum.start if self.curriculum else env.get('start_s',0.)))
        if env.get("random_start", False) and "start_s" not in options:
            self.start_s = self.np_random.uniform(0, max(0, self.route[-1, 0] - env["route_length_m"]))
        if not 0 <= self.start_s < self.route[-1, 0]: raise ValueError("Start outside route")
        length=self.curriculum.length if self.curriculum else env['route_length_m']
        self.goal_s = min(self.route[-1, 0], self.start_s + length)
        self.goal_xy = np.array([np.interp(self.goal_s, self.route[:, 0], self.route[:, j]) for j in (1, 2)])
        self.episode_limit=max(60.,length*7.) if self.curriculum else env['episode_seconds']
        self.index = max(0, min(len(self.route)-2, np.searchsorted(self.route[:, 0], self.start_s)-1))
        point = np.array([np.interp(self.start_s, self.route[:, 0], self.route[:, j]) for j in (1, 2, 3)])
        delta = self.route[self.index+1, 1:3] - self.route[self.index, 1:3]
        yaw = np.arctan2(delta[1], delta[0])
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = point + [0, 0, .56]
        self.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.controller.yaw_integral = 0.
        self.controller.initialize(self.data)
        if env.get('terrain_aware_reset',False):
            lateral = float(env.get('reset_lateral_jitter_m', 0.))
            heading = float(env.get('reset_yaw_jitter_rad', 0.))
            if not (0 <= lateral <= .1 and 0 <= heading <= .15):
                raise ValueError('Reset perturbations exceed configured conservative bounds')
            self.reset_perturbation = {
                'lateral_m': float(self.np_random.uniform(-lateral, lateral)) if lateral else 0.,
                'yaw_rad': float(self.np_random.uniform(-heading, heading)) if heading else 0.}
            initialize_on_terrain(self.model,self.data,self.controller,self.route,self.start_s,
                self.reset_perturbation['lateral_m'], self.reset_perturbation['yaw_rad'])
        mujoco.mj_forward(self.model, self.data)
        if self.matrix_mode:
            self.controller.start(self.data)
            if env.get('validate_reset_stability',False) and self._failure():
                self.controller.close()
                raise RuntimeError('Invalid Robot MC reset: '+self._failure())
        else:
            for _ in range(round(3. / self.model.opt.timestep)):
                self.controller.command(self.data, 0., 0.)
                mujoco.mj_step(self.model, self.data)
                if env.get('validate_reset_stability',False):
                    failure=self._failure()
                    if failure:
                        raise RuntimeError('Invalid physical reset during settling: '+failure)
        self.episode_start_time=float(self.data.time)
        mujoco.mj_forward(self.model, self.data)
        self.controller.yaw_integral = 0.
        self.executed_command[:]=0
        if self.curriculum and self.curriculum.noise_enabled is not None:
            self.noise.config=replace(self.noise.config,enabled=self.curriculum.noise_enabled)
        self.noise.reset(seed)
        self.last_action[:] = 0
        self.index, self.progress, self.error = self._project_route()
        self.reward_progress,self.reward_error=self._reward_signal()
        self.progress_potential = self._progress_potential()
        self.elapsed = 0.
        self.best_progress=self.progress_potential
        self.last_progress_time=0.
        self.episode_return=0.
        self.steps = 0
        self.physics_steps = 0
        self.done = False
        self.body_clearance = self.clearance_probe.measure(self.model, self.data) if self.clearance_probe else None
        return self._obs(), self._info("reset")

    def _reward_signal(self):
        if self.config['reward'].get('signed_route_progress',False):
            return signed_route_signal(self.route,self.data.qpos[:2],self.index,self.progress,self.error)
        return self.progress,self.error

    def _progress_potential(self):
        if self.config['reward'].get('goal_bounded_progress', False):
            return goal_progress_potential(self.reward_progress, self.goal_s,
                                           np.linalg.norm(self.data.qpos[:2] - self.goal_xy))
        return self.reward_progress

    def _project_route(self):
        if self.route_projection == 'metric_heading_v1':
            from atec_nav.sim.metric_route_projection import metric_route_position
            rotation = self.data.xmat[1].reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            return metric_route_position(self.route, self.data.qpos[:2], self.index, yaw)
        return route_position(self.route, self.data.qpos[:2], self.index)

    def _failure(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all(): return "nonfinite"
        if self.data.xmat[1].reshape(3, 3)[2, 2] < .75: return "fall"
        for contact in self.data.contact:
            if contact.dist >= -.0001: continue
            geoms = [self.model.geom(g).name for g in (contact.geom1, contact.geom2)]
            bodies = [self.model.body(int(self.model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
            if any(name.startswith("obstacle_") for name in geoms): return "collision"
            if "world" in bodies and not any("FOOT_LINK" in name for name in bodies): return "collision"
        return ""

    def _info(self, reason):
        return {"backend": "mujoco_m1", "progress_m": self.progress-self.start_s,
                "locomotion_controller": self.config['environment'].get('locomotion_controller','rolling'),
                "controller_state": self.controller.last_state if self.matrix_mode else None,
                "controller_pauses": self.controller.pause_count if self.matrix_mode else 0,
                "cross_track_m": self.reward_error, "sim_time_s": self.elapsed,
                "target_length_m": self.goal_s-self.start_s,
                "goal_distance_m": float(np.linalg.norm(self.data.qpos[:2]-self.goal_xy)),
                "collision_contacts": self._collision_contacts() if reason == 'collision' else [],
                "start_s": self.start_s,
                "reset_perturbation": getattr(self, 'reset_perturbation', None),
                "body_clearance_estimate": self.body_clearance,
                "signed_progress_m": self.reward_progress-self.start_s,
                "progress_potential_m": self.progress_potential-self.start_s,
                "raw_action": self.last_action.tolist(),
                "executed_command": self.executed_command.tolist(),
                "curriculum_stage": self.curriculum.stage if self.curriculum else None,
                "localization": "simulation_ground_truth", "navigation_policy": "agent_actions",
                "route_projection": self.route_projection,
                "waypoint_reference": self.config['observations'].get('waypoint_reference'),
                "stagnation_age_observation":self.stagnation_settings,
                "progress_stagnation_s":max(0.,self.elapsed-self.last_progress_time),
                "reason": reason, "success": reason == "success", "physics_steps": self.physics_steps}

    def _collision_contacts(self):
        result=[]
        for contact in self.data.contact:
            if contact.dist >= -.0001:continue
            geoms=[self.model.geom(g).name for g in (contact.geom1,contact.geom2)]
            bodies=[self.model.body(int(self.model.geom_bodyid[g])).name for g in (contact.geom1,contact.geom2)]
            if any(n.startswith('obstacle_') for n in geoms) or ('world' in bodies and not any('FOOT_LINK' in n for n in bodies)):
                result.append({'geoms':geoms,'bodies':bodies,'penetration_m':-float(contact.dist),
                               'position':contact.pos.tolist()})
        return result

    def step(self, action):
        if self.done: raise RuntimeError("Reset after episode termination")
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action): raise ValueError("Action must be finite (vx, wz) in [-1, 1]")
        limits = self.config["actions"]
        previous = self.progress_potential
        previous_command=self.executed_command.copy()
        target=action*np.array([limits['vx_mps'],limits['wz_radps']])
        rates=limits.get('slew_rates')
        reason = ""
        substep_collision_contacts = None
        try:
            for _ in range(self.substeps):
                self.executed_command=(slew_command(self.executed_command,target,rates,self.model.opt.timestep)
                                       if rates else target.copy())
                self.controller.command(self.data,*map(float,self.executed_command))
                mujoco.mj_step(self.model, self.data)
                self.physics_steps += 1
                reason = self._failure()
                if reason:
                    if reason == "collision":
                        substep_collision_contacts = self._collision_contacts()
                    break
        finally:
            if self.matrix_mode:self.controller.pause(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.index, self.progress, self.error = self._project_route()
        self.reward_progress,self.reward_error=self._reward_signal()
        self.progress_potential = self._progress_potential()
        self.steps += 1
        self.elapsed = float(self.data.time - self.episode_start_time)
        self.best_progress,self.last_progress_time=update_progress_clock(
            self.progress_potential,self.best_progress,self.last_progress_time,self.elapsed)
        stall_timeout=self.config['environment'].get('stall_timeout_s',0.)
        if not reason and stall_timeout and self.elapsed-self.last_progress_time>stall_timeout:
            reason='stalled'
        if not reason and self.reward_error > .6: reason = "corridor_exit"
        if not reason and self.progress >= self.goal_s - .15 and np.linalg.norm(self.data.qpos[:2]-self.goal_xy) < .25:
            reason = "success"
        terminated = bool(reason)
        episode_limit=self.episode_limit if self.curriculum else self.config['environment']['episode_seconds']
        truncated = not terminated and self.elapsed >= episode_limit - 1e-8
        if truncated: reason = "timeout"
        reward_cfg = self.config["reward"]
        reward = reward_cfg["progress_weight"] * (self.progress_potential-previous) - reward_cfg["time_cost_per_second"] * self.dt
        change=(self.executed_command-previous_command)/np.array([limits['vx_mps'],limits['wz_radps']]) if rates else action-self.last_action
        reward -= reward_cfg["action_change_weight"] * float(np.square(change).sum())
        reward -= reward_cfg.get('cross_track_weight',0.) * self.reward_error * self.dt
        if self.clearance_probe:
            from atec_nav.sim.clearance import clearance_cost
            self.body_clearance = self.clearance_probe.measure(self.model, self.data)
            reward -= reward_cfg['clearance_weight'] * clearance_cost(
                self.body_clearance['minimum_m'], self.clearance_probe.margin) * self.dt
        if reason == "success": reward += reward_cfg["success_bonus"]
        elif reason == "fall": reward -= reward_cfg["fall_penalty"]
        elif terminated: reward -= reward_cfg["collision_penalty"]
        self.last_action[:] = action
        self.done = terminated or truncated
        self.episode_return+=float(reward)
        info=self._info(reason or 'running')
        if substep_collision_contacts is not None:
            info["collision_contacts"] = substep_collision_contacts
            info["collision_contacts_source"] = "terminal_physics_substep"
        if self.done:
            log=self.config['environment'].get('episode_log')
            if log:
                path=project_path(self.root,log);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a') as stream:stream.write(json.dumps({**info,'return':self.episode_return,'steps':self.steps})+'\n')
            if self.curriculum:self.curriculum.finish(reason=='success')
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        renderer = self._renderer()
        renderer.update_scene(self.data, camera="navigation", scene_option=self.option)
        return renderer.render().copy()

    def close(self):
        if self.matrix_mode:self.controller.close()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.grid.close()


def make_env(config, root, index=0):
    return M1NavigationEnv(config, root, index)
