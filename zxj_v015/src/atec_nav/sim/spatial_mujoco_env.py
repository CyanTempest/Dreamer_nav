"""Opt-in development environment for spatial waypoint observations.

The measured scene, physics, actions, progress, rewards and termination all
remain in M1NavigationEnv. Only the navigation goal observation is replaced.
"""
import hashlib
from pathlib import Path

import numpy as np

from atec_nav.rl.spatial_waypoint_observation import SpatialWaypointObservation
from atec_nav.sim.mujoco_env import M1NavigationEnv


class SpatialM1NavigationEnv(M1NavigationEnv):
    def __init__(self, config, root, index=0):
        if not config['observations'].get('spatial_waypoint_reference'):
            raise ValueError('Spatial development environment needs an explicit observation contract')
        super().__init__(config, root, index)
        transport=config['environment'].get('speed_transport')
        if transport is not None:
            from atec_nav.robots.calibrated_matrix_controller import CalibratedMatrixController
            if config['actions']!=dict(vx_mps=.5,vy_mps=0.,wz_radps=.3,slew_rates=[.5,1.]):
                raise ValueError('Calibrated transport requires the explicit 0.5 m/s action contract')
            self.controller.close()
            self.controller=CalibratedMatrixController(self.model,self.robot_info,root,transport)
        self.spatial_waypoint_observation = SpatialWaypointObservation(self.route, config)
        self.spatial_observation_source_sha256 = hashlib.sha256(
            (Path(__file__).parents[1]/'rl/spatial_waypoint_observation.py').read_bytes()).hexdigest()

    def _obs(self):
        observation = super()._obs()
        rotation = self.data.xmat[1].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        observation['goal'] = self.spatial_waypoint_observation(
            self.data.qpos[:2], yaw, self.progress, self.goal_s)
        return observation

    def _info(self, reason):
        result = {
            **super()._info(reason),
            'spatial_waypoint_reference': self.config['observations']['spatial_waypoint_reference'],
            'spatial_observation_source_sha256': self.spatial_observation_source_sha256,
            'experimental_observation_probe': True,
        }
        transport=self.config['environment'].get('speed_transport')
        if transport is not None:
            if self.controller.transport_log is not None:self.controller.transport_log.flush()
            result['speed_transport']=dict(settings=transport,
                packet_log=str(self.controller.transport_paths[-1]) if self.controller.transport_paths else None)
        return result


def make_env(config, root, index=0):
    return SpatialM1NavigationEnv(config, root, index)
