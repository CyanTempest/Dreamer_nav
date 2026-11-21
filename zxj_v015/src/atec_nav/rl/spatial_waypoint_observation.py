"""Explicit spatial distance/bearing observations; no action or reward changes.

This prepared observation contract is not enabled by existing environments.
Construct once per route, then reuse the cached geometry for each observation.
"""
from copy import deepcopy

import numpy as np

from atec_nav.rl.waypoint_observation import waypoint_goal
from atec_nav.robots.reference_tracking import teacher_reference
from atec_nav.robots.spatial_reference import LocalSpatialReference


class SpatialWaypointObservation:
    def __init__(self, route, config):
        self.config = deepcopy(config)
        settings = self.config['observations'].get('spatial_waypoint_reference')
        expected = {'kind', 'window_m', 'sigma_m', 'spacing_m', 'search_m'}
        if (not isinstance(settings, dict) or set(settings) != expected
                or settings['kind'] != 'local_spatial_v1'):
            raise ValueError('Explicit local_spatial_v1 observation settings required')
        if self.config['environment'].get('route_projection') != 'metric_heading_v1':
            raise ValueError('Spatial waypoints require metric_heading_v1 route matching')
        reference = self.config['observations'].get('waypoint_reference')
        if not isinstance(reference, dict) or not reference:
            raise ValueError('Explicit waypoint reference profile required')
        self.profile = {'teacher_' + key: value for key, value in reference.items()}
        self.reference = LocalSpatialReference(
            route, self.profile, window=settings['window_m'], sigma_m=settings['sigma_m'],
            spacing_m=settings['spacing_m'], search_m=settings['search_m'])
        self.route = self.reference.route
        # Apply the existing profile validator, including rejection of action
        # settings such as a target speed in the navigation observation.
        waypoint_goal(self.route, self.route[0, 1:3], 0., self.route[0, 0],
                      self.route[-1, 0], self.config)

    def __call__(self, xy, yaw, progress, goal_s):
        legacy = waypoint_goal(self.route, xy, yaw, progress, goal_s, self.config)
        lookahead, offset = teacher_reference(self.profile, progress)
        # Preserve the original point goal, including short stages ending in a
        # blend region; no lateral endpoint is introduced by the new reference.
        if self.reference.weight(progress) == 0. or progress + lookahead >= goal_s:
            return legacy
        target = self.reference.target(np.asarray(xy), progress, goal_s, lookahead, offset)
        delta = target - np.asarray(xy)
        angle = np.arctan2(delta[1], delta[0]) - yaw
        return np.array([np.linalg.norm(delta), np.sin(angle), np.cos(angle)], np.float32)
