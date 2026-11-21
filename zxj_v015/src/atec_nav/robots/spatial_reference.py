"""Opt-in local reference geometry; never changes route scoring or actions.

This experimental planner is not enabled in the current Dreamer observations.
The original route parameter remains authoritative for progress and the goal.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from atec_nav.robots.reference_tracking import teacher_reference


def tracking_command(target, xy, yaw, progress, goal_s, speed, respect_yaw_limit=False):
    """Teacher-only pure pursuit using an explicit target; no recovery override."""
    delta = np.asarray(target)-xy
    angle = (np.arctan2(delta[1], delta[0])-yaw+np.pi) % (2*np.pi)-np.pi
    vx = min(speed*max(.2, np.cos(angle)), max(.08, (goal_s-progress)*.6))
    curvature = 2*np.sin(angle)/max(.4, np.linalg.norm(delta))
    if respect_yaw_limit and abs(curvature) > 1e-9:
        vx = min(vx, .3/abs(curvature))
    return float(vx), float(np.clip(vx*curvature, -.3, .3))


class LocalSpatialReference:
    def __init__(self, route, profile, *, window=(35., 38., 48.5, 51.5),
                 sigma_m=.75, spacing_m=.02, search_m=3.):
        self.route = np.asarray(route, dtype=float).copy()
        self.window = np.asarray(window, dtype=float)
        if (self.route.ndim != 2 or self.route.shape[1] < 3 or len(self.route) < 2
                or not np.isfinite(self.route).all()
                or np.any(np.diff(self.route[:, 0]) <= 0)
                or self.window.shape != (4,) or not np.isfinite(self.window).all()
                or np.any(np.diff(self.window) <= 0)
                or not np.isfinite([sigma_m, spacing_m, search_m]).all()
                or min(sigma_m, spacing_m, search_m) <= 0):
            raise ValueError('Invalid spatial reference geometry')
        self.search_m = search_m
        count = int(np.ceil(np.ptp(self.route[:, 0]) / spacing_m)) + 1
        self.original_s = np.linspace(self.route[0, 0], self.route[-1, 0], count)
        base = np.column_stack([np.interp(self.original_s, self.route[:, 0], self.route[:, j]) for j in (1, 2)])
        actual_spacing = self.original_s[1] - self.original_s[0]
        center = base + self.weight(self.original_s)[:, None] * (
            gaussian_filter1d(base, sigma_m / actual_spacing, axis=0, mode='nearest') - base)
        tangent = np.gradient(center, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm < 1e-9):
            raise ValueError('Reference contains a degenerate tangent')
        tangent /= norm[:, None]
        offset = np.array([teacher_reference(profile, s)[1] for s in self.original_s])
        if not np.isfinite(offset).all() or np.any(abs(offset) > .2):
            raise ValueError('Invalid lateral reference')
        self.xy = center + offset[:, None] * np.column_stack([-tangent[:, 1], tangent[:, 0]])
        self.segment = np.diff(self.xy, axis=0)
        self.length = np.linalg.norm(self.segment, axis=1)
        if np.any(self.length < 1e-9):
            raise ValueError('Reference contains a zero-length segment')
        self.arc = np.r_[0., np.cumsum(self.length)]

    def weight(self, s):
        start, full_start, full_end, end = self.window
        def smooth(x):
            x = np.clip(x, 0., 1.)
            return x*x*(3.-2.*x)
        return smooth((s-start)/(full_start-start))*smooth((end-s)/(end-full_end))

    def point(self, s):
        return np.array([np.interp(s, self.route[:, 0], self.route[:, j]) for j in (1, 2)])

    def target(self, xy, progress, goal_s, lookahead, lateral_offset):
        if (not np.isfinite([*xy, progress, goal_s, lookahead, lateral_offset]).all()
                or lookahead <= 0 or abs(lateral_offset) > .2):
            raise ValueError('Invalid spatial target query')
        target_s = min(goal_s, progress + lookahead)
        legacy = self.point(target_s)
        if lateral_offset:
            tangent = self.point(target_s+.1)-self.point(target_s-.1)
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            legacy += lateral_offset*np.array([-tangent[1], tangent[0]])
        weight = float(self.weight(progress))
        if weight == 0.:
            return legacy
        lo = max(0, int(np.searchsorted(self.original_s, progress-self.search_m))-1)
        hi = min(len(self.segment), int(np.searchsorted(self.original_s, progress+self.search_m))+1)
        if hi <= lo:
            raise ValueError('Progress outside spatial reference')
        vec = self.segment[lo:hi]
        fraction = np.clip(np.sum((xy-self.xy[lo:hi])*vec, axis=1)/np.sum(vec*vec, axis=1), 0., 1.)
        feet = self.xy[lo:hi]+fraction[:, None]*vec
        index = int(np.argmin(np.sum((feet-xy)**2, axis=1)))
        projected = self.arc[lo+index]+fraction[index]*self.length[lo+index]
        goal_arc = np.interp(goal_s, self.original_s, self.arc)
        target_arc = min(goal_arc, projected+lookahead)
        target = np.array([np.interp(target_arc, self.arc, self.xy[:, j]) for j in (0, 1)])
        # Short evaluation stages keep their original endpoint, including when
        # their finish lies inside the altered local reference section.
        endpoint_weight = float(np.clip((goal_arc-target_arc)/.5, 0., 1.))
        target = self.point(goal_s)+endpoint_weight*(target-self.point(goal_s))
        return legacy+weight*(target-legacy)
