"""Metric route reward and physical-command slew limits, independent of policy."""
import numpy as np

def signed_route_signal(route, xy, index, progress, error):
    # Extend only the first segment backwards. Route completion still uses the
    # clamped projection and the independent goal-distance check.
    if index == 0:
        vector=route[1,1:3]-route[0,1:3]
        fraction=float(np.dot(xy-route[0,1:3],vector)/np.dot(vector,vector))
        if fraction<0:
            return (float(route[0,0]+fraction*(route[1,0]-route[0,0])),
                    float(np.linalg.norm(xy-(route[0,1:3]+fraction*vector))))
    return float(progress),float(error)

def goal_progress_potential(progress, goal_s, goal_distance):
    """Route approach followed by point-goal approach; never rewards overshoot.

    The remaining cost is at least the Euclidean distance to the actual goal.
    On the route before the goal this preserves ordinary progress. Moving away
    after passing it reduces the potential, and a round trip has zero net gain.
    """
    if not np.isfinite([progress, goal_s, goal_distance]).all() or goal_distance < 0:
        raise ValueError('Finite progress and a nonnegative goal distance required')
    return float(goal_s - max(goal_s - progress, goal_distance))

def slew_command(current, target, rates, dt):
    rates=np.asarray(rates,dtype=float)
    if np.any(rates<=0) or dt<=0:raise ValueError('Positive slew rates and timestep required')
    return current+np.clip(target-current,-rates*dt,rates*dt)
