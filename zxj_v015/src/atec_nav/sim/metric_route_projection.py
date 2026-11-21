"""Opt-in ordered map matching with a metric search window and heading guard."""
import numpy as np


def metric_route_position(route, xy, previous_index, yaw, forward_m=3., backward_m=.2):
    """Project onto the original route; never alter its coordinates or length.

    The window is measured in original route metres, so dense samples cannot
    trap the match behind a small capture-path loop. Distant forward candidates
    on an oppositely directed return lane are excluded using measured heading.
    This does not change the physical corridor width or the endpoint test.
    """
    if (not np.isfinite([*xy, yaw, forward_m, backward_m]).all()
            or forward_m <= 0 or backward_m < 0 or not 0 <= previous_index < len(route)-1):
        raise ValueError('Invalid metric route projection query')
    anchor = float(route[previous_index, 0])
    low, high = max(route[0, 0], anchor-backward_m), min(route[-1, 0], anchor+forward_m)
    start = max(0, int(np.searchsorted(route[:, 0], low, side='right'))-1)
    end = min(len(route)-1, int(np.searchsorted(route[:, 0], high, side='right')))
    indices = np.arange(start, end)
    a = route[indices, 1:3]
    v = route[indices+1, 1:3]-a
    length2 = np.sum(v*v, axis=1)
    span = route[indices+1, 0]-route[indices, 0]
    if np.any(span <= 0) or np.any(length2 <= 1e-14):
        raise ValueError('Degenerate source route segment')
    lower = np.maximum(0., (low-route[indices, 0])/span)
    upper = np.minimum(1., (high-route[indices, 0])/span)
    fraction = np.clip(np.sum((xy-a)*v, axis=1)/length2, lower, upper)
    progress = route[indices, 0]+fraction*span
    distance = np.linalg.norm(a+fraction[:, None]*v-xy, axis=1)
    before = np.column_stack([np.interp(progress-.3, route[:, 0], route[:, j]) for j in (1, 2)])
    after = np.column_stack([np.interp(progress+.3, route[:, 0], route[:, j]) for j in (1, 2)])
    tangent = after-before
    alignment = tangent@np.array([np.cos(yaw), np.sin(yaw)])
    # Close matches remain available during turns and reversals. Only a jump
    # farther ahead needs a route direction consistent with the robot heading.
    allowed = (progress <= anchor+.6) | (alignment > 1e-9)
    distance_for_selection = np.where(allowed, distance, np.inf)
    chosen = int(np.argmin(distance_for_selection))
    if not np.isfinite(distance_for_selection[chosen]):
        raise ValueError('No ordered route projection candidate')
    return int(indices[chosen]), float(progress[chosen]), float(distance[chosen])
