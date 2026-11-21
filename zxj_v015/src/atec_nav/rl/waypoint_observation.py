"""Planned metric waypoint observations, never motor or velocity overrides."""
import numpy as np
from atec_nav.robots.reference_tracking import teacher_reference


def waypoint_goal(route,xy,yaw,progress,goal_s,config):
    reference=config['observations'].get('waypoint_reference')
    if reference is None:
        lookahead=config['environment'].get('lookahead_m',2.)
        offset=0.
    else:
        # Reuse the validated path-profile interpolation, not the teacher's
        # velocity/recovery controller. Progress is absolute along the map,
        # including when an evaluation starts midway through the route.
        allowed={'lookahead_m','lateral_offset_m','reference_until_m','reference_fade_m',
                 'after_lookahead_m','reference_sections'}
        if set(reference)-allowed:raise ValueError('Unknown planned waypoint setting')
        lookahead,offset=teacher_reference({'teacher_'+key:value for key,value in reference.items()},progress)
    if not np.isfinite([lookahead,offset,progress,goal_s,yaw,*xy]).all() or lookahead<=0 or abs(offset)>.2:
        raise ValueError('Invalid planned waypoint geometry')
    target_s=min(goal_s,progress+lookahead)
    point=lambda s:np.array([np.interp(s,route[:,0],route[:,j]) for j in (1,2)])
    target=point(target_s)
    if offset and target_s<goal_s:
        tangent=point(min(route[-1,0],target_s+.1))-point(max(route[0,0],target_s-.1))
        tangent/=max(np.linalg.norm(tangent),1e-9)
        # Smoothly return to the original point goal, even for a short stage
        # whose finish lies inside a lateral-reference section.
        offset*=min(1.,max(0.,(goal_s-target_s)/.5))
        target+=offset*np.array([-tangent[1],tangent[0]])
    delta=target-np.asarray(xy)
    angle=np.arctan2(delta[1],delta[0])-yaw
    return np.array([np.linalg.norm(delta),np.sin(angle),np.cos(angle)],np.float32)
