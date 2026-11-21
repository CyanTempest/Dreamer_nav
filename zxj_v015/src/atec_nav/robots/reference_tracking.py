"""Optional smooth local reference for explicitly labelled teacher rollouts."""
import numpy as np


def teacher_reference(stage, progress_from_start):
    lookahead = float(stage.get('teacher_lookahead_m', 1.2))
    offset = float(stage.get('teacher_lateral_offset_m', 0.))
    if 'teacher_reference_until_m' in stage:
        until = float(stage['teacher_reference_until_m'])
        fade = float(stage.get('teacher_reference_fade_m', 1.))
        after = float(stage.get('teacher_after_lookahead_m', 1.2))
        if not np.isfinite([lookahead, offset, until, fade, after]).all() or until < 0 or fade <= 0:
            raise ValueError('Invalid teacher reference transition')
        weight = smooth((until + fade - progress_from_start) / fade)
        lookahead, offset = after + weight*(lookahead-after), weight*offset
    previous_end=-np.inf
    for section in stage.get('teacher_reference_sections',[]):
        start,end,fade=section['start_m'],section['end_m'],section['fade_m']
        length,lateral=section['lookahead_m'],section['lateral_offset_m']
        if (not np.isfinite([start,end,fade,length,lateral]).all() or
                not 0<=start<end or fade<=0 or fade>end-start or start<previous_end or
                length<=0 or abs(lateral)>.2):
            raise ValueError('Invalid or overlapping teacher reference section')
        previous_end=end+fade
        weight=smooth((progress_from_start-start)/fade)*smooth((end+fade-progress_from_start)/fade)
        lookahead+=weight*(length-lookahead)
        offset+=weight*(lateral-offset)
    return lookahead,offset


def smooth(fraction):
    fraction=float(np.clip(fraction,0.,1.))
    return fraction*fraction*(3-2*fraction)


def teacher_curvature_limited(stage,progress_from_start):
    """Optional teacher constraint, scoped to explicit route intervals."""
    return bool(stage.get('teacher_curvature_speed_limit',False) or any(
        section.get('curvature_speed_limit',False) and
        section['start_m']<=progress_from_start<=section['end_m']+section['fade_m']
        for section in stage.get('teacher_reference_sections',[])))


def teacher_speed_limit(stage,progress_from_start,nominal):
    """Smooth optional section speed limits for teacher data collection only."""
    speed=float(nominal)
    if not np.isfinite(speed) or speed<=0:raise ValueError('Invalid teacher nominal speed')
    for section in stage.get('teacher_reference_sections',[]):
        if 'max_speed_mps' not in section:continue
        limit=float(section['max_speed_mps'])
        if not np.isfinite(limit) or not 0<limit<=.45:raise ValueError('Invalid teacher section speed limit')
        start,end,fade=section['start_m'],section['end_m'],section['fade_m']
        if fade<=0 or not 0<=start<end:raise ValueError('Invalid teacher speed interval')
        weight=smooth((progress_from_start-start)/fade)*smooth((end+fade-progress_from_start)/fade)
        speed+=weight*(min(speed,limit)-speed)
    return speed
