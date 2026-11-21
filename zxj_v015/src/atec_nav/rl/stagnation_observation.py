"""Observed time without progress; contains no recovery phase or action rule.

The two-action M1 cannot command lateral velocity. With an explicit observation
contract, its formerly constant-zero proprio[10] slot carries this measured age.
"""
import numpy as np


def validate_stagnation_settings(settings):
    if settings is None:return
    if set(settings)!={'scale_s','clip_s'}:
        raise ValueError('Stagnation observation requires scale_s and clip_s only')
    if not np.isfinite(list(settings.values())).all() or min(settings.values())<=0:
        raise ValueError('Stagnation observation times must be positive and finite')


def update_progress_clock(progress,best,last_time,now):
    # This is the same existing clock used by the environment's stall timeout.
    if progress>best+.1:return progress,now
    return best,last_time


def stagnation_feature(now,last_time,settings):
    validate_stagnation_settings(settings)
    if not np.isfinite([now,last_time]).all() or last_time>now+1e-8:
        raise ValueError('Invalid observed progress clock')
    return np.float32(np.clip(now-last_time,0.,settings['clip_s'])/settings['scale_s'])


def recorded_stagnation_features(times,potentials,settings):
    times=np.asarray(times,dtype=float);potentials=np.asarray(potentials,dtype=float)
    if times.ndim!=1 or times.shape!=potentials.shape or not len(times):
        raise ValueError('Aligned nonempty progress and time records required')
    if not np.isfinite(times).all() or not np.isfinite(potentials).all() or np.any(np.diff(times)<0):
        raise ValueError('Recorded progress clock must be finite and monotonic')
    best=potentials[0];last_time=times[0];result=[]
    for now,progress in zip(times,potentials):
        best,last_time=update_progress_clock(progress,best,last_time,now)
        result.append(stagnation_feature(now,last_time,settings))
    return np.asarray(result,np.float32)
