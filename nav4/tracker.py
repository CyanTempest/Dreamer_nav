"""The single 20 Hz target-tracking law (env substep, planner rollout and support filter all use it).

command(): local target + speed -> body velocity command
    v_xy = target/|target| * min(speed, |target|),   |target| floored at tracker.norm_floor
    w    = clip(yaw_gain * speed * target_y / |target|^2, +-max_yaw_rate)
substep(): the command enters a queue and acts `delay` substeps later, scaled by the motor gain,
through a first-order response (motor_tau) with acceleration limits; pose integrates the new velocity.
"""
import jax.numpy as jnp

from .common import control_dt, rotate, wrap


def command(local, speed, cfg):
    t = cfg['tracker']
    norm = jnp.maximum(jnp.linalg.norm(local, axis=-1), t['norm_floor'])
    xy = local/norm[..., None]*jnp.minimum(speed, norm)[..., None]
    w = jnp.clip(t['yaw_gain']*speed*local[..., 1]/norm**2, -t['max_yaw_rate'], t['max_yaw_rate'])
    return jnp.concatenate([xy, w[..., None]], -1)


def substep(pos, yaw, vel, queue, target, speed, delay, gain, cfg):
    """One control period. queue[..., 0, :] is the newest command; delay (int) indexes into it."""
    t, dt = cfg['tracker'], control_dt(cfg)
    queue = jnp.concatenate([command(rotate(target-pos, -yaw), speed, cfg)[..., None, :], queue[..., :-1, :]], -2)
    delayed = jnp.take_along_axis(queue, delay[..., None, None], -2)[..., 0, :]
    limit = jnp.array([t['max_accel'], t['max_accel'], t['max_yaw_accel']])*dt
    vel = vel+jnp.clip((delayed*gain[..., None]-vel)*(dt/t['motor_tau']), -limit, limit)
    return pos+rotate(vel[..., :2], yaw)*dt, wrap(yaw+vel[..., 2]*dt), vel, queue


def delay_steps(seconds, cfg):
    return int(round(seconds/control_dt(cfg)))
