"""Simulation-only rolling controller for the supplied 16-joint M1 model."""
import numpy as np


class RollingController:
    def __init__(self, model, manifest):
        import mujoco
        self.model = model
        self.radius = manifest["wheel_radius_m"]
        self.gains = manifest["gains"]
        self.targets = manifest["stance_targets_rad"]
        self.yaw_integral = 0.0
        self.joints = []
        for aid in range(model.nu):
            jid = int(model.actuator_trnid[aid, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self.joints.append((name, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))

    def initialize(self, data):
        for name, qi, _ in self.joints:
            if name in self.targets:
                data.qpos[qi] = self.targets[name]

    def command(self, data, vx, wz):
        # Skid steering needs feedback: the geometric wheel-speed split alone
        # substantially understeers with four parallel wheels in frictional contact.
        yaw_error = wz - data.qvel[5]
        self.yaw_integral = float(np.clip(self.yaw_integral + yaw_error * self.model.opt.timestep, -.8, .8))
        steering = np.clip(8.0 * wz + 6.0 * yaw_error + 4.0 * self.yaw_integral, -4.0, 4.0)
        # Track width from URDF wheel centre y coordinates: +/-0.2074 m.
        for aid, (name, qi, vi) in enumerate(self.joints):
            if "FOOT" in name:
                right = name.startswith(("FAR", "RAR"))
                wheel_speed = (vx + (1 if right else -1) * 0.2074 * steering) / self.radius
                torque = self.gains["wheel_kv"] * (wheel_speed - data.qvel[vi])
            else:
                torque = self.gains["leg_kp"] * (self.targets[name] - data.qpos[qi]) - self.gains["leg_kd"] * data.qvel[vi]
            data.ctrl[aid] = np.clip(torque, *self.model.actuator_ctrlrange[aid])
        if not np.isfinite(data.ctrl).all():
            raise ValueError("Nonfinite motor torque")
