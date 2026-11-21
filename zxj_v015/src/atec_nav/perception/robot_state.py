"""Measured proprioception for the optional physical_v2 navigation contract."""
import mujoco
import numpy as np


class JointContactState:
    """Joint encoders plus four wheel normal loads; no route or terrain oracle."""
    def __init__(self, model):
        self.joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.qpos = model.jnt_qposadr[self.joints]
        self.qvel = model.jnt_dofadr[self.joints]
        self.feet = [model.body(name + '_FOOT_LINK').id
                     for name in ('FAR', 'FBL', 'RAR', 'RBL')]
        if len(self.joints) != 16:
            raise ValueError('physical_v2 requires the original 16-joint M1')
        self.support_load = float(model.body_mass.sum()) * np.linalg.norm(model.opt.gravity) / 4
        self.force = np.zeros(6)

    def observe(self, model, data):
        angle = data.qpos[self.qpos]
        loads = np.zeros(4)
        for i, contact in enumerate(data.contact):
            b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 != 0 and b2 != 0:
                continue
            foot = int(b2 if b1 == 0 else b1)
            if foot in self.feet:
                mujoco.mj_contactForce(model, data, i, self.force)
                loads[self.feet.index(foot)] += max(0., self.force[0])
        # Wheels are continuous joints, so sin/cos avoid encoder wrap jumps.
        return np.concatenate([np.sin(angle), np.cos(angle),
            np.clip(data.qvel[self.qvel] / 20., -5., 5.),
            np.clip(loads / self.support_load, 0., 5.)]).astype(np.float32)
