"""Torque-only slope support extension of the M1 rolling engineering baseline.

This controller is not a stair gait and never changes the free-base state.
Wheel PI rejects sustained slope loads; stance feedforward balances gravity.
"""
import mujoco
import numpy as np

from atec_nav.robots.rolling import RollingController


class TerrainRollingController(RollingController):
    def __init__(self, model, manifest):
        super().__init__(model, manifest)
        self.wheel_integrals = np.zeros(model.nu)
        self.mass = float(model.body_mass.sum())
        self.wheel_bodies = [model.body(name+'_FOOT_LINK').id for name in ('FAR','FBL','RAR','RBL')]
        self.jac = np.zeros((3,model.nv))
        self.jacr = np.zeros_like(self.jac)
        self.unloaded_body = None
        self.body_target = None

    def initialize(self, data):
        super().initialize(data)
        self.wheel_integrals[:] = 0
        self.yaw_integral = 0.

    def command(self, data, vx, wz):
        super().command(data,vx,wz)
        yaw_error=wz-data.qvel[5]
        steering=np.clip(8*wz+6*yaw_error+4*self.yaw_integral,-4,4)
        contacted=set()
        for contact in data.contact:
            if contact.dist > .003:
                continue
            bodies=[int(self.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
            if 0 in bodies:
                contacted.update(b for b in bodies if b in self.wheel_bodies and b != self.unloaded_body)
        # Apply a measured-model gravity load only through supported wheel legs.
        feedforward=np.zeros(self.model.nv)
        if contacted:
            body_ids=sorted(contacted)
            xy=data.xpos[body_ids,:2]-data.subtree_com[1,:2]
            balance=np.vstack([np.ones(len(body_ids)),xy.T])
            weight=self.mass*abs(self.model.opt.gravity[2])
            loads=np.linalg.lstsq(balance,np.array([weight,0.,0.]),rcond=None)[0]
            loads=np.maximum(loads,0);loads*=weight/max(loads.sum(),1e-9)
            forces=np.column_stack([np.zeros((len(loads),2)),loads])
            if self.body_target is not None:
                target_position,target_rotation=self.body_target
                rotation=data.xmat[1].reshape(3,3)
                acceleration=15*(target_position-data.qpos[:3])-6*data.qvel[:3]
                acceleration=np.clip(acceleration,-3,3)
                orientation_error=.5*sum(np.cross(rotation[:,j],target_rotation[:,j]) for j in range(3))
                moment=100*orientation_error-12*(rotation@data.qvel[3:6])
                blocks=[]
                for body in body_ids:
                    x,y,z=data.xpos[body]-data.subtree_com[1]
                    blocks.append(np.vstack([np.eye(3),np.array([[0,-z,y],[z,0,-x],[-y,x,0]])]))
                wrench=np.r_[self.mass*(acceleration-self.model.opt.gravity),moment]
                forces=np.linalg.lstsq(np.concatenate(blocks,axis=1),wrench,rcond=None)[0].reshape(-1,3)
                forces[:,2]=np.maximum(forces[:,2],0)
                forces[:,:2]=np.clip(forces[:,:2],-.7*forces[:,2,None],.7*forces[:,2,None])
            for body,force in zip(body_ids,forces):
                mujoco.mj_jacBody(self.model,data,self.jac,self.jacr,body)
                feedforward-=self.jac.T@force
        for aid,(name,qi,vi) in enumerate(self.joints):
            if 'FOOT' in name:
                right=name.startswith(('FAR','RAR'))
                target=(vx+(1 if right else -1)*.2074*steering)/self.radius
                error=target-data.qvel[vi]
                self.wheel_integrals[aid]=np.clip(self.wheel_integrals[aid]+error*self.model.opt.timestep,-2,2)
                torque=data.ctrl[aid]+8*self.wheel_integrals[aid]
            else:
                torque=data.ctrl[aid]+feedforward[vi]+data.qfrc_bias[vi]
            data.ctrl[aid]=np.clip(torque,*self.model.actuator_ctrlrange[aid])
