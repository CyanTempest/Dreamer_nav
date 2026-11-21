"""Real-time Robot MC adapter with lock-safe pauses between Gym steps.

Requires run_matrix_isolated.sh. The policy is the unmodified vendor runtime;
only desired forward velocity and yaw rate come from the navigation agent.
"""
import ctypes as C
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import yaml

from atec_nav.robots.matrix_app import MatrixApp


class MatrixController:
    radius = .0975
    yaw_integral = 0.

    def __init__(self, model, robot_info, root):
        if os.environ.get('ATEC_MATRIX_ISOLATED') != '1':
            raise RuntimeError('Robot MC requires run_matrix_isolated.sh')
        self.model = model
        self.root = Path(root).resolve()
        self.mc = self.root/'third_party/runtimes/MATRiX_Robot_MC-v0.6.6'
        self.params = yaml.safe_load((self.mc/'build/export/config/zgws/zg_wheels-user-parameters.yaml').read_text())
        self.signs = np.array([self.params[k+'_side_sign'] for k in ['abad','hip','knee','wheel']]).T.ravel()
        self.offsets = np.array([self.params[k+'_offset'] for k in ['abad','hip','knee','wheel']]).T.ravel()
        names = [l+'_'+j+'_JOINT' for l in ['FAR','FBL','RAR','RBL'] for j in ['ABAD','HIP','KNEE','FOOT']]
        jids = np.array([model.joint(n).id for n in names])
        self.qi, self.vi = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
        self.aids = np.array([np.flatnonzero(model.actuator_trnid[:,0] == j)[0] for j in jids])
        self.limits = np.max(np.abs(model.actuator_ctrlrange[self.aids]),axis=1)
        self.lib = lib = C.CDLL(str(self.root/'build/matrix/libatec_matrix_shm.so'))
        arr = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.atec_matrix_create.restype = C.c_void_p
        lib.atec_matrix_error.restype = C.c_char_p
        lib.atec_matrix_close.argtypes = [C.c_void_p]
        lib.atec_matrix_pause.argtypes = [C.c_void_p,C.c_int]
        lib.atec_matrix_write.argtypes = [C.c_void_p]+[arr]*6
        lib.atec_matrix_read.argtypes = [C.c_void_p,arr,np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')]
        self.cmd = np.zeros((16,5))
        self.stamp = np.zeros(16,dtype=np.uint64)
        self.bridge = None
        self.app = None
        self.process = None
        self.paused = False
        self.log = None
        self.pause_count = 0
        self.last_state = None

    def initialize(self, data):
        # Same official-ready stance as the verified supplied-M1 probe.
        for leg in ['FAR','FBL','RAR','RBL']:
            data.qpos[self.model.joint(leg+'_ABAD_JOINT').qposadr] = 0.
            data.qpos[self.model.joint(leg+'_HIP_JOINT').qposadr] = .6 if leg.startswith('F') else -.6
            data.qpos[self.model.joint(leg+'_KNEE_JOINT').qposadr] = -1.2 if leg.startswith('F') else 1.2

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.atec_matrix_error().decode())

    def _feedback(self, data):
        values = [self.signs*data.qpos[self.qi]+self.offsets,
                  self.signs*data.qvel[self.vi],self.signs*data.qfrc_actuator[self.vi],
                  data.sensor('imu_acc').data.copy(),data.sensor('imu_gyro').data.copy(),
                  data.sensor('imu_quat').data[[1,2,3,0]].copy()]
        self._check(self.lib.atec_matrix_write(self.bridge,*[np.ascontiguousarray(v,dtype=np.float64) for v in values]))

    def _read(self):
        if self.process.poll() is not None:
            raise RuntimeError('Robot MC exited; see '+str(self.directory/'controller.log'))
        self._check(self.lib.atec_matrix_read(self.bridge,self.cmd,self.stamp))
        if not np.isfinite(self.cmd).all():
            raise RuntimeError('Nonfinite Robot MC command')

    def _send(self, vx, wz, control=18, motion=20):
        self.app.send(control=control,motion=motion,
                      forward=vx/self.params['cmpc_x_vel'],yaw=wz/self.params['cmpc_yaw_vel'])
        self.last_state = self.app.receive()

    def _torque(self, data):
        q = self.signs*data.qpos[self.qi]+self.offsets
        v = self.signs*data.qvel[self.vi]
        cmd = self.cmd
        tau = self.signs*(cmd[:,2]+cmd[:,3]*(cmd[:,0]-q)+cmd[:,4]*(cmd[:,1]-v))
        data.ctrl[self.aids] = np.clip(tau,-self.limits,self.limits)

    def start(self, data):
        self.close()
        output = self.root/'outputs/matrix_integration/gym_runtime'
        output.mkdir(parents=True,exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='env-',dir=output))
        runtime = self.directory/'runtime'
        (runtime/'mc/bin').mkdir(parents=True)
        for p in (self.mc/'build/export').iterdir():
            if p.name not in ['mc','config']:
                (runtime/p.name).symlink_to(p,target_is_directory=p.is_dir())
        for p in (self.mc/'build/export/mc/bin').iterdir():
            (runtime/'mc/bin'/p.name).symlink_to(p)
        shutil.copytree(self.mc/'build/export/config',runtime/'config')
        defaults = runtime/'config/robot-defaults.yaml'
        settings = yaml.safe_load(defaults.read_text());settings['use_gamepad']=0
        defaults.write_text(yaml.safe_dump(settings))
        try:
            self.bridge = self.lib.atec_matrix_create()
            if not self.bridge:raise RuntimeError(self.lib.atec_matrix_error().decode())
            self.app = MatrixApp()
            self._feedback(data)
            env = os.environ.copy()
            deps = self.mc/'local_deps'
            env.update(ROBOT_TYPE='ZGWS',SDK_CLIENT_IP='127.0.0.1',
                       LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=0',
                       LD_LIBRARY_PATH=':'.join(map(str,[self.mc/'build/export/mc/bin',deps/'usr/lib',deps/'usr/local/lib',deps/'usr/lib/x86_64-linux-gnu'])))
            self.log = (self.directory/'controller.log').open('w')
            self.process = subprocess.Popen(['./mc_ctrl','r'],cwd=runtime/'mc/bin',env=env,
                                            stdout=self.log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            # Four seconds of sensor/transport preparation, then five seconds
            # of physical stand-up and policy settling, exactly as the probe.
            for step in range(4500):
                t=step*.002
                self._feedback(data);self._read()
                if step%20==0:
                    self._send(0.,0.,control=0 if t<=3.5 else (1 if t<5.5 else 18),motion=1 if t<8 else 20)
                if t>=4:
                    if np.max(time.monotonic()*1000-self.stamp.astype(float))>=250:
                        raise RuntimeError('Stale Robot MC during reset')
                    self._torque(data);mujoco.mj_step(self.model,data)
                    if not np.isfinite(data.qpos).all() or data.xmat[1].reshape(3,3)[2,2]<.75:
                        raise RuntimeError('Unstable Robot MC reset')
                time.sleep(max(0.,start+(step+1)*.002-time.monotonic()))
            self.pause(data)
        except BaseException:
            self.close()
            raise

    def resume(self, data, vx, wz):
        if not self.paused:return
        self._feedback(data)
        os.kill(self.process.pid,signal.SIGCONT)
        self.paused=False
        for _ in range(150):
            self._feedback(data);self._send(vx,wz);self._read()
            if np.max(time.monotonic()*1000-self.stamp.astype(float))<10:break
            time.sleep(.002)
        else:raise RuntimeError('Robot MC did not resume fresh commands')
        self.wall_target=time.monotonic()
        self.next_app_send=0.

    def command(self, data, vx, wz):
        self.resume(data,vx,wz)
        self.wall_target+=self.model.opt.timestep
        time.sleep(max(0.,self.wall_target-time.monotonic()))
        self._feedback(data)
        if data.time>=self.next_app_send:
            self._send(vx,wz);self.next_app_send=data.time+.04
        self._read()
        if np.max(time.monotonic()*1000-self.stamp.astype(float))>=250:
            raise RuntimeError('Stale Robot MC commands')
        self._torque(data)

    def pause(self, data):
        if self.process and not self.paused:
            self._feedback(data)
            self._check(self.lib.atec_matrix_pause(self.bridge,self.process.pid))
            self.paused=True
            self.pause_count+=1

    def close(self):
        if self.process and self.process.poll() is None:
            os.kill(self.process.pid,signal.SIGCONT)
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process=None;self.paused=False
        if self.log:self.log.close();self.log=None
        if self.bridge:self.lib.atec_matrix_close(self.bridge);self.bridge=None
        if self.app:self.app.close();self.app=None
