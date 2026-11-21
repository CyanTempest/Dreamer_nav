"""Explicit experimental AppCmd calibration for the requested faster cruise.

Requests through 0.45 m/s retain the original mapping before fast operation.
The added high-speed range has its own bounded transport ramp and packet log.
This changes the low-level command mapping, not poses, contacts or torques.
"""
import json
import numpy as np
from atec_nav.robots.matrix_controller import MatrixController


SETTINGS = dict(kind='high_speed_blend_v1', reference_mps=.45, maximum_mps=.5,
                maximum_gain=2., transport_slew_mps2=.5)
FAST_APPROACH_SETTINGS = {**SETTINGS, 'kind': 'high_speed_blend_v2'}


def calibrated_target(requested, settings):
    if settings not in (SETTINGS, FAST_APPROACH_SETTINGS):
        raise ValueError('Unvalidated transport calibration contract')
    if not np.isfinite(requested) or abs(requested)>settings['maximum_mps']+1e-6:
        raise ValueError('Request outside the calibrated speed range')
    fraction=float(np.clip((abs(requested)-settings['reference_mps']) /
                           (settings['maximum_mps']-settings['reference_mps']),0.,1.))
    blend=fraction*fraction*(3-2*fraction)
    if settings == FAST_APPROACH_SETTINGS:
        # Reach more of the calibrated gain near cruise without raising its
        # peak or changing the mapping used for slow stair recovery.
        blend = 1. - (1. - blend)**2
    return requested*(1.+blend*(settings['maximum_gain']-1.))


class CalibratedMatrixController(MatrixController):
    def __init__(self, model, robot_info, root, settings):
        calibrated_target(0., settings)
        self.transport_settings=dict(settings)
        self.transport_log=None
        self.transport_paths=[]
        self.transport_value=0.
        self.transport_time=None
        self.feedback_time=0.
        self.fast_transport_active=False
        super().__init__(model,robot_info,root)

    def start(self,data):
        self.transport_value=0.;self.transport_time=None;self.fast_transport_active=False
        return super().start(data)

    def _feedback(self,data):
        self.feedback_time=float(data.time)
        return super()._feedback(data)

    def _send(self,vx,wz,control=18,motion=20):
        target=calibrated_target(vx,self.transport_settings)
        dt=0. if self.transport_time is None else max(0.,self.feedback_time-self.transport_time)
        if abs(vx)<=self.transport_settings['reference_mps']+1e-9 and not self.fast_transport_active:
            value=float(vx)
        else:
            bound=self.transport_settings['transport_slew_mps2']*dt
            value=self.transport_value+float(np.clip(target-self.transport_value,-bound,bound))
        self.fast_transport_active=abs(value-vx)>1e-8 or abs(vx)>self.transport_settings['reference_mps']+1e-9
        self.transport_value=value;self.transport_time=self.feedback_time
        forward=value/self.params['cmpc_x_vel']
        if abs(forward)>1:raise ValueError('AppCmd calibration exceeded normalized packet bounds')
        if self.transport_log is None:
            path=self.directory/'speed_transport.jsonl'
            self.transport_log=path.open('x');self.transport_paths.append(path)
        self.transport_log.write(json.dumps(dict(sim_time_s=self.feedback_time,requested_vx_mps=float(vx),
            mapped_target=target,transport_value=value,transmitted_forward=forward,
            transmitted_yaw=float(wz/self.params['cmpc_yaw_vel']),control=control,motion=motion))+'\n')
        result=super()._send(value,wz,control,motion)
        if self.last_state is not None:
            self.last_state={**self.last_state,'atec_transport':dict(settings=self.transport_settings,
                packet_log=str(self.transport_paths[-1]),requested_vx_mps=float(vx),transmitted_forward=forward)}
        return result

    def close(self):
        try:return super().close()
        finally:
            if self.transport_log is not None:self.transport_log.close();self.transport_log=None
