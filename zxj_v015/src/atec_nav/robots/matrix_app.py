"""Local Robot MC v0.6.6 AppCmd/AppState transport using its bundled eCAL C API."""
import ctypes as C
import os
import time
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def message_types():
    # Wire fields from the release's deps/usr/include/robot_sdk.pb.h.
    schema=descriptor_pb2.FileDescriptorProto(name='atec_matrix_app.proto',package='robot_sdk.pb',syntax='proto3')
    cmd=schema.message_type.add(name='AppCmd')
    for name,number,kind in [('vx',3,2),('vy',4,2),('yaw_rate',5,2),('pitch_rate',6,2),
                             ('control_mode',11,13),('motion_mode',12,13),('time_stamp',13,4)]:
        cmd.field.add(name=name,number=number,type=kind,label=1)
    state=schema.message_type.add(name='AppState')
    for name,number,kind,label in [('cur_ctrl_mode',3,13,1),('cur_motion_mode',4,13,1),
                                   ('position',11,2,3),('time_stamp',13,4,1)]:
        state.field.add(name=name,number=number,type=kind,label=label)
    pool=descriptor_pool.DescriptorPool();pool.Add(schema)
    return tuple(message_factory.GetMessageClass(pool.FindMessageTypeByName('robot_sdk.pb.'+n))
                 for n in ['AppCmd','AppState'])


class MatrixApp:
    def __init__(self):
        if os.environ.get('ATEC_MATRIX_ISOLATED')!='1':raise RuntimeError('Simulation namespace required')
        self.Command,self.State=message_types()
        self.lib=lib=C.CDLL('libecal_core_c.so.5')
        lib.eCAL_Initialize.argtypes=[C.c_int,C.c_void_p,C.c_char_p,C.c_uint]
        if lib.eCAL_Initialize(0,None,b'atec_sim_navigation',0x83)<0:raise RuntimeError('eCAL init failed')
        for kind in ['Pub','Sub']:
            getattr(lib,'eCAL_'+kind+'_New').restype=C.c_void_p
            getattr(lib,'eCAL_'+kind+'_Create').argtypes=[C.c_void_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_int]
            getattr(lib,'eCAL_'+kind+'_Destroy').argtypes=[C.c_void_p]
        lib.eCAL_Pub_Send.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_longlong]
        lib.eCAL_Pub_IsSubscribed.argtypes=[C.c_void_p]
        lib.eCAL_Sub_Receive_ToBuffer.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.POINTER(C.c_longlong),C.c_int]
        self.pub=lib.eCAL_Pub_New();self.sub=lib.eCAL_Sub_New()
        if not lib.eCAL_Pub_Create(self.pub,b'app_cmd',b'proto:robot_sdk.pb.AppCmd',b'',0):raise RuntimeError('AppCmd creation failed')
        if not lib.eCAL_Sub_Create(self.sub,b'app_state',b'proto:robot_sdk.pb.AppState',b'',0):raise RuntimeError('AppState creation failed')
        self.buffer=C.create_string_buffer(65536)
        self.last_state=None

    def send(self, control=18, motion=1, forward=0., yaw=0.):
        cmd=self.Command(vx=float(forward),yaw_rate=float(yaw),control_mode=control,motion_mode=motion,
                         time_stamp=int(time.monotonic()*1000))
        payload=cmd.SerializeToString()
        return self.lib.eCAL_Pub_Send(self.pub,payload,len(payload),-1)

    def receive(self):
        stamp=C.c_longlong()
        n=self.lib.eCAL_Sub_Receive_ToBuffer(self.sub,self.buffer,len(self.buffer),C.byref(stamp),0)
        if n>0:
            msg=self.State();msg.ParseFromString(self.buffer.raw[:n])
            self.last_state={'control':msg.cur_ctrl_mode,'motion':msg.cur_motion_mode,
                             'position':list(msg.position),'stamp':msg.time_stamp}
        return self.last_state

    def close(self):
        self.lib.eCAL_Pub_Destroy(self.pub);self.lib.eCAL_Sub_Destroy(self.sub);self.lib.eCAL_Finalize(0)
