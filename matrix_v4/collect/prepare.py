"""Prepare an instance inside the temporary VM. Replace files atomically to break hardlinks."""
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np


def vector(a):
    return ' '.join(f'{v:.12g}' for v in a)


def configure(instance, window, pose):
    model=instance/'UeSim/Content/model'
    ext=instance/'External/zgws'
    ext.mkdir(parents=True,exist_ok=True)
    for path in (model/'zgws').iterdir():
        target=ext/path.name
        if not target.exists():
            target.symlink_to(path.resolve())
    # Only the selected reference and robot home differ from the release.
    entry=ext/'zgws.xml'
    entry.unlink(missing_ok=True)
    shutil.copyfile(model/'zgws'/f'zgws_atec_{window}.xml',entry)
    filename=f'zgws_atec_robot_{window}.xml'
    xml=ET.parse(model/'zgws'/filename)
    q=np.array(pose['qpos'])
    body=xml.find("./worldbody/body[@name='base_link']")
    body.set('pos',vector(q[:3]))
    body.set('quat',vector(q[3:7]))
    legs=[b for b in body.iter('body') if b.find('joint') is not None]
    if len(legs)!=len(q)-7:
        raise ValueError('Unexpected ZGWS joint count')
    for b,angle in zip(legs,q[7:]):
        j=b.find('joint')
        half=(angle-float(j.get('ref','0')))/2
        old=np.fromstring(b.get('quat','1 0 0 0'),sep=' ')
        axis=np.fromstring(j.get('axis'),sep=' ')
        xyz=np.sin(half)*axis
        new=np.r_[old[0]*math.cos(half)-old[1:]@xyz,
                  old[0]*xyz+old[1:]*math.cos(half)+np.cross(old[1:],xyz)]
        b.set('quat',vector(new))
        j.set('ref',str(angle))
    key=xml.find("./keyframe/key[@name='home']")
    key.set('qpos',vector(q))
    torque=np.clip(190*(np.array(pose['pd_joint_target'])-q[7:]),-140,140)
    torque[3::4]=np.clip(torque[3::4],-38,38)
    key.set('ctrl',vector(torque))
    temp=ext/'robot.tmp'
    xml.write(temp,encoding='utf-8')
    temp.replace(ext/filename)
    path=model/'config/config.json'
    cfg=json.loads(path.read_text())
    robot=cfg['robot']
    robot.update(mujoco_model=str(entry),position=dict(zip('xyz',q[:3])),
                 zenoh_router='tcp/127.0.0.1:7447')
    sensors=robot['sensors']
    if sorted(v['sensor_type'] for v in sensors.values())!=['airy','airy','imu','odom']:
        raise ValueError('Expected vendor dual Airy plus IMU and Odom')
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(cfg))
    temp.replace(path)


if __name__=='__main__':
    instance,bank,seed,meta=sys.argv[1:]
    p=Path(bank)
    poses=json.loads(p.read_text())
    idx=int(np.random.default_rng(int(seed)).integers(len(poses)))
    configure(Path(instance),p.stem,poses[idx])
    Path(meta).write_text(json.dumps(dict(poses[idx],seed=int(seed),bank_index=idx,window=p.stem)))
