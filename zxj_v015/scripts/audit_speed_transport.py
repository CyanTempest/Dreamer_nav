"""Bind calibrated AppCmd packets to saved teacher physics and measured speed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from _bootstrap import ROOT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_transport_settings(settings):
    if (settings.get('kind') not in ('high_speed_blend_v1', 'high_speed_blend_v2')
            or {key: value for key, value in settings.items() if key != 'kind'} != {
                'reference_mps': .45, 'maximum_mps': .5, 'maximum_gain': 2., 'transport_slew_mps2': .5}):
        raise ValueError('Unrecognized transport calibration')


def audit(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    evidence=json.loads((folder/'calibrated_transport_provenance.json').read_text())
    settings=evidence['settings']
    validate_transport_settings(settings)
    assert len(evidence['packet_logs'])==len(manifest['results'])
    for name,digest in evidence['source_sha256'].items():assert sha(ROOT/name)==digest
    results=[]
    for row,log in zip(manifest['results'],evidence['packet_logs']):
        assert row['teacher'] and row['success'] and not row['collision_contacts']
        assert sha(log['path'])==log['sha256']
        packets=[json.loads(line) for line in Path(log['path']).read_text().splitlines()]
        path=folder/(Path(row['replay_file']).stem+'.physics.npz')
        with np.load(path) as z:
            t=z['time'];q=z['qpos'];v=z['qvel'];commands=z['executed_command']
        assert len(t)==row['transitions'] and row['physics_steps']==(len(t)-1)*100
        np.testing.assert_allclose(np.diff(t),.2,atol=1e-8,rtol=0)
        packet_review = verify_packet_history(packets, t, commands, settings)
        w,x,y,z=q[:,3:7].T;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        speed=v[:,0]*np.cos(yaw)+v[:,1]*np.sin(yaw)
        steady=(commands[:,0]>=.475)&(abs(commands[:,1])<=.05)&(t>=t[0]+2)
        results.append(dict(seed=row['seed'],stage=row['stage'],success=row['success'],**packet_review,
            physical_states=len(t),physics_sha256=sha(path),packet_log_sha256=log['sha256'],
            steady_speed_median_mps=float(np.median(speed[steady])) if np.any(steady) else None))
    return dict(claim='physical_teacher_packet_mapping_and_speed_verified_not_learned_navigation',
        manifest_sha256=sha(folder/'manifest.json'),source_sha256=sha(Path(__file__)),results=results)


def verify_packet_history(packets,t,commands,settings):
    validate_transport_settings(settings)
    previous_value=0.;previous_time=None;active=False;max_mapping_error=0.
    for packet in packets:
        now=packet['sim_time_s'];request=packet['requested_vx_mps']
        assert abs(request)<=.5+1e-6
        dt=0. if previous_time is None else now-previous_time
        assert dt>=-1e-9
        fraction=min(1.,max(0.,(abs(request)-.45)/.05))
        blend=fraction*fraction*(3.-2.*fraction)
        if settings['kind']=='high_speed_blend_v2':
            blend=blend*(2.-blend)
        target=request*(1.+blend)
        if abs(request)<=.450000001 and not active:expected=request
        else:expected=previous_value+min(.5*dt,max(-.5*dt,target-previous_value))
        error=abs(expected-packet['transport_value']);max_mapping_error=max(error,max_mapping_error)
        assert error<1e-8 and abs(target-packet['mapped_target'])<1e-8
        assert abs(packet['transmitted_forward']-expected/2.)<1e-8
        if abs(request)>.01:assert packet['control']==18 and packet['motion']==20
        active=abs(expected-request)>1e-8 or abs(request)>.450000001
        previous_time=now;previous_value=expected
    packet_times=np.array([x['sim_time_s'] for x in packets])
    max_request_difference=0.
    for stamp,command in zip(t,commands):
        index=np.searchsorted(packet_times,stamp-1e-8,side='right')-1
        assert index>=0
        packet=packets[index];lag=stamp-packet['sim_time_s']
        assert -1e-7<=lag<=.05
        requested=np.array([packet['requested_vx_mps'],packet['transmitted_yaw']*2.])
        difference=np.abs(command-requested)
        assert np.all(difference<=np.array([.5,1.])*lag+2e-6)
        max_request_difference=max(max_request_difference,float(difference.max()))
    return dict(packets=len(packets),maximum_mapping_error=max_mapping_error,
        maximum_request_sampling_difference=max_request_difference)


def audit_policy_transport(run,config):
    run=Path(run);metrics=json.loads((run/'metrics.json').read_text())
    evidence=metrics['speed_transport'];assert evidence['settings']==config['environment']['speed_transport']
    path=Path(evidence['packet_log']);packets=[json.loads(line) for line in path.read_text().splitlines()]
    with np.load(run/'physical_states.npz') as z:times=z['time'];commands=z['executed_command']
    result=verify_packet_history(packets,times,commands,evidence['settings'])
    return dict(**result,packet_log=str(path),packet_log_sha256=sha(path),
        requested_and_transmitted_commands_distinguished=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    result=audit(args.folder);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'trials':len(result['results']),'packets':sum(x['packets'] for x in result['results']),'output':str(args.output)}))


if __name__=='__main__':main()
