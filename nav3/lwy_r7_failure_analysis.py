"""Full-rate terminal histories and unchanged terrain limits; diagnostic only."""
import json
from pathlib import Path
import numpy as np
from .lwy_obstacles_v2 import Raster,rotate


def main():
    rows=[];cache={}
    for dataset in ['original','blind']:
        d=json.loads(Path(f'metrics/lwy_r7_final_dreamer_lwy_{dataset}_ob0_sf0.json').read_text())
        for r in d['rows']:
            if r['success']:continue
            if r['map'] not in cache:cache[r['map']]=Raster(Path('maps_lwy_v2')/r['map']/'map.npz')
            raster=cache[r['map']];a=raster.a;pos=np.array(r['terminal_xy']);yaw=r['terminal_yaw']
            index=((a['route'][:,:2]-pos)**2).sum(-1).argmin();road_yaw=a['yaw'][index]
            lateral=float(rotate(pos-a['route'][index,:2],-road_yaw)[1])
            points=pos+rotate(np.array([[0.,0.],[.4,0.],[-.42,0.],[.42,0.]]),yaw)
            samples=raster.sample(points)
            history=np.array(r['terminal_history_5hz']);actions=np.clip(history[:,-3:],-1,1)
            angle=actions[:,1]*np.pi/3;radius=.5+(actions[:,0]+1)*1.25
            record=dict(dataset=dataset,map=r['map'],seed=r['noise_seed'],failure=r['failure'],section=r['section'],
                s=r['progress_m'],lateral_m=lateral,width_m=float(a['width'][index]),
                terminal_target_body_m=[float(radius[-1]*np.cos(angle[-1])),float(radius[-1]*np.sin(angle[-1]))],
                requested_speed_mps=float((actions[-1,2]+1)*.4),terminal_velocity_body=r['terminal_velocity_body'],
                realized_speed_mps=float(np.linalg.norm(r['terminal_velocity_body'][:2])),
                near_root=bool(np.any(samples[:2,6]==2)),steep_height_difference_m=float(abs(samples[3,0]-samples[2,0])),
                history_seconds=len(history)*.2,history_progress_m=float(history[-1,1]-history[0,1]),
                steer_std_degrees=float(np.std(angle)*180/np.pi),
                steer_sign_change_fraction=float(np.mean(angle[1:]*angle[:-1]<0)),
                history_requested_speed_mean_mps=float(np.mean((actions[:,2]+1)*.4)),
                reward_component_sums=r['reward_component_sums'])
            rows.append(record)
    result=dict(rows=rows,notes=['Steering sign changes reflect stochastic actions, not proof of a deterministic limit cycle.',
        'Terrain sampling here is a terminal nearest-raster diagnostic; the actual steep-speed limiter uses linear ground interpolation.',
        'Targets decode clipped actions, as in the unchanged physical environment.'])
    with Path('metrics/lwy_r7_failure_analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({ds:{f:sum(r['dataset']==ds and r['failure']==f for r in rows) for f in ['support_or_drop','collision','stuck_or_timeout']} for ds in ['original','blind']}))


if __name__=='__main__':main()
