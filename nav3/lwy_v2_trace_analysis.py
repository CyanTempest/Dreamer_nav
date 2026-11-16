"""Relate sampled-planner decisions to drift and first failing 20 Hz footprint."""
import argparse
import json
from pathlib import Path
import numpy as np
from .lwy_import import Route
from .lwy_geometry import Geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='v2_probe_baseline_run');a=p.parse_args()
    result=json.loads(Path(f'metrics/lwy_{a.tag}.json').read_text())
    frames=dict(np.load(f'metrics/lwy_{a.tag}.npz'));rows=[]
    for i,episode in enumerate(result['rows']):
        ids=np.flatnonzero(frames['episode']==i)
        if not len(ids):
            rows.append(dict(**episode,recorded=False,reason='episode ended before diagnostic recording window'))
            continue
        directory=Path(result['arguments']['root'])/episode['map']
        spec=json.loads((directory/'spec.json').read_text());route=Route(spec)
        ss=frames['s'][ids];normals=np.array([route.frame(float(s))[1][:2] for s in ss])
        centers=np.array([route.position(float(s))[:2] for s in ss])
        lateral=((frames['pos'][ids]-centers)*normals).sum(-1)
        route_yaw=np.array([np.arctan2(*route.frame(float(s))[0][1::-1]) for s in ss])
        yaw_error=(frames['yaw'][ids]-route_yaw+np.pi)%(2*np.pi)-np.pi
        drift=(frames['drift'][ids,:2]*normals).sum(-1)
        vel=frames['sub_vel'][ids,:,:2];cmd=frames['sub_cmd'][ids,:,:2]
        failure=None
        bad=(frames['sub_support'][ids]<0)|(frames['sub_obstacle'][ids]<0)
        if bad.any():
            tick,sub=np.argwhere(bad)[0];idx=ids[tick]
            xy=frames['sub_pos'][idx,sub];yaw=frames['sub_yaw'][idx,sub]
            bundle=json.loads((directory/'scene/scene_bundle.json').read_text());g=Geometry(bundle,route)
            exact=g.clearance(g.footprint(xy,yaw))
            failure=dict(tick=int(frames['tick'][idx]),substep=int(sub),s=float(ss[tick]),
                xy=xy.tolist(),yaw=float(yaw),lateral_m=float(lateral[tick]),drift_lateral_m=float(drift[tick]),
                yaw_error_rad=float(yaw_error[tick]),
                selected_offset_m=float(frames['offset'][idx]),risk=float(frames['risk'][idx]),border=float(frames['border'][idx]),
                exact_support_m=exact[0],exact_obstacle_m=exact[1],exact_feasible=exact[2],nearest=exact[3],
                raster_support_m=float(frames['sub_support'][idx,sub]),raster_obstacle_m=float(frames['sub_obstacle'][idx,sub]),
                action=frames['action'][idx].tolist(),vel=frames['sub_vel'][idx,sub].tolist(),cmd=frames['sub_cmd'][idx,sub].tolist())
        row=dict(**episode,max_abs_lateral_m=float(np.max(np.abs(lateral))),
            max_abs_drift_lateral_m=float(np.max(np.abs(drift))),
            tracking_velocity_error_p95_mps=float(np.quantile(np.linalg.norm(vel-cmd,axis=-1),.95)),
            chosen_offset_range_m=[float(frames['offset'][ids].min()),float(frames['offset'][ids].max())],
            mean_actual_speed_mps=float(np.linalg.norm(vel,axis=-1).mean()),first_bad_substep=failure)
        rows.append(row)
    output=Path(f'metrics/lwy_{a.tag}_analysis.json')
    with output.open('x') as f:json.dump(dict(rows=rows,physics_trace_endpoint_error_m=result['physics_trace_endpoint_error_m']),f,indent=2)
    print(json.dumps(dict(rows=rows)),flush=True)


if __name__=='__main__':main()
