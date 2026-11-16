"""Read-only evidence at the eight old terminal-descent failures."""
import json
from pathlib import Path
import numpy as np
import shapely as sh
from .lwy_import import Route
from .lwy_geometry import Geometry
from .lwy_audit import sample


def main():
    targets = {630001,630006,630008,630011,630013,630014,630016,630019}
    source = json.loads(Path('metrics/lwy_g1_blind_dynamic.json').read_text())
    rows = []
    xx, yy = np.meshgrid(np.linspace(-.515,.515,43),np.linspace(-.29,.29,25),indexing='ij')
    foot = np.stack([xx,yy],-1).reshape(-1,2)
    for endpoint in source['endpoints']:
        seed = int(endpoint['map'].split('_')[-1])
        if seed not in targets: continue
        directory = Path('maps_lwy')/endpoint['map']
        spec = json.loads((directory/'spec.json').read_text())
        bundle = json.loads((directory/'scene/scene_bundle.json').read_text())
        route = Route(spec); g = Geometry(bundle,route); s=endpoint['s']
        ts, ns = route.frame(s); yaw = np.arctan2(ts[1],ts[0])
        around = np.arange(s-2,s+2.001,.2)
        yaws=np.unwrap([np.arctan2(*route.frame(float(x))[0][1::-1]) for x in around])
        zone = next(z for z in spec['track']['zones'] if z['start_m'] <= s <= z['end_m'])
        center=route.position(s)[:2]
        nearest = g.block_tree.query(sh.Point(center).buffer(2))
        props=[g.props[g.block_ids[i]] for i in nearest]
        margins=[]
        data=dict(np.load(directory/'map.npz'))
        for lat in [-.17,-.10,0,.10,.17]:
            xy=center+ns[:2]*lat
            rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            raster=sample(data,xy+foot@rot.T)
            exact=g.clearance(g.footprint(xy,yaw))
            margins.append(dict(lateral_m=lat,exact_support_m=exact[0],exact_obstacle_m=exact[1],
                nearest=exact[3],raster_support_m=float(.025-raster[:,2].max()),raster_obstacle_m=float(raster[:,5].min())))
        row=dict(**endpoint,seed=seed,zone=zone,flights=[f for f in route.stairs if f['start']-2<=s<=f['end']+2],
            zone_transition_distance_m=min(abs(s-z['start_m']) for z in spec['track']['zones']),
            width_range_m=[route.width(float(x)) for x in [s-2,s,s+2]],
            curvature_max_rad_per_m=float(np.max(np.abs(np.diff(yaws)/.2))),
            actual_lateral_m=float((np.asarray(endpoint['xy'])-center)@ns[:2]),
            near_primitives=props,cross_section=margins,
            centerline_samples=[dict(s=float(x),z=route.height(float(x)),width=route.width(float(x))) for x in around])
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['seed','s','flights','zone_transition_distance_m','width_range_m','curvature_max_rad_per_m','actual_lateral_m']}),flush=True)
    with Path('metrics/lwy_v2_geometry_diagnosis.json').open('x') as f: json.dump(dict(rows=rows),f,indent=2)


if __name__=='__main__':main()
