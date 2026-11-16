"""Distinguish G1's conservative support termination from original contact."""
import json
from pathlib import Path
import numpy as np
from .lwy_import import Route
from .lwy_geometry import Geometry


def main():
    rows = []; cached = {}
    for tag in ['g1_original_dynamic_repeat', 'g1_blind_dynamic']:
        data = json.loads(Path(f'metrics/lwy_{tag}.json').read_text())
        for endpoint in data['endpoints']:
            name = endpoint['map']
            if name not in cached:
                directory = Path('../../ATEC_Env') if name == 'test_862428' else Path('maps_lwy')/name
                spec_path = directory/'config/track_spec.json' if name == 'test_862428' else directory/'spec.json'
                spec = json.loads(spec_path.read_text()); route = Route(spec)
                bundle = json.loads((directory/'scene/scene_bundle.json').read_text())
                cached[name] = Geometry(bundle, route)
            geometry = cached[name]
            foot = geometry.footprint(endpoint['xy'], endpoint['yaw'])
            edge, ob, feasible, prop = geometry.clearance(foot)
            center = geometry.route.position(endpoint['s'])
            tangent, normal = geometry.route.frame(endpoint['s'])
            route_yaw = np.arctan2(tangent[1], tangent[0])
            ce, co, cf, _ = geometry.clearance(geometry.footprint(center[:2], route_yaw))
            rows.append(dict(**endpoint, support_margin_m=edge, obstacle_margin_m=ob,
                original_geometry_free=feasible, nearest_primitive=prop,
                centerline_geometry_free=cf, centerline_margin_m=min(ce, co),
                lateral_m=float((np.asarray(endpoint['xy'])-center[:2])@normal[:2])))
    with Path('metrics/lwy_endpoint_geometry.json').open('x') as f:
        json.dump(dict(rows=rows, original_geometry_free_at_termination=sum(r['original_geometry_free'] for r in rows)), f, indent=2)
    print(json.dumps(dict(endpoint_audit_rows=len(rows), original_geometry_free_at_termination=sum(r['original_geometry_free'] for r in rows))), flush=True)


if __name__ == '__main__': main()
