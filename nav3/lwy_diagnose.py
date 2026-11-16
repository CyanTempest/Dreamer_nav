"""Evidence for the narrow-lane planner regression; no production mutation."""
import json
from pathlib import Path
import numpy as np
from .lwy_import import Route
from .lwy_geometry import Geometry
from .lwy_audit import sample


def main():
    spec = json.loads(Path('../../ATEC_Env/config/track_spec.json').read_text())
    bundle = json.loads(Path('../../ATEC_Env/scene/scene_bundle.json').read_text())
    route = Route(spec); geometry = Geometry(bundle, route)
    data = dict(np.load('maps_lwy/test_862428/map.npz'))
    xx, yy = np.meshgrid(np.linspace(-.515, .515, 43), np.linspace(-.29, .29, 25), indexing='ij')
    foot = np.stack([xx, yy], -1).reshape(-1, 2)
    result = []
    original = json.loads(Path('metrics/lwy_g1_original_repeat.json').read_text())
    for endpoint in original['endpoints']:
        if endpoint['s'] >= 999: continue
        s = endpoint['s']; xy = np.array(endpoint['xy']); yaw = endpoint['yaw']
        center = route.position(s)[:2]; tangent, normal = route.frame(s)
        route_yaw = np.arctan2(tangent[1], tangent[0])
        exact = geometry.clearance(geometry.footprint(xy, yaw))
        on_center = geometry.clearance(geometry.footprint(center, route_yaw))
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        raster = sample(data, xy+foot@rot.T)
        result.append(dict(**endpoint, lateral_m=float((xy-center)@normal[:2]),
            yaw_error_rad=float((yaw-route_yaw+np.pi)%(2*np.pi)-np.pi),
            original_support_margin_m=exact[0], original_obstacle_margin_m=exact[1],
            original_geometry_free=exact[2], centerline_geometry_free=on_center[2],
            centerline_margin_m=min(on_center[:2]), raster_support_margin_m=float(.025-raster[:, 2].max()),
            nearest_primitive=exact[3]))
    out = dict(rows=result, intervention='Only replace planner 3x3 observed-obstacle max pooling with identity; physical checker unchanged',
        old_single_command='python -m nav3.lwy_eval --start 900 --episodes 1 --seed-start 940004 --tag diag_single_legacy --assert-min-progress 999',
        hypotheses=['redundant observed-obstacle dilation pushes lane toward edge', 'localization drift alone', 'import/projection error'])
    with Path('metrics/lwy_planner_diagnosis.json').open('x') as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
