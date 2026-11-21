"""Training-only clearance estimate; strict MuJoCo contacts remain the gate.

Obstacle distance uses convex geom queries. Ground clearance is vertical
distance at sampled body surfaces, not exact Euclidean separation of meshes.
No clearance value is exposed as a sensor observation to the navigation agent.
"""
import itertools
import mujoco
import numpy as np
from scipy.spatial import cKDTree, ConvexHull


class BodyClearance:
    def __init__(self, model, margin=.04, include_foot_obstacles=False):
        if not 0 < margin <= .2:
            raise ValueError('Clearance margin must be in (0, .2] metres')
        self.margin = float(margin)
        self.ground = model.geom('road_collision').id
        self.field = int(model.geom_dataid[self.ground])
        self.top = model.geom_pos[self.ground, 2] + model.hfield_size[self.field, 2] + 1.
        self.obstacles = np.array([g for g in range(model.ngeom)
                                  if model.geom(g).name.startswith('obstacle_')], dtype=int)
        self.tree = cKDTree(model.geom_pos[self.obstacles]) if len(self.obstacles) else None
        self.obstacle_radius = float(model.geom_rbound[self.obstacles].max()) if len(self.obstacles) else 0.
        self.samples = {}
        self.obstacle_geoms = []
        for g in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[g])).name
            if model.geom_bodyid[g] == 0 or not model.geom_contype[g]:
                continue
            if 'FOOT_LINK' in body:
                if include_foot_obstacles:self.obstacle_geoms.append(g)
                continue
            self.obstacle_geoms.append(g)
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                points = np.array(list(itertools.product((-1., 0., 1.), repeat=3))) * size
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                angles = np.arange(16) * (2*np.pi/16)
                points = np.array([[size[0]*np.cos(a), size[0]*np.sin(a), z]
                                   for z in (-size[1], 0., size[1]) for a in angles])
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]; start = model.mesh_vertadr[mesh]
                points = model.mesh_vert[start:start+model.mesh_vertnum[mesh]].copy()
                # MuJoCo collides ordinary meshes as convex hulls. Keep every
                # hull vertex and triangle centroid; omit interior vertices.
                hull = ConvexHull(points)
                points = np.concatenate([points[hull.vertices], points[hull.simplices].mean(1)])
            else:
                raise ValueError('Unsupported M1 clearance geom type: ' + str(kind))
            self.samples[g] = points
        self.direction = np.array([0., 0., -1.])

    def measure(self, model, data):
        road, obstacle = self.margin, self.margin
        for g, local in self.samples.items():
            position = data.geom_xpos[g]
            points = local @ data.geom_xmat[g].reshape(3, 3).T + position
            for p in points:
                # Cast only through the physical heightfield, excluding all
                # robot and visual proxy geometry from the query.
                origin = np.array([p[0], p[1], self.top])
                distance = mujoco.mj_rayHfield(model, data, self.ground, origin, self.direction)
                if distance >= 0:
                    road = min(road, float(p[2] - (self.top-distance)))
        if self.tree is not None:
            for g in self.obstacle_geoms:
                position = data.geom_xpos[g]
                radius = float(model.geom_rbound[g]) + self.obstacle_radius + self.margin
                for index in self.tree.query_ball_point(position, radius):
                    value = mujoco.mj_geomDistance(model, data, g, int(self.obstacles[index]), self.margin, None)
                    obstacle = min(obstacle, float(value))
        return {'road_sampled_m':road, 'obstacle_convex_m':obstacle,
                'minimum_m':min(road, obstacle)}


def clearance_cost(clearance, margin):
    if not np.isfinite(clearance) or not margin > 0:
        raise ValueError('Invalid clearance cost input')
    return float(np.clip((margin-clearance) / margin, 0., 1.) ** 2)
