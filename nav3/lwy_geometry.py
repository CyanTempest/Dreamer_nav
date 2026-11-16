"""Original LWY surface triangles and conservative planar primitive projections.

The only joined seams are <= 60 micrometre numerical gaps between adjoining
road panels. Banks never provide support. Coordinates always remain world XY.
"""
import numpy as np
import shapely as sh
from shapely import affinity
from shapely.strtree import STRtree

SEAM = 3e-5


def primitive_polygon(p):
    if p['type'] == 'capsule':
        ends = np.asarray(p['fromto']).reshape(2, 3)
        xy = ends[:, :2]
        line = sh.Point(xy[0]) if np.linalg.norm(xy[1]-xy[0]) < 1e-10 else sh.LineString(xy)
        return line.buffer(p['radius'], quad_segs=32)
    center = np.asarray(p['position'])
    size = np.asarray(p['size'])
    if p['type'] == 'ellipsoid':
        g = affinity.scale(sh.Point(0, 0).buffer(1, quad_segs=64), size[0], size[1])
        g = affinity.rotate(g, p.get('yaw', 0), use_radians=True)
        return affinity.translate(g, *center[:2])
    if 'axis_x' in p:
        x = np.asarray(p['axis_x']); y = np.asarray(p['axis_y'])
        z = np.cross(x, y); z /= np.linalg.norm(z)
    else:
        a = p.get('yaw', 0)
        x, y, z = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0]), np.array([0, 0, 1])
    corners = [center+i*size[0]*x+j*size[1]*y+k*size[2]*z
               for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
    return sh.MultiPoint(np.asarray(corners)[:, :2]).convex_hull


class Geometry:
    def __init__(self, bundle, route):
        self.route = route
        tri = np.array([p['vertices'][:3] for p in bundle['colliders']])
        polygons = sh.polygons(tri[:, :, :2])
        valid = sh.area(polygons) > 1e-12
        self.tri = tri[valid]; self.surfaces = polygons[valid]
        self.surface_tree = STRtree(self.surfaces)
        road = [g for p, g in zip(bundle['colliders'], polygons) if p['name'].startswith('road_')]
        self.support = sh.union_all(road).buffer(SEAM, join_style=2).buffer(-SEAM, join_style=2)
        # Break long edges into individual segments for indexed exact distances.
        lines = []
        for part in sh.get_parts(sh.boundary(self.support)):
            coords = np.asarray(part.coords)
            lines.extend(sh.linestrings(np.stack([coords[:-1], coords[1:]], 1)))
        self.edge_tree = STRtree(lines)
        self.edge_lines = np.asarray(lines, object)
        sh.prepare(self.support)
        self.props = [p for p in bundle['primitives'] if p.get('collision', True)]
        self.polygons = np.asarray([primitive_polygon(p) for p in self.props], object)
        self.prop_tree = STRtree(self.polygons)
        self.blocking = np.array([not p['name'].startswith('root_') for p in self.props])
        self.block_polygons = self.polygons[self.blocking]
        self.block_tree = STRtree(self.block_polygons)
        self.block_ids = np.flatnonzero(self.blocking)

    def signed_edge(self, xy):
        points = sh.points(xy)
        pair, distance = self.edge_tree.query_nearest(points, return_distance=True, all_matches=False)
        out = np.empty(len(points)); out[pair[0]] = distance
        return np.where(sh.contains(self.support, points), -out, out)

    def ground(self, xy):
        points = sh.points(xy)
        pair = self.surface_tree.query(points, predicate='intersects')
        ii, jj = pair
        a, b, c = self.tri[jj].transpose(1, 0, 2)
        normal = np.cross(b-a, c-a)
        z = a[:, 2]-(normal[:, :2]*(xy[ii]-a[:, :2])).sum(1)/normal[:, 2]
        ground = np.full(len(xy), -np.inf)
        np.maximum.at(ground, ii, z)
        return ground

    def obstacle_layers(self, xy, ground, pad):
        points = sh.points(xy)
        pair, dist = self.block_tree.query_nearest(points, return_distance=True, all_matches=False)
        distance = np.empty(len(xy)); distance[pair[0]] = dist
        ids = np.empty(len(xy), int); ids[pair[0]] = self.block_ids[pair[1]]
        # A closed projection is an obstacle even if a low rock top is <22 cm.
        # pad includes cell quantization and dense footprint sampling error.
        distance -= pad
        kinds = np.array([kind(p['name']) for p in self.props], np.float32)
        typ = kinds[ids]
        top = ground.copy()
        # Include all returns intersecting the cell, including traversable roots.
        boxes = sh.box(xy[:, 0]-pad, xy[:, 1]-pad, xy[:, 0]+pad, xy[:, 1]+pad)
        pair = self.prop_tree.query(boxes, predicate='intersects')
        ii, jj = pair
        for j in np.unique(jj):
            ix = ii[jj == j]; p = self.props[j]
            if p['type'] == 'capsule':
                a, b = np.asarray(p['fromto']).reshape(2, 3)
                v = b[:2]-a[:2]
                u = np.clip((xy[ix]-a[:2])@v/max(v@v, 1e-12), 0, 1)
                z = a[2]+u*(b[2]-a[2])+p['radius']
            else:
                z = np.full(len(ix), p['position'][2]+p['size'][2])
            np.maximum.at(top, ix, z)
        return distance, top, typ

    def footprint(self, xy, yaw):
        local = np.array([[-.515, -.29], [.515, -.29], [.515, .29], [-.515, .29]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return sh.Polygon(local@rot.T+xy)

    def clearance(self, footprint):
        outside = footprint.difference(self.support)
        if outside.area > 1e-8:
            # Negative penetration estimate, exact feasibility remains polygon coverage.
            coords = sh.get_coordinates(outside)
            edge = -float(np.max(np.maximum(self.signed_edge(coords), 0)))
        else:
            edge = float(footprint.distance(sh.boundary(self.support)))
        nearest = self.block_tree.nearest(footprint)
        ob = float(footprint.distance(self.block_polygons[nearest]))
        intersects = footprint.intersection(self.block_polygons[nearest]).area > 1e-10
        return edge, ob, bool(outside.area <= 1e-8 and not intersects), self.props[self.block_ids[nearest]]['name']


def kind(name):
    if name.startswith('rock_'): return 1
    if name.startswith('root_'): return 2
    if name.startswith('rail_'): return 3
    if name.startswith('stair_wall_'): return 4
    if name.startswith('tree_'): return 5
    return 6
