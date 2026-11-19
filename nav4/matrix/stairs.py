"""Fit individual planar treads with vertical risers from dense stair returns."""
import numpy as np
from scipy.optimize import least_squares

from nav4.matrix.ground_mesh import in_polygon
from nav4.matrix.register import summary

# Floor outline and winding of a convex tread prism, viewed from above.
PRISM_FACES = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                        [2,3,7],[2,7,6],[3,0,4],[3,4,7]])


def path_at(path, station):
    t = np.arange(len(path))*.05
    return np.array([np.interp(station, t, axis) for axis in path.T]).T


def heading(path, station):
    direction = path_at(path, station+.20) - path_at(path, station-.20)
    return direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def perpendicular(u):
    return np.array([-u[1], u[0]])


def tread_plane(points, center, expected):
    window = points[abs(points[:,2]-expected) < .095]
    if len(window) < 12:
        raise ValueError("Insufficient nearby tread returns")
    zs = np.arange(expected-.10, expected+.115, .01)
    freq, _ = np.histogram(window[:,2], zs)
    peak = zs[freq.argmax()]+.005
    use = window[abs(window[:,2]-peak) < .035]
    model = np.c_[use[:,:2]-center, np.ones(len(use))]
    fit = least_squares(lambda x: model@x-use[:,2], [0., 0., peak],
                        loss="soft_l1", f_scale=.009,
                        bounds=([-.18,-.18,peak-.04],[.18,.18,peak+.04]))
    return fit.x, use, summary(fit.fun)


def prism(polygon, heights, base):
    return np.vstack([np.c_[polygon, np.full(len(polygon), base)], np.c_[polygon, heights]]), PRISM_FACES.copy()


def riser_edge(points, center, direction, low, high, width, tuning):
    """Fit a free XY line to between-tread returns, supporting both ends."""
    cross = perpendicular(direction)
    offset = points[:,:2]-center
    uv = np.column_stack((offset@direction, offset@cross))
    lo, hi = sorted((low, high))
    use = (abs(uv[:,0]) < tuning["edge_search_m"]) & (uv[:,1] > width[0]+.06) & (
        uv[:,1] < width[1]-.06) & (points[:,2]>lo+.028) & (points[:,2]<hi-.028)
    xy = uv[use]
    if len(xy) < 14:
        return center, direction, dict(count=len(xy), fit=False)
    # Sweep slope and offset in a small, physically plausible window.
    # Score lateral span as well as return count to downweight foliage clusters.
    bins = np.linspace(-tuning["edge_search_m"], tuning["edge_search_m"], 43)
    group = np.floor(xy[:,1]/.10).astype(int)
    best = None
    for slope in np.linspace(-.55,.55,45):
        along = xy[:,0]-slope*xy[:,1]
        counts, _ = np.histogram(along,bins)
        idx = int(np.argmax(np.convolve(counts,[1,2,1],mode="same")))
        intercept = (bins[idx]+bins[idx+1])/2
        hits = abs(along-intercept) < .025
        score = np.unique(group[hits]).size * np.sqrt(hits.sum()) / (1+2*abs(intercept))
        if best is None or score > best[0]:
            best = score, slope, intercept, hits
    _, slope, intercept, hits = best
    if hits.sum() < 12 or np.ptp(xy[hits,1]) < .23:
        return center, direction, dict(count=int(hits.sum()), fit=False)
    fit = least_squares(lambda x: xy[hits,0]-x[0]*xy[hits,1]-x[1], [slope,intercept],
                        loss="soft_l1", f_scale=.007)
    normal = direction-fit.x[0]*cross
    normal /= np.linalg.norm(normal)
    return center+fit.x[1]*direction, normal, dict(fit=True, **summary(fit.fun))


def line_intersection(pt, normal, a, b):
    vec = b-a
    denominator = vec@normal
    if abs(denominator) < .07*np.linalg.norm(vec):
        raise ValueError("Stair edge and side are nearly parallel")
    return a+vec*((pt-a)@normal/denominator)


def fit_flight(p, path, flight, tuning):
    stations = np.array(flight["edges_m"])
    stencil = np.r_[flight["before_m"], stations, flight["end_m"]]
    mids = (stencil[1:] + stencil[:-1]) / 2
    widths = np.column_stack([np.interp(stencil, flight["width_station_m"], edge)
                              for edge in np.asarray(flight["lateral_bounds_m"]).T])
    center = path_at(path, stencil)
    dirs = heading(path, stencil)
    sides = center[:,None]+widths[:,:,None]*np.array([perpendicular(u) for u in dirs])[:,None]
    plane_params, results, samples = [], [], []
    for mid, guess, (lo, hi) in zip(mids, flight["tread_z_guess_m"], zip(stencil[:-1], stencil[1:])):
        anchor = path_at(path, mid)
        axis = heading(path, mid)
        pos = p[:,:2]-anchor
        along, lateral = pos@axis, pos@perpendicular(axis)
        keep = (abs(along) < min(.36,(hi-lo)*.32)) & (abs(lateral)<.25)
        coef, points, stats = tread_plane(p[keep], anchor, guess)
        plane_params.append((anchor, coef))
        results.append(stats)
        samples.append(points)

    def height(i, xy):
        a, c = plane_params[i]
        return (xy-a)@c[:2]+c[2]

    edges = [sides[0]]
    edge_stats, actual_station = [], []
    for k, station in enumerate(stations, 1):
        anchor, axis = center[k], dirs[k]
        front, normal, stats = riser_edge(p, anchor, axis, height(k-1,anchor),
                                          height(k,anchor), widths[k], tuning)
        boundary = []
        for j in range(2):
            # Evaluate the local side segment on either side of the nominal
            # edge so that the exterior follows the measured stair bend.
            a, b = sides[k-1,j], sides[k+1,j]
            q = line_intersection(front,normal,a,b)
            if np.linalg.norm(q-sides[k,j]) > .40:
                q = line_intersection(front,normal,sides[k,j]-.5*axis,sides[k,j]+.5*axis)
            boundary.append(q)
        edges.append(np.array(boundary))
        edge_stats.append(stats)
        actual_station.append(station+(front-anchor)@axis)
    edges.append(sides[-1])
    base = min(x[1][2] for x in plane_params)-.38
    shapes, metrics = [], []
    for k in range(1,len(stencil)-1):
        polygon = np.array([edges[k][0],edges[k+1][0],edges[k+1][1],edges[k][1]])
        top = height(k, polygon)
        v, f = prism(polygon, top, base)
        front_center = path_at(path, actual_station[k-1])
        current_edge, next_edge = edges[k], edges[k+1]
        front_normal = np.array([current_edge[1,1]-current_edge[0,1], current_edge[0,0]-current_edge[1,0]])
        front_normal /= np.linalg.norm(front_normal)
        metrics.append(dict(station_m=actual_station[k-1],
                            riser_m=float(height(k,front_center)-height(k-1,front_center)),
                            tread_depth_m=float((next_edge.mean(0)-current_edge.mean(0))@front_normal),
                            width_m=float(np.linalg.norm(current_edge[0]-current_edge[1])),
                            plane_normal=np.r_[-plane_params[k][1][:2],1]/np.linalg.norm(np.r_[plane_params[k][1][:2],1]),
                            tread_fit=results[k], riser_fit=edge_stats[k-1],
                            end_is_landing_boundary=k == len(stencil)-2,
                            v8_nearest_station_delta_m=None))
        shapes.append(dict(vertices=v, faces=f, polygon=polygon, edge=current_edge,
                           height_m=float(height(k, front_center)),
                           before_plane_anchor=plane_params[k-1][0],
                           before_plane_coeff=plane_params[k-1][1]))
    return shapes, metrics
