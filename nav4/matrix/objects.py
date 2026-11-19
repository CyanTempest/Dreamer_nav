"""Reconstruct explicitly supported hard objects and preserve v8 soft semantics."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from nav4.mapping.common import bilinear, cell_index
from nav4.matrix.cloud import groups, hull
from nav4.matrix.compose import write_obj
from nav4.matrix.ground_mesh import in_polygon
from nav4.matrix.register import summary, voxel_ids
from nav4.matrix.stairs import heading, path_at


def capsule(name, a, b, radius, evidence):
    return dict(name=name, material="rail", kind="rail",
                geom=dict(type="capsule", size=[radius], fromto=np.r_[a,b]),
                evidence=evidence)


def cylinder_section(points, prior, axis, radius_hint, guide_radius):
    """Recover a cylinder centre/radius in the plane perpendicular to its axis."""
    e1 = np.r_[-axis[1], axis[0], 0.]
    if np.linalg.norm(e1) < .1:
        e1 = np.array([1.,0.,0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    rel = points-prior
    pos = np.c_[rel@e1, rel@e2]
    keep = np.linalg.norm(pos,axis=1) <= guide_radius
    pos = pos[keep]
    if len(pos) < 10:
        return prior, radius_hint, dict(measured=False,count=len(pos))
    options = []
    def residual(w):
        return np.linalg.norm(pos-w[:2],axis=1)-w[2]
    for guess in (np.median(pos,axis=0), np.zeros(2)):
        optim = least_squares(residual, np.r_[np.clip(guess,-.065,.065),radius_hint],
                             bounds=([-.095,-.095,.012],[.095,.095,.048]),
                             loss="cauchy",f_scale=.007,max_nfev=35)
        r = residual(optim.x)
        hit = abs(r) < .010
        angle = np.arctan2((pos[hit]-optim.x[:2])[:,1],(pos[hit]-optim.x[:2])[:,0])
        quadrants = np.unique(np.floor(angle/1.57)).size
        score = hit.sum() * (1+.25*quadrants) / (1+np.linalg.norm(optim.x[:2])/.11)
        options.append((score,optim.x,hit,quadrants,r))
    _, fit, hit, sectors, error = max(options,key=lambda a:a[0])
    if hit.sum() < 10 or sectors < 3:
        return prior, radius_hint, dict(measured=False,count=int(hit.sum()))
    return prior+fit[0]*e1+fit[1]*e2, fit[2], dict(measured=True, **summary(error[hit]))


def rail_shapes(points, path, ground, origin, pitch, tracks):
    output = []
    tree = cKDTree(points)
    for spec in tracks:
        track = np.array(spec["track"])
        bins = np.linspace(track[0,0],track[-1,0],int(np.ceil(np.ptp(track[:,0])/.34))+1)
        bins = np.union1d(bins, np.array(spec["posts_m"])[
            (np.array(spec["posts_m"])>=bins[0]) & (np.array(spec["posts_m"])<=bins[-1])])
        offset = np.interp(bins,track[:,0],track[:,1])
        z = np.interp(bins,track[:,0],track[:,2])
        along = heading(path,bins)
        across = np.c_[-along[:,1],along[:,0]]
        xy = path_at(path,bins) + offset[:,None]*across
        top = np.c_[xy,z]
        fits = []
        for row, gap in (("top",0),("middle",spec["mid_offset_m"])):
            line = top - [0.,0.,gap]
            result = []
            for i, prior in enumerate(line):
                delta = line[min(i+1,len(line)-1)]-line[max(0,i-1)]
                axis = delta/np.linalg.norm(delta)
                in_range = tree.query_ball_point(prior,.27)
                p = points[in_range]
                p = p[abs((p-prior)@axis)<.16]
                result.append(cylinder_section(p,prior,axis,.025 if row=="top" else .019,.12))
            centers = np.array([r[0] for r in result])
            good = np.array([r[2]["measured"] for r in result])
            radii = np.array([r[1] for r in result])
            if good.sum() >= 2:
                for k in range(3):
                    correction = centers[:,k]-line[:,k]
                    centers[:,k] = line[:,k]+np.interp(bins,bins[good],correction[good])
                radius = float(np.median(radii[good]))
            else:
                radius = float(np.median(radii))
            for i in range(len(centers)-1):
                key = f"{spec['name']}_{row}_{i:02d}"
                output.append(capsule(key,centers[i],centers[i+1],radius,
                                      {"both_sections_measured":bool(good[i] and good[i+1]),
                                       "sections":[result[i][2],result[i+1][2]]}))
            fits.append((centers,radius))
        for i, post in enumerate(spec["posts_m"]):
            upper = np.array([np.interp(post,bins,axis) for axis in fits[0][0].T])
            floor = bilinear(ground,origin,upper[None,:2],res=pitch,fill=upper[2]-.92)[0]
            dz = float(np.clip(upper[2] - floor, .40, 1.25))
            prior = upper-[0.,0.,dz/2]
            p = points[tree.query_ball_point(prior,max(.31,dz*.44))]
            p = p[(abs(p[:,2]-prior[2])<dz*.44) & (np.linalg.norm(p[:,:2]-prior[:2],axis=1)<.075)]
            center,radius,notes = cylinder_section(p,prior,np.array([0.,0.,1.]),.023,.075)
            bottom, top_point = center.copy(),center.copy()
            bottom[2], top_point[2] = upper[2]-dz,upper[2]
            output.append(capsule(f"{spec['name']}_post_{i:02d}",bottom,top_point,radius,notes))
    return output


def rail_distance(points, rails):
    distance = np.full(len(points), np.inf)
    for item in rails:
        a,b = item["geom"]["fromto"].reshape(2,3)
        pad = item["geom"]["size"][0]+.08
        idx = np.flatnonzero(((points>=np.minimum(a,b)-pad) & (points<=np.maximum(a,b)+pad)).all(1))
        if len(idx):
            d = b-a
            p = points[idx]-a
            t = np.clip((p@d)/(d@d),0,1)
            distance[idx] = np.minimum(distance[idx],np.linalg.norm(p-t[:,None]*d,axis=1)-item["geom"]["size"][0])
    return distance


def solid_shapes(raw, path, v8_origin, layers, res, ground, ground_origin, pitch, stairs, rails, settings, out):
    ij = cell_index(raw[:,:2], v8_origin, res)
    hard = layers[...,6] == 1
    near_hard = ndi.maximum_filter(hard,size=3)
    soft = layers[...,6] == 2
    i,j = ij.T
    prior_ground = layers[i,j,0]
    dense_ground = bilinear(ground,ground_origin,raw[:,:2],res=pitch)
    # Use v8 semantics only for admitting points to hard volume reconstruction.
    # Geometry and height are measured on the dense source.
    above = raw[:,2]-np.minimum(prior_ground,dense_ground)
    selected = near_hard[i,j] & ~soft[i,j] & (above>.11) & (above<settings["max_height_m"])
    short = np.flatnonzero(selected)
    dt = cKDTree(path).query(raw[short,:2],workers=4)[0]
    selected[short] = (dt<2.1) & ((dt<1.35) | (above[short]<.67))
    p = raw[selected]
    excluded = np.zeros(len(p),dtype=bool)
    for shape in stairs:
        poly = shape["polygon"]
        box = ((p[:,:2]>=poly.min(0)-.015)&(p[:,:2]<=poly.max(0)+.015)).all(1)
        excluded[box] |= in_polygon(p[box,:2],poly,margin=.015)
    p = p[~excluded]
    p = p[voxel_ids(p,.025)]
    p = p[rail_distance(p,rails)>.06]
    tile = settings["tile_m"]
    key = np.floor(p/tile).astype(int)
    output = []
    for idx in groups(key):
        points = p[idx]
        if len(points) < 22 or np.linalg.norm(np.ptp(points,axis=0)) < .16:
            continue
        zero = points-np.median(points,axis=0)
        _, scale, axes = np.linalg.svd(zero,full_matrices=False)
        if scale[2] > .30*scale[1] or scale[1] < .09:
            continue
        distance = zero@axes[-1]
        core = abs(distance-np.median(distance)) < .031
        if core.mean() < .72:
            continue
        points = points[core]
        fitted = hull(points,settings["minimum_thickness_m"])
        if fitted is None:
            continue
        center, vertices, faces = fitted
        name = f"solid_{len(output):04d}"
        path = f"hard/{name}.obj"
        write_obj(out/path,vertices,faces)
        record = dict(name=name, geom=dict(type="mesh",pos=center), kind="wall_rock_or_root",
                      material="stone", file=path, faces=len(faces), returns=len(points))
        output.append(record)
    return output


def wall_shapes(points, path, specs, out):
    """Connect measured sections of a visible retaining face into solid prisms."""
    from nav4.matrix.stairs import PRISM_FACES

    walls = []
    for spec in specs:
        knobs = np.array(spec["track"])
        stations = np.linspace(knobs[0,0],knobs[-1,0],int(np.ceil(np.ptp(knobs[:,0])/.23))+1)
        reference = np.column_stack([np.interp(stations,knobs[:,0],knobs[:,i]) for i in (1,2)])
        along = heading(path,stations)
        normal = np.c_[-along[:,1],along[:,0]]
        centers = path_at(path,stations)
        tree = cKDTree(points[:,:2])
        face = []
        for i,(offset,foot) in enumerate(reference):
            guess = centers[i]+offset*normal[i]
            cloud = points[tree.query_ball_point(guess,.40)]
            rel = cloud[:,:2]-centers[i]
            local = rel@normal[i]
            use = (abs(rel@along[i])<.17)&(local>offset-.17)&(local<offset+.095)&(
                cloud[:,2]>foot+.11)&(cloud[:,2]<foot+.64)
            x,z = local[use],cloud[use,2]
            if len(x) < 16:
                face.append(None)
                continue
            bins = np.arange(offset-.16,offset+.106,.012)
            count,_ = np.histogram(x,bins)
            mode = np.argmax(np.convolve(count,[1,2,1],mode="same"))
            c = (bins[mode]+bins[mode+1])/2
            on = abs(x-c)<.026
            if on.sum() < 12 or np.ptp(z[on])<.14:
                face.append(None)
                continue
            top = max(foot+.18,float(np.percentile(z[on],90))+.025)
            face.append((centers[i]+np.median(x[on])*normal[i],foot-.13,top,int(on.sum())))
        for i,a in enumerate(face[:-1]):
            b = face[i+1]
            if a is None or b is None or np.linalg.norm(a[0]-b[0])>.40:
                continue
            polygon = np.array([a[0]-.17*normal[i], b[0]-.17*normal[i+1], b[0], a[0]])
            v = np.r_[np.c_[polygon,[a[1],b[1],b[1],a[1]]],
                      np.c_[polygon,[a[2],b[2],b[2],a[2]]]]
            center = v.mean(0)
            name = f"wall_{len(walls):03d}"
            filename = f"hard/{name}.obj"
            write_obj(out/filename,v-center,PRISM_FACES)
            walls.append(dict(name=name,kind="retaining_wall",material="stone",
                              geom=dict(type="mesh",pos=center),file=filename,faces=len(PRISM_FACES),
                              returns=a[3]+b[3],station_m=[stations[i],stations[i+1]]))
    return walls


def woody_shapes(points, path, ground, origin, pitch, specs, out):
    """Fit stacked trunk sections and project supported root ridges into closed meshes."""
    result = []
    for spec in specs:
        st, cross = spec["station_m"], spec["lateral_m"]
        along = heading(path,st)
        center = path_at(path,st)+cross*np.array([-along[1],along[0]])
        nearby = points[(abs(points[:,:2]-center)<1.3).all(1)]
        base = bilinear(ground, origin, center[None,:], res=pitch)[0]
        horizontal = np.linalg.norm(nearby[:,:2]-center,axis=1)
        trunk = nearby[horizontal<.49]
        sections = []
        for height in np.arange(.18,2.95,.26):
            z = base+height
            nearby_z = trunk[abs(trunk[:,2]-z)<.13]
            if len(nearby_z) < 25:
                continue
            xy=nearby_z[:,:2]
            xy=xy[np.linspace(0,len(xy)-1,min(4000,len(xy))).astype(int)]
            prior=spec["radius_m"]
            opt=least_squares(lambda w: np.linalg.norm(xy-center-w[:2],axis=1)-w[2],
                              [0.,0.,prior],loss="cauchy",f_scale=.012,
                              bounds=([-.28,-.28,.065],[.28,.28,.34]),max_nfev=45)
            inlier=abs(opt.fun)<.025
            valid=xy[inlier]-center-opt.x[:2]
            sectors=np.unique(np.floor(np.arctan2(valid[:,1],valid[:,0])/.9))
            if len(sectors)<4 or inlier.sum()<20:
                continue
            sections.append((np.r_[center+opt.x[:2],z],opt.x[2],int(inlier.sum()),summary(opt.fun[inlier])))
        if len(sections)<3:
            continue
        first = max(1,len(sections)//3)
        lower_x = np.median([x[0][:2] for x in sections[:first]],axis=0)
        structural_radius = float(np.median([x[1] for x in sections[:first]]))
        bottom = bilinear(ground,origin,lower_x[None,:],res=pitch)[0]-.08
        trunk_segments = [(np.r_[lower_x,bottom],structural_radius,0,{})]+sections
        for idx,(a,b) in enumerate(zip(trunk_segments[:-1],trunk_segments[1:])):
            if b[0][2]-a[0][2] > .57:
                continue
            m = tapered_solid(a[0],b[0],a[1],b[1],10)
            result.append(write_solid(m,f"trunk_{spec['name']}_{idx:02d}","trunk","trunk",
                                      dict(returns=a[2]+b[2],diameter_m=[2*a[1],2*b[1]],
                                           surface_residual=[a[3],b[3]]),out))
        # Branch roots only enter the collision model where there is a measured
        # coherent ridge above a locally smoothed terrain reference.
        smooth = ndi.gaussian_filter(ground,.19/pitch)
        relative = nearby[:,:2]-lower_x
        floor = bilinear(smooth,origin,nearby[:,:2],res=pitch)
        z_offset = nearby[:,2]-floor
        radial = np.linalg.norm(relative,axis=1)
        low = (radial>structural_radius*.65)&(radial<1.15)&(z_offset>.055)&(z_offset<.28)
        p = nearby[low]
        if not len(p):
            continue
        rel = p[:,:2]-lower_x
        angle = np.arctan2(rel[:,1],rel[:,0])
        bins=np.arange(-np.pi,np.pi+.001,np.pi/12)
        histogram,_=np.histogram(angle,bins)
        for angle_id in np.argsort(histogram)[-3:]:
            az=(bins[angle_id]+bins[angle_id+1])/2
            d=np.array([np.cos(az),np.sin(az)])
            u=rel@d
            v=rel@np.array([-d[1],d[0]])
            use=(u>structural_radius*.7)&(abs(v)<.16)
            sample=p[use]
            if len(sample)<50 or np.ptp(u[use])<.27:
                continue
            edges=np.arange(structural_radius*.72,np.percentile(u[use],86)+.03,.12)
            for i,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                land=sample[(u[use]>=low)&(u[use]<=high)]
                if len(land)<16 or np.ptp(land[:,:2],axis=0).max()<.08:
                    break
                xyz = np.vstack([land,land-[0.,0.,.095]])
                fitted=hull(xyz,.025)
                if fitted is not None:
                    name=f"root_{spec['name']}_{angle_id:02d}_{i:02d}"
                    result.append(write_solid(fitted,name,"root","trunk",dict(returns=len(land)),out))
    return result


def tapered_solid(a,b,ra,rb,sides):
    """Closed convex section of a slightly tapered trunk."""
    axis=b-a
    axis/=np.linalg.norm(axis)
    perpendicular=np.cross(axis,[1.,0.,0.])
    perpendicular/=np.linalg.norm(perpendicular)
    second=np.cross(axis,perpendicular)
    t=np.arange(sides)*2*np.pi/sides
    ring=np.cos(t)[:,None]*perpendicular+np.sin(t)[:,None]*second
    return hull(np.vstack([a+ra*ring,b+rb*ring]),.025)


def write_solid(shape,name,kind,material,evidence,out):
    center,vertices,faces=shape
    path=f"hard/{name}.obj"
    write_obj(out/path,vertices,faces)
    return dict(name=name,kind=kind,material=material,geom=dict(type="mesh",pos=center),file=path,
                faces=len(faces),**evidence)
