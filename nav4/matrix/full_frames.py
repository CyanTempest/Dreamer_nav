"""Coarsely align the separate 4 Hz trail video with timestamped scanner poses."""
from pathlib import Path
import time

import cv2
import numpy as np
from scipy import ndimage as ndi

from nav4.mapping.common import write_json
from nav4.mapping.grid import load_centreline
from nav4.matrix.full_ground import FULL, WORK, grid_sample
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import LIMITS
from nav4.matrix.full_visual import images, panel
from nav4.matrix.stairs import heading, path_at
from PIL import Image

FRAMES = Path.home()/"ATEC_NAV"/"frames4hz"
SEC = .50
ROUTE_DS = .25
# The first platform and exit railing are visible in the film. Station
# intervals allow for the position of the observer relative to the landmark.
ENTRY = (0., 0.25, 2.0)
EXIT = (715.5, 713.75, 715.40)


def flow():
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    paths = sorted(FRAMES.glob("*.jpg"))[::2]
    output = np.zeros((len(paths),3))
    previous = None
    for i, file in enumerate(paths):
        gray = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (384, 216), interpolation=cv2.INTER_AREA)
        gray = gray[18:163, 22:362]
        if previous is not None:
            p = cv2.goodFeaturesToTrack(previous, 220, .022, 6.)
            if p is not None and len(p) >= 30:
                q, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, p, None, winSize=(25,25),
                                                   maxLevel=4)
                back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, previous, q, None, winSize=(25,25),
                                                       maxLevel=4)
                reliable = (ok[:,0]>0) & (ok2[:,0]>0) & (np.linalg.norm((back-p)[:,0], axis=1)<1.5)
                good_p, good_q = p[reliable,0], q[reliable,0]
                if len(good_p) >= 18:
                    affine, fit = cv2.estimateAffinePartial2D(good_p, good_q, method=cv2.RANSAC,
                                                                ransacReprojThreshold=3.)
                    if affine is not None:
                        center = np.array([170., 68.])
                        turn, pitch = affine[:, :2]@center+affine[:,2]-center
                        output[i] = turn/182, pitch/182, fit.mean()*min(1, len(good_p)/80)
        previous = gray
    np.save(WORK/"video_flow.npy", output)
    return output


def route_times(path):
    line = load_centreline()
    # The frozen route is a refined version of the scanned centerline.
    # Limit nearest-neighbor correspondence by its nominal station, so
    # a nearby switchback cannot cause a jump to a later visit.
    estimated = np.linspace(0., line["s"][-1], len(path))
    centers = np.searchsorted(line["s"], estimated)
    candidates = np.clip(centers[:, None] + np.arange(-60, 61), 0, len(line["xy"])-1)
    scale = np.sum((line["xy"][candidates]-path[:, None])**2, axis=2)
    idx = candidates[np.arange(len(path)), np.argmin(scale,axis=1)]
    anchor = np.maximum.accumulate(idx)
    return line["t"][anchor], np.linalg.norm(line["xy"][anchor]-path,axis=1)


def align(path, returns):
    count = len(returns)
    ss = np.arange(0, 722.01, ROUTE_DS)
    movie_t = np.arange(count)*SEC
    moving = heading(path, ss)
    theta = np.unwrap(np.arctan2(moving[:,1], moving[:,0]))
    theta = ndi.gaussian_filter1d(theta, 1.2/ROUTE_DS)
    map_change = ndi.uniform_filter1d(np.gradient(theta, ROUTE_DS), size=23)*4.5
    estimated = np.clip(returns[:,0], -.35,.35)
    expected_yaw = ndi.uniform_filter1d(estimated, size=11) * 9.
    support = ndi.uniform_filter1d(returns[:,2], size=11)
    # Angular sign/scale can drift with video stabilization, so bound its influence
    # and regularize the warping toward the duration and trimmed route endpoints.
    linear = np.interp(movie_t, [0., movie_t[-1]], [.8, 714.65])
    dynamic = np.full(len(ss), np.inf)
    start = (ss >= ENTRY[1]) & (ss <= ENTRY[2])
    dynamic[start] = (ss[start]-.85)**2*.025
    previous = np.full((count,len(ss)), -1, np.int8)
    speeds = np.arange(0,7)
    travel = [np.r_[np.full(v,np.inf), np.zeros(len(ss)-v)] for v in speeds]
    bridge_approach_time = 540.0
    for i in range(1,count):
        candidates = []
        for j, v in enumerate(speeds):
            candidate = travel[j].copy()
            candidate[v:] += dynamic[:len(ss)-v] + .018*(v-2.0)**2
            candidates.append(candidate)
        candidates = np.asarray(candidates)
        best = np.argmin(candidates, axis=0)
        reward = .80 * support[i] * (np.minimum(abs(expected_yaw[i]-map_change), .60))**2
        smooth_prior = ((ss-linear[i])/37.)**2*.016
        incoming = candidates[best, np.arange(len(ss))]
        dynamic = incoming + reward + smooth_prior
        mask = abs(ss-linear[i]) > 70.
        if abs(movie_t[i]-15.25)<.27:
            mask |= (ss<13.75)|(ss>17.5)
        if abs(movie_t[i]-bridge_approach_time)<.27:
            mask |= (ss<566.)|(ss>584.)
        dynamic[mask] = np.inf
        previous[i] = best
    end = (ss >= EXIT[1]) & (ss <= EXIT[2])
    j = np.flatnonzero(end)[np.argmin(dynamic[end])]
    location = np.empty(count,float)
    for i in range(count-1,-1,-1):
        location[i] = ss[j]
        if i:
            j -= previous[i,j]
    return movie_t, location, linear


def metadata():
    start = time.monotonic()
    stations, path = route()
    flow_path = WORK/"video_flow.npy"
    vectors = np.load(flow_path) if flow_path.exists() else flow()
    ts, ss, affine = align(path, vectors)
    track_time, track_xy = route_times(path)
    np.savez_compressed(WORK/"frame_alignment.npz", video_pts_s=ts, video_estimated_station_m=ss,
                        constant_speed_station_m=affine, trajectory_station_m=stations,
                        official_timestamp_s=track_time, scanner_xy_gap_m=track_xy)
    payload = dict(video_frames=len(list(FRAMES.glob("*.jpg"))), video_sampling_hz=4,
                   trial_pts_window_s=[float(ts[0]),float(ts[-1])], start_anchor=ENTRY, end_anchor=EXIT,
                   soft_landmark_anchors=[[15.25,13.75,17.5],[540.0,566.,584.]],
                   max_monotonic_warp_offset_m=float(np.max(abs(ss-affine))),
                   median_track_map_xy_distance_m=float(np.median(track_xy)),
                   sync_verified=False, seconds=time.monotonic()-start)
    write_json(FULL/"frame_alignment.json", payload)
    print(payload, flush=True)


def full_figures():
    _, path = route()
    align_data = np.load(WORK/"frame_alignment.npz")
    t, s = (align_data[k] for k in ("video_pts_s", "video_estimated_station_m"))
    ts, timestamp = (align_data[k] for k in ("trajectory_station_m", "official_timestamp_s"))
    surfaces = np.load(FULL/"surface.npz")
    base, z = surfaces["origin"],surfaces["z"]
    target = FULL/"figures"
    target.mkdir(exist_ok=True)
    rows = []
    stations = [35, 82, 129, 177, 225, 274, 323, 373, 424, 473, 524, 558, 584, 612, 636, 676, 701, 714]
    for n, station in enumerate(stations,1):
        i = int(np.searchsorted(LIMITS[1:-1], station))
        # Use the adjacent window when the sightline can cross a section boundary.
        filename = (FULL/f"window_{min(i, len(LIMITS)-3):02d}.xml")
        if min(station-LIMITS[i],LIMITS[i+1]-station) >= 7:
            filename = FULL/"segments"/f"seg_{i:02d}"/"scene.xml"
        pt = path_at(path, station)
        ahead = path_at(path, min(station+3.9,721.95))
        ht = grid_sample(z,base,np.array([pt,ahead]))
        camera = dict(eye=np.r_[pt,ht[0]+1.55],target=np.r_[ahead,ht[1]+.54],vertical_fov_deg=51.)
        official_time = float(np.interp(station, ts, timestamp))
        idx = int(np.argmin(abs(s-station)))
        pts = float(t[idx])
        frame_num = int(round(pts*4))+1
        real = Image.open(FRAMES/f"{frame_num:05d}.jpg")
        rendered = next(images(filename,[camera]))
        artifact = f"field_{n:02d}_{station:03d}m.jpg"
        alternative = np.interp(station, align_data["constant_speed_station_m"], t)
        panel([(real,f"Trail video  #{frame_num:05d}  +{pts:.2f}s  approx."),
               (rendered,f"OSMesa  {station:.1f} m  scanner UTC {official_time:.2f}")],target/artifact)
        rows.append(dict(station_m=station, official_timestamp_s=official_time, video_pts_s=pts,
                         frame=frame_num, video_synced=False, constant_rate_video_pts_s=alternative,
                         horizontal_landmark_error_m=None, artifact=artifact, camera=camera))
        print("rendered field", station, frame_num, flush=True)
        write_json(target/"field_positions.json", rows)


if __name__ == "__main__":
    import sys
    {"align":metadata, "render":full_figures}[sys.argv[1]]()
