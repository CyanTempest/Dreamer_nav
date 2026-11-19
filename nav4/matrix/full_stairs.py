"""Survey abrupt transverse edges in the raw-supported route surface, then fit LAS treads."""
import csv
import json

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from nav4.mapping.common import cell_index, load_config, write_json
from nav4.mapping.grid import load_centreline
from nav4.mapping.report_trail_map import detect_stairs
from nav4.matrix.full_ground import FULL, grid_sample, source_patch
from nav4.matrix.stairs import fit_flight, heading, path_at
from nav4.matrix.survey import OUT


def seed_profiles(path, z, origin):
    stations = np.arange(len(path)) * .05
    dy = heading(path, stations)
    side = np.c_[-dy[:, 1], dy[:, 0]]
    offsets = np.array([-.36, -.24, -.12, 0., .12, .24, .36])
    xy = path[:, None] + side[:, None]*offsets[None, :, None]
    profiles = grid_sample(z, origin, xy.reshape(-1, 2)).reshape(xy.shape[:2])
    at = np.arange(6, len(path) - 6)
    med = lambda offsets: np.median(profiles[at[:, None] + np.asarray(offsets)], axis=1)
    before, after = med([-3, -2]), med([2, 3])
    d = after - before
    left, right = before - med([-6, -5]), med([5, 6]) - after
    run = np.maximum(0, np.maximum(left*np.sign(d), right*np.sign(d)))
    excess = np.sign(d) * (abs(d) - run)
    score = []
    for sign in (1, -1):
        pair = ndi.maximum_filter1d(sign*excess, size=3, axis=0)
        vote = np.sort(pair, axis=1)[:, -4]
        vote = np.minimum(vote, np.maximum.reduce(sign*excess[:, 2:5], axis=1))
        score.append(vote)
    combined = np.maximum(*score)
    peaks = find_peaks(combined, height=.058, prominence=.023, distance=5)[0]
    return stations[at[peaks]], profiles, combined[peaks]


def v8_steps(path, v8):
    line = load_centreline()
    cfg, _ = load_config()
    st = np.arange(0., line["s"][-1], .05)
    xy = np.column_stack([np.interp(st, line["s"], a) for a in line["xy"].T])
    ij = cell_index(xy, v8.origin, v8.resolution)
    heights = v8.layers[ij[:, 0], ij[:, 1], 0]
    _, rows = detect_stairs(st, heights, cfg)
    previous = cKDTree(path).query(xy, workers=4)[1] * .05
    for i, row in enumerate(rows):
        row["scanner_station_m"] = row.pop("s")
        index = round(row["scanner_station_m"]/.05)
        row["station_m"] = float(previous[index])
        row["riser_m"] = row.pop("h")
        if i+1 < len(rows):
            end = rows[i+1]["s"]
            delta = end - row["scanner_station_m"]
            if delta <= 1.4 and np.sign(rows[i+1]["h"]) == np.sign(row["riser_m"]):
                row["tread_m"] = delta
    return [r for r in rows if r["station_m"] < 722.]


def spans(seeds, profiles):
    groups = []
    for seed, strength in seeds:
        k = round(seed/.05)
        sign = np.sign(np.median(profiles[k+2, 2:5]-profiles[k-2, 2:5]))
        row = (seed, strength, sign)
        if not groups or seed-groups[-1][-1][0] > 1.23 or sign != groups[-1][-1][2]:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def fit_group(path, profiles, edges, label):
    a = edges[0] - .24
    b = min((len(path)-1)*.05, edges[-1]+.28)
    clip = np.asarray([a, *edges, b])
    xy = path_at(path, np.arange(a, b+.05, .05))
    raw, _ = source_patch(xy.min(0)-1.10, xy.max(0)+1.10)
    raw = raw[raw[:, 3] == 0, :3]
    if not len(raw):
        raise ValueError("Empty LAS crop")
    guesses = []
    for left, right in zip(clip[:-1], clip[1:]):
        mid = round((left+right)/.10)
        guesses.append(float(np.median(profiles[mid, 2:5])))
    # Search over a conservative width for the edge; supported width is
    # independently recorded by the riser fit's transverse extent.
    widths = []
    for edge in clip:
        k = round(edge/.05)
        profile = profiles[k]
        keep = abs(profile-np.median(profile[2:5])) < .15
        left = .46 if not keep[1] else .58 if not keep[0] else .68
        right = .46 if not keep[-2] else .58 if not keep[-1] else .68
        widths.append([-left, right])
    flight = dict(edges_m=edges, before_m=a, end_m=b, tread_z_guess_m=guesses,
                  width_station_m=clip, lateral_bounds_m=widths)
    shapes, metrics = fit_flight(raw, path, flight, {"edge_search_m": .12})
    for shape, metric in zip(shapes, metrics):
        metric["flight"] = label
        shape["station_m"] = metric["station_m"]
    return shapes, metrics


def survey_stairs(path, v8, surface, origin):
    edge_bank, profiles, strength = seed_profiles(path, surface, origin)
    old = v8_steps(path, v8)
    seeds = list(zip(edge_bank[edge_bank > 25], strength[edge_bank > 25]))
    for step in old:
        s = step["station_m"]
        if 25 < s < 721.65 and all(abs(s - a) > .23 for a, _ in seeds):
            seeds.append((s, 0.))
    seeds = sorted((s, v) for s, v in seeds if 25 < s < 721.65)
    np.savez_compressed(FULL/"stairs_profile.npz", station_m=np.arange(len(path))*.05,
                        cross_profiles_m=profiles, raw_candidate_stations_m=edge_bank,
                        raw_candidate_strength_m=strength)
    accepted, rejects, all_shapes = [], [], []
    pilot = json.loads((OUT/"stairs.json").read_text())
    for flight in pilot["flights"]:
        for row in flight["steps"]:
            accepted.append(dict(row, flight=flight["flight"], origin="pilot_dense_LAS"))
    for sequence, group in enumerate(spans(seeds, profiles), 1):
        edges = [s for s, _, _ in group]
        label = f"dense_{sequence:03d}"
        try:
            shapes, metrics = fit_group(path, profiles, edges, label)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rejects.extend(dict(seed_station_m=s, raw_jump_score_m=v, reason=str(exc)) for s, v, _ in group)
            continue
        for shape, row, (seed, score, _) in zip(shapes, metrics, group):
            riser = abs(row["riser_m"])
            reason = []
            if not row["riser_fit"]["fit"] or row["riser_fit"].get("rms", 1.) > .033:
                reason.append("weak vertical LAS fit")
            if not (.075 <= riser <= .35 and row["tread_depth_m"] >= .12):
                reason.append("below riser/depth limits or implausible height")
            if row["tread_fit"]["rms"] > .032:
                reason.append("nonplanar tread")
            if reason:
                rejects.append(dict(seed_station_m=seed, raw_jump_score_m=score,
                                    provisional_station_m=row["station_m"], riser_m=row["riser_m"],
                                    reason=", ".join(reason)))
                continue
            all_shapes.append(shape)
            accepted.append(dict(row, origin="dense_LAS_survey", raw_jump_score_m=score))

    # The v8 compare is one-to-one by station and direction. Preserve unused old
    # candidates for investigating merged steps and false positives.
    pairs = sorted((abs(r["station_m"]-e["station_m"]), i, j)
                   for i, r in enumerate(accepted) for j, e in enumerate(old)
                   if np.sign(r["riser_m"]) == np.sign(e["riser_m"]) and
                   abs(r["station_m"]-e["station_m"]) <= .32)
    use_new, use_old = set(), set()
    for row in accepted:
        for k in tuple(row):
            if k.startswith("v8_"):
                row.pop(k)
    for d, i, j in pairs:
        if i in use_new or j in use_old:
            continue
        use_new.add(i)
        use_old.add(j)
        new, prev = accepted[i], old[j]
        new.update(v8_station_m=prev["station_m"], v8_scanner_station_m=prev["scanner_station_m"],
                   v8_riser_m=prev["riser_m"], v8_tread_depth_m=prev.get("tread_m"),
                   riser_minus_v8_m=new["riser_m"]-prev["riser_m"])
    accepted.sort(key=lambda a: a["station_m"])
    columns = ("flight", "station_m", "riser_m", "tread_depth_m", "width_m", "origin",
               "end_is_landing_boundary", "v8_station_m", "v8_scanner_station_m", "v8_riser_m",
               "v8_tread_depth_m", "riser_minus_v8_m")
    with (FULL/"stairs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    write_json(FULL/"stairs.json", dict(accepted=accepted, review_candidates=rejects,
                                        v8_unpaired=[a for i,a in enumerate(old) if i not in use_old],
                                        v8_all=old))
    print("stairs", len(accepted), "needs review", len(rejects), flush=True)
    return all_shapes
