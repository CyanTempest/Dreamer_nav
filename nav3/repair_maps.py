"""Migrate existing v2 rasters without rerunning the expensive terrain generator."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from .maps import ROOT,Scene,rasterize,repair_bypass


def repair(record):
    folder=ROOT/'maps'/record['name']
    meta=json.loads((folder/'metadata.json').read_text())
    if meta.get('raster_version')==3:return meta
    with np.load(folder/'map.npz') as f:arrays={k:f[k] for k in f.files}
    sc=Scene.__new__(Scene)
    sc.mini=meta['mini'];sc.features=meta['features'];sc.res=float(arrays['res']);sc.origin=arrays['origin']
    layers=arrays['layers'].astype(np.float32)
    sc.ground=layers[...,0];sc.edge=layers[...,2];sc.main_edge=layers[...,3];sc.projected_s=layers[...,4]
    sc.route=arrays['route'];sc.xy=sc.route[:,:2];sc.s=arrays['s'];sc.width=arrays['width'];sc.yaw=arrays['yaw']
    sc.props=json.loads((folder/'props.json').read_text())
    repairs=repair_bypass(sc)
    if repairs:
        arrays['layers']=rasterize(sc).astype(np.float16)
        np.savez_compressed(folder/'map.npz',**arrays)
        (folder/'props.json').write_text(json.dumps(sc.props))
    meta.update(raster_version=3,bypass_repairs=repairs,sha256=hashlib.sha256((folder/'map.npz').read_bytes()).hexdigest())
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(name=record['name'],relocated_rocks=len(repairs))),flush=True)
    return meta


def main():
    manifest=json.loads((ROOT/'maps/manifest.json').read_text())
    with ProcessPoolExecutor(8) as pool:manifest['maps']=list(pool.map(repair,manifest['maps']))
    for path in ['maps/manifest.json','metrics/map_manifest.json']:(ROOT/path).write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
