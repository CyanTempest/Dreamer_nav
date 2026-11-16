"""Quantify old versus shoulder corridor with the real raster and full footprint."""
import json
from pathlib import Path
import numpy as np
from .lwy_obstacles_v2 import Raster,rotate
from .lwy_import import PAD,RES


def main():
    records=json.loads(Path('maps_lwy_v2/manifest.json').read_text())['maps'];rows=[]
    for r in records:
        if r['split']!='test':continue
        raster=Raster(Path('maps_lwy_v2')/r['name']/'map.npz');a=raster.a
        old=np.load(Path('maps_lwy_v2_obstacles/lwy')/r['name']/'catalog.npz')['obstacles']
        new=np.load(Path('maps_lwy_r7_obstacles/lwy')/r['name']/'catalog.npz')['obstacles']
        for slot in range(len(old)):
            for i in range(3):
                ob=old[slot,i];idx=((a['route'][:,:2]-ob[:2])**2).sum(-1).argmin()
                yaw=a['yaw'][idx];width=float(a['width'][idx]);station=float(a['s'][idx]);entry=dict(slot=slot,obstacle=i,station_m=station,width_m=width,half_sizes_m=ob[2:4].tolist())
                offsets=np.arange(-.55,.5501,.005);pose=a['route'][idx,:2]+rotate(np.column_stack([np.zeros(len(offsets)),offsets]),yaw)
                for label,boxes in [('legacy',old[slot]),('shoulder',new[slot])]:
                    b=boxes[i];lat=rotate(b[:2]-a['route'][idx,:2],-yaw)[1]
                    intrude=width/2-abs(float(lat))+float(b[3])
                    margin=raster.margins(pose,np.full(len(pose),yaw),boxes);valid=offsets[margin>=0]
                    entry[label]=dict(lateral_m=float(lat),intrusion_m=intrude,geometric_passage_m=width-intrude,
                        admissible_center_min_m=float(valid.min()) if len(valid) else None,
                        admissible_center_max_m=float(valid.max()) if len(valid) else None,
                        admissible_center_span_m=float(valid.max()-valid.min()) if len(valid) else 0.,
                        center_margin_m=float(margin[np.argmin(abs(offsets))]))
                rows.append(entry)
    out=dict(resolution_m=RES,conservative_pad_m=PAD,support_threshold_m=.025,envelope_m=[1.03,.58],rows=rows,
        interpretation='Support layer stores edge + .025 + PAD and terminates above .025, so the threshold cancels; it is not 2.5 cm extra drivable shoulder. Static obstacles use PAD, random boxes use exact signed distance. Perpendicular slices do not alone certify a connected path; the catalog audit retains the full witness.')
    with Path('metrics/lwy_r7_obstacle_geometry.json').open('x') as f:json.dump(out,f,indent=2)


if __name__=='__main__':main()
