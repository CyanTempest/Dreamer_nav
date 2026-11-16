"""Verify raw geometry clearance along the relaxed-raster counterfactual."""
import json
from pathlib import Path
import numpy as np
from .diag_analysis import raw_gap
from .diag_geometry import write_json


def main():
    r=dict(np.load('metrics/diag_log_center_occupancy_full_v2.npz'));props=json.loads(Path('maps/mini_trail/props.json').read_text())
    rows=[]
    for ep in range(30):
        ids=np.flatnonzero((r['episode']==ep)&(r['s']>162)&(r['s']<166))
        gaps=[raw_gap(p,y,props) for i in ids for p,y in zip(r['sub_pos'][i],r['sub_yaw'][i])]
        rows.append(dict(episode=ep,min_gap_m=min(gaps),checked_substeps=len(gaps)))
    write_json('metrics/diag_physical_clearance.json',dict(rows=rows,physical_xy_projection_overlap=sum(x['min_gap_m']<=0 for x in rows),
        all_min_gap_m=min(x['min_gap_m'] for x in rows),definition='Rectangle 1.03x0.58 versus full raw log/bark XY projection at 20 Hz; cylinder axis sampled <=1mm'))

if __name__=='__main__':main()
