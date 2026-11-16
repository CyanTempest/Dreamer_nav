"""Reserve geographically separate diagnostic routes before expanded evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser();p.add_argument('survey',type=Path);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dev=json.loads((a.development/'manifest.json').read_text())['maps']
    xy=np.concatenate([np.load(a.development/r['name']/'map.npz')['route'][:,:2] for r in dev])
    tree=cKDTree(xy);manifest=json.loads((a.survey/'manifest.json').read_text());accepted=[];excluded=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for record in manifest['maps']:
        with np.load(a.survey/record['name']/'map.npz') as z:route=z['route'][:,:2]
        separation=float(tree.query(route)[0].min())
        updated={**record,'min_development_route_separation_m':separation}
        if separation<20:excluded.append(updated);continue
        target=a.output/record['name'];target.mkdir(exist_ok=True)
        for file in ['map.npz','metadata.json']:shutil.copy2(a.survey/record['name']/file,target/file)
        accepted.append(updated)
    result={**manifest,'maps':accepted,'split_rule':'At least 20 m XY separation from two development routes; no policy outcome used for selection',
            'excluded_for_development_proximity':[r['name'] for r in excluded]}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'holdout_maps':len(accepted),'excluded':len(excluded),'names':[r['name'] for r in accepted]}))


if __name__=='__main__':main()
