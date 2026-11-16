"""Final artifact/link/shape/integrity verification without rerunning experiments."""
import ast,json,re,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from .r8_p1_common import ROOT,GEO,dump

def main():
    sources=list((ROOT/'nav3').glob('r8_p1_*.py'))
    for p in sources:ast.parse(p.read_text(),filename=str(p))
    jsons=list((ROOT/'metrics').glob('r8_p1_*.json'))
    for p in jsons:json.loads(p.read_text())
    npzs=list(GEO.glob('r8_p1_*.npz'))
    for p in npzs:
        with zipfile.ZipFile(p) as z:assert z.testzip() is None,str(p)
    imgs=list((ROOT/'figures').glob('r8_p1_*.png'))
    for p in imgs:
        with Image.open(p) as im:im.verify()
    report=ROOT/'reports/R8_P1_SIMPLE_BEV.md'
    links=re.findall(r'\]\(([^)]+)\)',report.read_text())
    missing=[link for link in links if not (report.parent/link).resolve().exists()]
    assert not missing,missing
    refs=np.load(GEO/'r8_p1_references.npz');preds=np.load(GEO/'r8_p1_predictions.npz')
    for k in preds.files:assert preds[k].shape==(43,64,64,6) and preds[k].dtype==np.uint8,(k,preds[k].shape)
    assert np.unique(refs['poses'],axis=0).shape[0]==40
    metrics=json.loads((ROOT/'metrics/r8_p1_validation.json').read_text())
    assert len(metrics['rows'])==43
    before=json.loads((GEO/'r8_p1_provenance_before.json').read_text())
    # Detect any writes to old P0 cache files after the task started, as caches
    # are not in the protected-data hash list. mtime is only a supplemental check.
    start=(ROOT/'scripts/r8_p1_env.sh').stat().st_mtime_ns
    changed_cache=[str(p.relative_to(ROOT)) for p in (GEO/'r8_cache').rglob('*') if p.is_file() and p.stat().st_mtime_ns>start]
    assert not changed_cache,changed_cache
    result=dict(status='pass',python_sources=len(sources),json_files=len(jsons),npz_crc_checked=len(npzs),png_verified=len(imgs),comparison_figures=43,
        markdown_links_checked=len(links),missing_links=missing,prediction_shapes_valid=True,distinct_poses=40,old_p0_cache_new_mtimes=changed_cache,
        protected_existing_files=len(before),integrity=json.loads((ROOT/'metrics/r8_p1_integrity.json').read_text()))
    dump(ROOT/'metrics/r8_p1_artifact_check.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
