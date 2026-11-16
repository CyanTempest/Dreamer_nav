"""Canonicalize two pre-fix floating endpoints, proving raster arrays identical.

Only unpublished round-six artifacts are updated; frozen prior-round trees and
the original test package are never written. Stored specs match final generator.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .lwy_dataset import variant, vary_props, build_bundle
from .lwy_import import export


def rebuild(name):
    directory = Path('maps_lwy')/name
    old_spec = json.loads((directory/'spec.json').read_text())
    base = json.loads(Path('../../ATEC_Env/config/track_spec.json').read_text())
    expected = variant(base, old_spec['seed'])
    probe = json.loads(json.dumps(old_spec)); probe['track']['zones'][-1]['end_m'] = 1000.
    assert probe == expected and abs(old_spec['track']['zones'][-1]['end_m']-1000) < 1e-10
    metadata = json.loads((directory/'metadata.json').read_text())
    scratch = Path('tmp/lwy_canonical')/name
    route, bundle, _ = build_bundle(expected, scratch); vary_props(expected, route, bundle)
    (scratch/'scene/scene_bundle.json').write_text(json.dumps(bundle, separators=(',', ':')))
    fresh = export(expected, bundle, scratch, metadata['source_start_m'], metadata['source_end_m'])
    with np.load(directory/'map.npz') as a, np.load(scratch/'map.npz') as b:
        assert set(a.files) == set(b.files)
        changed = [key for key in a.files if not np.array_equal(a[key], b[key])]
        assert not changed, (name, changed)
    (directory/'spec.json').write_text(json.dumps(expected, indent=2))
    shutil.copyfile(scratch/'scene/scene_bundle.json', directory/'scene/scene_bundle.json')
    metadata['spec_sha256'] = hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest()
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return dict(name=name, old_endpoint=old_spec['track']['zones'][-1]['end_m'], new_endpoint=1000.,
                raster_arrays_identical=True, raster_file_unchanged=True, sha256=metadata['sha256'])


def main():
    assert Path('metrics/lwy_split_manifest.json').exists(), 'Wait for dataset publication first'
    with ProcessPoolExecutor(2) as pool: rows = list(pool.map(rebuild, ['train_610004', 'train_610011']))
    manifest = json.loads(Path('maps_lwy/manifest.json').read_text())
    manifest['maps'] = [json.loads((Path('maps_lwy')/r['name']/'metadata.json').read_text()) for r in manifest['maps']]
    for path in ['maps_lwy/manifest.json', 'metrics/lwy_split_manifest.json']:
        Path(path).write_text(json.dumps(manifest, indent=2))
    with Path('metrics/lwy_canonical_specs.json').open('x') as f: json.dump(dict(passed=True, rows=rows), f, indent=2)
    print(json.dumps(dict(passed=True, rows=rows)), flush=True)


if __name__ == '__main__': main()
