"""Clone exactly the official 3 h state. Never write source files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    assert a.name.startswith('dreamer_lwy_r7')
    source=Path('checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl')
    payload=pickle.load(source.open('rb'));src=Path(payload['official_checkpoint'])
    out=Path('checkpoints')/a.name;assert not out.exists()
    cp=out/'official_ckpt/ckpt'/src.name;shutil.copytree(src,cp)
    (cp.parent/'latest').write_text(src.name)
    shutil.copy2(source,out/'latest.pkl')
    import elements
    replay=pickle.load((cp/'replay.pkl').open('rb'))
    runtime=pickle.load((cp/'runtime.pkl').open('rb'));runtime.update(wall_seconds=0.,updates=0,sample_old=0,sample_total=0,last_report_old=0,last_report_total=0,
        old_ids=[elements.UUID(r[0]).value for r in replay['chunks'] if int(r[1].removesuffix('.npz').split('-')[-1])>0])
    pickle.dump(runtime,(cp/'runtime.pkl').open('wb'),pickle.HIGHEST_PROTOCOL)
    # driver paths are read-only source references until the first new save.
    replay=pickle.load((cp/'replay.pkl').open('rb'));dst=Path('replay')/a.name;dst.mkdir()
    records=[r for r in replay['chunks'] if int(r[1].removesuffix('.npz').split('-')[-1])>0]
    assert all((Path('replay/dreamer_lwy_runtime2')/r[1]).is_file() for r in records)
    def copy(r):shutil.copy2(Path('replay/dreamer_lwy_runtime2')/r[1],dst/r[1])
    with ThreadPoolExecutor(4) as pool:list(pool.map(copy,records))
    result=dict(source=str(source),official_state=str(src),name=a.name,nonempty_chunks=len(records),
        empty_chunks=len(replay['chunks'])-len(records),replay_items=len(replay['items']),
        future_samples_used=False,driver_source_read_only=True)
    with Path(f'metrics/lwy_r7_clone_{a.name}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
