"""Aggregate only finished fixed-duration tiers. Wall-time denominators include cold reset."""
import collections
import json
import os
from pathlib import Path
import re
import statistics
import numpy as np


def analyze(base):
    results=[]
    for file in sorted((base/'results').glob('*/*/done.json')):
        folder=file.parent
        cfg=json.loads(file.read_text())
        card=folder.parent.name
        rate=float(re.search(r'Usage rate: ([.\d]+)',(base/'logs'/f'mv4col-{card}.usage').read_text())[1])
        samples=[json.loads(s) for s in (folder/'resources.jsonl').read_text().splitlines()]
        reasons=collections.Counter()
        qualified=steps=stands=attempts=0
        wall=cfg['end']-cfg['start']
        metrics=[]
        for slot in range(cfg['n']):
            durations=collections.Counter()
            topics=collections.Counter()
            ratios=[]
            sm=[]
            peak=0
            cpu=cpu_time=0
            prev=None
            for row in samples:
                info=row['slots'].get(str(slot))
                if info:
                    if info.get('sm_pct') is not None:
                        sm.append(info['sm_pct'])
                    peak=max(peak,info.get('gpu_mib') or 0)
                    if prev and prev[1]['pid']==info['pid']:
                        cpu+=info['cpu_s']-prev[1]['cpu_s']
                        cpu_time+=row['epoch']-prev[0]
                prev=(row['epoch'],info) if info else None
            for ep in sorted(folder.glob(f'*-s{slot}-e*/episode.json')):
                obj=json.loads(ep.read_text())
                reason=obj['reason']
                reasons[reason]+=1
                steps+=obj['steps']
                qualified+=obj['qualified']
                if 'topic_counts' in obj:
                    attempts+=1
                    stands+=int(obj['stand_accepted'])
                log=ep.parent/'console.log'
                if log.is_file():
                    ratios.extend(float(a)/float(b) for a,b in re.findall(r'\[MujocoTiming\] actual=([.\d]+)Hz target=([.\d]+)Hz',log.read_text(errors='replace')))
                for topic,(first,last) in obj.get('topic_period',{}).items():
                    durations[topic]+=last-first
                    topics[topic]+=obj['topic_counts'][topic]-1
            hz=lambda k: round(topics[k]/durations[k],3) if durations[k] else None
            metrics.append(dict(slot=slot,rtf=round(statistics.mean(ratios),4) if ratios else None,
                                front_hz=hz('front_lidar'),rear_hz=hz('rear_lidar'),
                                ue_cpu_pct=round(100*cpu/cpu_time,1) if cpu_time else None,peak_vram_mib=peak,sm_pct=round(statistics.mean(sm),1) if sm else None))
        assert qualified==cfg['qualified'] and steps==cfg['steps']
        cards=np.array([[int(i) for i in s['card'].split(',')] for s in samples])
        results.append(dict(card=card,n=cfg['n'],wall_s=wall,qualified=qualified,steps=steps,
                         good_per_h=3600*qualified/wall,rate=rate,good_per_cu=3600*qualified/(wall*rate),
                         stood=stands,attempts=attempts,reasons=dict(reasons),instances=metrics,
                         card_util_pct=cards[:,0].mean(),card_peak_mib=int(cards[:,1].max())))
    return results


if __name__=='__main__':
    base=Path(os.environ['MATRIX_COLLECTION_DIR']).resolve()
    results=analyze(base)
    (base/'results/summary.json').write_text(json.dumps(results,indent=2))
    for r in results:
        print(r)
