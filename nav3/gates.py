"""No hand-entered gate values. G2 uses common fixed-map return probes."""
import json
from pathlib import Path
import numpy as np


def read(name):
    path=Path('metrics')/f'train_{name}.jsonl'
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    result={'gate':'G2','passed':False,'definition':'100000 online transitions; final fixed TRAIN-map return > initial for both methods; Dreamer total trav IoU > initial'}
    for name in ['dreamer_s0','ppo_s0']:
        rows=read(name);probes=[r for r in rows if r['event']=='probe']
        end=rows[-1] if rows else {}
        if len(probes)<2:
            result[name]=dict(passed=False,reason='missing initial/final probes');continue
        first,last=probes[0],probes[-1]
        checks=dict(at_least_100k=end.get('steps',0)>=100000,
                    return_increased=last['probe_return']>first['probe_return'])
        detail=dict(initial_return=first['probe_return'],final_return=last['probe_return'],
            online_steps=end.get('steps'),updates=end.get('updates'),wall_seconds=end.get('wall_seconds'))
        if name.startswith('dreamer'):
            checks['iou_increased']=last['trav_iou'][0]>first['trav_iou'][0]
            detail.update(initial_iou=first['trav_iou'][0],final_iou=last['trav_iou'][0])
        result[name]=dict(passed=all(checks.values()),checks=checks,**detail)
    result['passed']=all(result[n]['passed'] for n in ['dreamer_s0','ppo_s0'])
    Path('metrics/g2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
