"""Summarize audit_20261116/results/B_*.json: per-run table, pooled Wilson CIs, contrasts."""
import json
import math
from pathlib import Path
from scipy.stats import binomtest, fisher_exact

R = Path(__file__).resolve().parent/'results'
SEEDS = [914901, 20261116]
CELLS = [('planner', None, 2), ('planner', None, 4), ('dreamer', 'sample', 2), ('dreamer', 'sample', 4),
         ('dreamer', 'mode', 2), ('dreamer', 'mode', 4)]


def name(m, act, g, s):
    return f'B_{m}_g{g}_s{s}_r8' if m == 'planner' else f'B_{m}_{act}_g{g}_s{s}_r8'


def wilson(k, n, z=1.959964):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z/d*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return c-h, c+h


def ok(r):
    return r['valid_start'] and r.get('gates') == 8 and r.get('failure') == 0


data = {}
print('| method | action | gain | seed | valid | succ | rate | failure codes | mean s | loop s |')
print('|---|---|---|---|---|---|---|---|---|---|')
for m, act, g in CELLS:
    for s in SEEDS:
        d = json.loads((R/(name(m, act, g, s)+'.json')).read_text())
        data[(m, act, g, s)] = d
        S = d['summary']
        print(f"| {m} | {act or '-'} | {g} | {s} | {S['valid_starts']} | {S['successes']} | {S['success_rate']:.3f} | "
              f"{S['failure_code_counts']} | {S['mean_s_at_termination']:.2f} | {d['seconds']:.1f} |")

print('\nPooled over seeds (Wilson 95%)')
print('| method | action | gain | succ/valid | rate | 95% CI |')
print('|---|---|---|---|---|---|')
pooled = {}
for m, act, g in CELLS:
    k = sum(data[(m, act, g, s)]['summary']['successes'] for s in SEEDS)
    n = sum(data[(m, act, g, s)]['summary']['valid_starts'] for s in SEEDS)
    pooled[(m, act, g)] = (k, n)
    lo, hi = wilson(k, n)
    print(f"| {m} | {act or '-'} | {g} | {k}/{n} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] |")


def paired(a, b):
    # Same seed + slot => same start pose, env PRNG stream and map; outcomes differ only via policy/gain.
    n10 = n01 = n11 = n00 = 0
    for s in SEEDS:
        ra, rb = data[a+(s,)]['rows'], data[b+(s,)]['rows']
        for i in ra:
            if not (ra[i]['valid_start'] and rb[i]['valid_start']): continue
            x, y = ok(ra[i]), ok(rb[i])
            n11 += x and y; n00 += (not x) and (not y); n10 += x and not y; n01 += y and not x
    p = binomtest(n10, n10+n01, .5).pvalue if n10+n01 else 1.
    return n11, n10, n01, n00, p


print('\nContrasts (pooled): Fisher exact (unpaired) and exact McNemar (paired by seed,slot)')
print('| A | B | A rate | B rate | diff | Fisher p | both ok / A-only / B-only / both fail | McNemar p |')
print('|---|---|---|---|---|---|---|---|')
for a, b in [(('planner', None, 2), ('planner', None, 4)), (('dreamer', 'sample', 2), ('dreamer', 'sample', 4)),
             (('dreamer', 'mode', 2), ('dreamer', 'mode', 4)), (('dreamer', 'sample', 2), ('dreamer', 'mode', 2)),
             (('dreamer', 'sample', 4), ('dreamer', 'mode', 4))]:
    (ka, na), (kb, nb) = pooled[a], pooled[b]
    _, pf = fisher_exact([[ka, na-ka], [kb, nb-kb]])
    n11, n10, n01, n00, pm = paired(a, b)
    lab = lambda c: '/'.join(str(x) for x in c if x is not None)
    print(f'| {lab(a)} | {lab(b)} | {ka/na:.3f} | {kb/nb:.3f} | {kb/nb-ka/na:+.3f} | {pf:.3g} | {n11}/{n10}/{n01}/{n00} | {pm:.3g} |')

print('\nPer-map successes pooled over seeds (16 episodes per map)')
maps = sorted(data[CELLS[0]+(SEEDS[0],)]['summary']['per_map'])
print('| cell | ' + ' | '.join(m.replace('segment_', '') for m in maps) + ' |')
print('|---|' + '---|'*len(maps))
for c in CELLS:
    row = []
    for mp in maps:
        k = sum(data[c+(s,)]['summary']['per_map'][mp]['successes'] for s in SEEDS)
        n = sum(data[c+(s,)]['summary']['per_map'][mp]['valid'] for s in SEEDS)
        row.append(f'{k}/{n}')
    print(f"| {c[0]}/{c[1] or '-'}/g{c[2]} | " + ' | '.join(row) + ' |')

print('\nMode-run mean policy stddev on active steps (dims: distance, angle, speed)')
for g in (2, 4):
    for s in SEEDS:
        print(g, s, [round(x, 3) for x in data[('dreamer', 'mode', g, s)]['mode_policy_stddev_mean_active']])
