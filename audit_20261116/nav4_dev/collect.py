import json, glob, os
D = 'audit_20261116/nav4_dev'
rows = []
for f in sorted(glob.glob(f'{D}/*.json')):
    r = json.load(open(f))
    if 'summary' not in r: continue
    s = r['summary']
    pm = {k[-4:]: f"{v['strict']}/{v['traverses']}" for k, v in s.get('per_map', {}).items()}
    print(os.path.basename(f)[:-5], f"{s['strict_completions']}/{s['traverses']}", 'codes', s['first_failure_codes'],
          'takeover %.3f' % s['takeover_fraction'], 'speed %.2f' % s['mean_speed'], 'mean_s %.1f' % s['mean_final_station'],
          'leak', (r.get('leakage') or {}).get('privileged_actor_carry_max_error'), 'loop', r['loop_steps'], 'sec %.0f' % r['seconds'])
    print('    ', pm)
