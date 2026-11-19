"""Independent 22-CU closer. Do not stop any other named allocation."""
import re
import subprocess
import time
import os
from pathlib import Path
base=Path(os.environ['MATRIX_COLLECTION_DIR']).resolve()
cli=str(Path.home()/'.local/bin/colab')
initial=float(re.search(r'balance: ([\d.]+)',(base/'logs/usage_initial.log').read_text())[1])
last=None
with (base/'logs/guard22.log').open('a',buffering=1) as log:
    while not (base/'logs/closed').exists():
        try:
            report=subprocess.check_output([cli,'usage'],text=True,timeout=25,stderr=subprocess.STDOUT)
            bal=float(re.search(r'balance: ([\d.]+)',report)[1])
            rate=float(re.search(r'rate: ([\d.]+)',report)[1])
            now=time.monotonic()
            # Integrate rates independently of delayed billing snapshots.
            if last is None:
                estimate=initial-bal
            else:
                estimate=max(initial-bal,estimate+(now-last)*prior_rate/3600)
            last,prior_rate=now,rate
            print(time.strftime('%FT%T'),bal,rate,round(estimate,3),file=log)
            if max(estimate,initial-bal)+rate*120/3600>=22:
                (base/'logs/budget_stop').touch()
                sessions=subprocess.check_output([cli,'sessions'],text=True)
                for label in set(re.findall(r'\[(mv4col[\w-]*)\]',sessions)):
                    subprocess.run([cli,'stop','-s',label],timeout=45,stdout=log,stderr=log)
        except Exception as e:
            print(str(e),file=log)
        time.sleep(25)
