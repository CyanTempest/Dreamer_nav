"""Run the actual queue with fake child processes to verify failure isolation."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from nav3 import runner
from nav3.memory import MemoryMonitor


def scenario(dreamer_codes):
    original=Path.cwd()
    with tempfile.TemporaryDirectory(dir=original/'tmp') as folder:
        os.chdir(folder)
        try:
            for name in ['metrics','logs','configs','checkpoints/dreamer_s0','checkpoints/ppo_s0']:
                Path(name).mkdir(parents=True,exist_ok=True)
            Path('metrics/g2.json').write_text('{"passed":true}')
            Path('metrics/jobs.json').write_text('[{"status":"failed","wall_seconds":999999}]')
            Path('configs/runtime.json').write_text('{"dreamer_envs":4608,"ppo_envs":1024}')
            for method in ['dreamer','ppo']:
                Path(f'checkpoints/{method}_s0/progress.json').write_text('{"steps":114688,"wall_seconds":0}')
            commands=[];codes=iter(dreamer_codes)
            class Child:
                pid=123
                def __init__(self,command,**kwargs):
                    self.command=command;commands.append(command)
                def wait(self,**kwargs):
                    module=self.command[2]
                    code=next(codes) if module=='nav3.train_dreamer' else 0
                    if code==0 and module.startswith('nav3.train_'):
                        name=self.command[self.command.index('--name')+1]
                        Path(f'checkpoints/{name}/progress.json').write_text('{"steps":999999,"wall_seconds":18001}')
                    return code
            with patch.object(runner,'require_g1'),patch.object(runner.sys,'argv',['runner']),\
                 patch.object(runner.subprocess,'Popen',Child),\
                 patch.object(runner.subprocess,'check_output',return_value='24000\n'),\
                 patch.object(runner.subprocess,'run'),contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            assert sum(c[2]=='nav3.train_dreamer' for c in commands)==3
            assert sum(c[2]=='nav3.train_ppo' for c in commands)==1
            assert sum(c[2]=='nav3.eval_model' for c in commands)==3
            assert '--resume' in commands[0]
            monitor=MemoryMonitor('test')
            with patch('nav3.memory.rss_bytes',return_value=50_000_000_001),contextlib.redirect_stdout(io.StringIO()):
                assert monitor.check('train',1299872,force=True)
            assert json.loads(Path('metrics/memory_test.jsonl').read_text())['rss_gb']>50
            transient=MemoryMonitor('transient')
            with patch('nav3.memory.rss_bytes',side_effect=[51_000_000_000,30_000_000_000]),\
                 patch('nav3.memory.release_host_memory'),contextlib.redirect_stdout(io.StringIO()):
                assert transient.check('train',1299872,force=True)
            row=json.loads(Path('metrics/memory_transient.jsonl').read_text())
            assert row['rss_before_trim_bytes']==51_000_000_000 and row['rss_gb']==30
        finally:
            os.chdir(original)


Path('tmp').mkdir(exist_ok=True)
scenario([-9,75,0])
scenario([-9,-9,-9])
print('PASS: retry from checkpoint; max three failures; PPO/evaluation continue; historic failures do not consume budget; RSS watchdog threshold')
