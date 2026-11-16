"""Explain the observed PPO footprint and choose the requested compatible scale."""
import json
from pathlib import Path
old=json.loads(Path('metrics/config_ppo_s0.json').read_text())
rows=[]
for n in [128,640]:
    p=Path(f'metrics/lwy_runtime2_memory_ppo_{n}.json')
    d=json.loads(p.read_text()) if p.exists() else {}
    peaks={}
    for r in d.get('samples',[]):peaks[r['stage']]=max(peaks.get(r['stage'],0),r['used_mib'])
    rows.append(dict(envs=n,completed=d.get('completed',False),finite_loss=d.get('finite_loss',False),
        peak_used_mib=d.get('peak_used_mib'),stage_peak_mib=peaks,error=d.get('error'),
        artifact=str(p),training_runtime=d.get('training_runtime')))
assert rows[0]['completed'] and rows[0]['finite_loss']
selected=640 if rows[1]['completed'] and rows[1]['finite_loss'] else 128
explanation=('原ppo_s0实际环境数为640，上一轮报告称128为原配置不准确。PPO模型仅741,015参数；显存不是由参数量或环境数单独决定。'
 '新任务常驻全部200张LWY裁片（约4.90GiB）和200张旧图（约1.33GiB），还包括128步rollout的BEV、循环状态、序列重排/堆叠临时张量、反向计算及默认BFC分配器保留池。'
 'nvidia-smi统计的是整卡占用，包括约911MiB桌面基线，不能把17,717MiB都称作128个环境的状态显存。'
 +('本次原640环境完整测量通过，按用户条件恢复640，学习率、rollout128、sequence32、epochs4、minibatch16和GRU256均保持原配置。' if selected==640 else
   '本次640环境在完整新旧地图+Agent+优化器测试中失败，128通过，因此维持用户指定的128，属于新增地图/采集缓冲的显存限制，不能声称训练规模与原ppo_s0完全一致；其余算法配置保持原值。'))
result=dict(original_envs=old['envs'],prepared_envs=128,selected_envs=selected,measurements=rows,explanation=explanation)
with Path('metrics/lwy_runtime2_ppo_memory_analysis.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
