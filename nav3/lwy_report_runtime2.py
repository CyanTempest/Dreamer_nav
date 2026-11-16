"""Render the approved run from new artifacts; exclude invalid neural results."""
from datetime import datetime, timezone
import json
from pathlib import Path

MARKER='\n<a id="lwy-runtime2"></a>\n'
PREFIX='lwy_runtime2'

def read(path,default=None):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else default

def link(path):return f'[{Path(path).name}]({path})'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    path=Path('LWY_REPORT.md');old=path.read_text().split(MARKER)[0]
    # Historical reports remain, but invalid level-0 neural results must never
    # appear in a comparison table, even inside the retained historical text.
    old='\n'.join(line for line in old.splitlines() if not (line.startswith(('| dreamer_s0 | blind |','| dreamer_s0 | original |','| ppo_s0 | blind |','| ppo_s0 | original |'))))
    if old.startswith('> **第六轮'):old=old[old.index('\n# nav3 第六轮'):].lstrip('\n')
    old=old.replace('已产生 **5/80** 个最终条件结果。','此前产生的神经网络结果已全部作废，不能计入正式最终条件。')
    old=old.replace('注意：下表神经模型结果使用已确认存在数值异常的level0推理，不可据此单独判断模型泛化能力；决定前仅保留为诊断记录。','注意：此前level0神经模型结果已作废并从下表移除，原始JSON仅作故障记录。')
    q=read(f'metrics/{PREFIX}_queue.json',{});selection=read(f'metrics/{PREFIX}_model_selection.json',{})
    evs=[(p,read(p)) for p in sorted(Path('metrics').glob(f'{PREFIX}_final_*_rt1.json'))]
    flags='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
    assert all(r['runtime']['xla_flags']==flags and r['runtime']['policy_batch_width']==100 for _,r in evs)
    status=q.get('status','准备启动');text=['# 第六轮续 2：正式训练与最终评估',f'更新时间：{datetime.now(timezone.utc).isoformat()}；队列状态：`{status}`。',
        '## 协议修正与历史结果作废',
        '验收方已批准所有方法（Dreamer、PPO、规划器）的训练与评估统一使用 `--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true`，并设 `TF_CUDNN_DETERMINISTIC=1`。此决定替代 TASK_LWY2.md 第1.5条旧设置要求。评估继续使用每回合独立PRNG、单环境vmap、固定100条策略流。',
        '原因：本机level0导致Dreamer可通行图恒为0.48828125、原始梯度约10¹⁵；冻结同一批次改为level1后恢复正常。level1已通过600条完整轨迹重复一致性及G1复核。所有level0神经网络评估均作废，包括旧模型零样本0/100；作废结果不进入任何对比表，在新协议下全部重测。规划器G1与autotune无关，原版10/10、盲测20/20继续有效。'+link('metrics/lwy_v2_evaluation_runtime_decision.json')+'。',
        '第五轮0.5–1小时的续训崩溃不是autotune问题：当时训练、评估都是level1。此次保留踩出路面失败位置/动作、奖励分量及新旧replay采样比例，用于识别相同的行为退化。',
        '## 初始化、规模与保护机制',
        '正式训练分别从原始健康 `dreamer_s0`、`ppo_s0` 只读复制至 `checkpoints/dreamer_lwy_runtime2/`、`checkpoints/ppo_lwy_runtime2/`。已有冒烟检查点和旧队列均保持不变；本次重新从原始权重开始。保留优化器、Dreamer旧replay的100,000个序列起点；正式阶段的段间恢复保存环境、PRNG、策略和replay状态。冒烟/恢复证据沿用上一轮已完成的两段短运行，PPO新协议另经完整地图+Agent+优化器实测。',
        'Dreamer：4608环境，size12m，batch16×64，train ratio32，官方训练循环。每条训练计划8段×30分钟，训练不开过滤。每段复制独立快照并验证20图×2回合；原始梯度>10¹²或非有限立即停止。连续两次比最佳低超过30个百分点，或连续两次通过率下降，均停止该方法并诊断，不自动重启。最终按验证全通过率、平均门数、较早快照依次选择，测试集不参与选择。']
    memory=read('metrics/lwy_runtime2_ppo_memory_analysis.json')
    if memory:
        text += [memory['explanation'],link('metrics/lwy_runtime2_ppo_memory_analysis.json')+'。',table(['环境数','完成','峰值MiB','各阶段峰值MiB'],[[r['envs'],r['completed'],r['peak_used_mib'],json.dumps(r['stage_peak_mib'],ensure_ascii=False)] for r in memory['measurements']])]
    charged=q.get('gpu_job_wall_seconds',0)+q.get('preceding_g1_and_diagnostics_allowance_seconds',0)
    text += [f'已记账GPU占用 {charged/3600:.3f} 小时 / 12小时；含此前准备、诊断、冒烟和本次显存检查。两个评估进程并行时按占用墙钟并集计账。独立tmux队列有互斥检查、每子作业超时及预算上限。',
        '## 训练曲线与模型选择']
    training=q.get('training',{})
    text.append(table(['方法','快照h','验证成功','平均门','停止原因'],[[run.get('run_name',name),r['label'],f"{r['success_rate']:.1%}",f"{r['mean_gates']:.3f}",run.get('stop_reason','')] for name,run in training.items() for r in run['rounds']]))
    if training:
        plot(training);text.append('![验证曲线](figures/lwy_runtime2_learning_curve.png)')
    if selection:
        text.append(table(['最终方法','选中快照','验证成功','平均门'],[[name,r['checkpoint'],f"{r['success_rate']:.1%}",f"{r['mean_gates']:.3f}"] for name,r in selection['selected'].items()]))
        text.append(link(f'metrics/{PREFIX}_model_selection.json')+'。')
    else:text.append('尚未完成模型选择。')
    diagnoses=sorted(Path('metrics').glob('lwy_v2_collapse_*_runtime2_*.json'))
    text += ['## 崩溃诊断', '\n'.join(link(p) for p in diagnoses) if diagnoses else '尚未触发验证崩溃早停。数值保护与行为退化按不同证据判断。',
        '## 最终对比',f'已完成 {len(evs)}/80 个条件；完整矩阵为五方法×四数据集×随机障碍/过滤四条件。原版、盲测、旧留出各100回合，mini30回合。所有下表数字由新协议JSON生成。']
    if evs:
        text.append(table(['方法','数据集','障碍/过滤','N','全通过率 [Wilson95%]','平均门','失败分类','接管率','速度m/s','IoU全/遮挡/障碍'],[
            [r['method'],r['dataset'],f"{int(r['random_obstacles'])}/{int(r['safety_filter'])}",r['episodes'],f"{r['success_rate']:.1%} [{r['success_wilson95'][0]:.1%},{r['success_wilson95'][1]:.1%}]",f"{r['mean_gates']:.3f}",json.dumps(r.get('failures',r.get('failure_counts',{})),ensure_ascii=False),f"{r['takeover_rate']:.2%}",f"{r['mean_speed_mps']:.3f}",'不适用' if r['trav_iou'] is None else '/'.join('无定义' if v is None else f'{v:.3f}' for v in r['trav_iou'])] for _,r in evs]))
        text.append('逐回合失败s、对应1km区段、动作与奖励分量在 `metrics/lwy_runtime2_final_*_rt1.json`；IoU每20策略步采样，空并集记为无定义。')
    video=read(f'metrics/{PREFIX}_dreamer_vs_ppo_rt1.json')
    text += ['## 视频与检查',link(video['path'])+'：同种子、不加过滤，原版XML回放快环境真实推理位姿；不是腿部动力学验证。' if video else '推理对比视频待最终评估后生成。']
    for name in ['check_diag','checks','cleanup']:
        p=f'metrics/{PREFIX}_{name}_final.json';d=read(p)
        if d:text.append(link(p)+'：'+json.dumps(d,ensure_ascii=False))
    text += ['## 未解决问题与产物',
        '地图仍是视频近似，未经实测；保守栅格、离散通路和快环境动力学的限制沿用上文。仅一个训练种子；没有以测试成绩调参。若学习策略低于规划器，按实际结果汇报，不用过滤成绩代替无过滤能力。',
        '队列状态：'+link('metrics/lwy_runtime2_queue.json')+'；训练日志 `metrics/train_*_lwy_runtime2.jsonl`；快照、replay与大型地图只留远端 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`。报告、指标、图、日志、小视频同步回Mac。',
        '本次改动：`lwy_training_runtime.py`统一批准设置；`lwy_queue_v2.py --released`隔离正式运行与预算；`lwy_train_dreamer.py`支持独立目录；`train_ppo.py`逐更新检查原始梯度；`lwy_memory_v2.py`隔离测量输出；`lwy_collapse_v2.py`读取指定运行；`lwy_video_v2.py`读取指定模型及参考轨迹；`lwy_report_runtime2.py`从有效结果生成本节。']
    if q.get('error'):text.append('队列原始错误：`'+q['error']+'`；日志：`'+str(q.get('active_log') or q.get('active_logs'))+'`。')
    path.write_text(f'> **第六轮续2最新状态**：{status}；G1通过；有效最终评估 {len(evs)}/80。见[最新结果](#lwy-runtime2)。以下历史记录以最新协议修正为准。\n\n'+old.rstrip()+MARKER+'\n\n'.join(text)+'\n')
    print(json.dumps(dict(status=status,final_conditions=len(evs))))

def plot(training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,data in training.items():
        rows=data['rounds'];x=[float(r['label']) for r in rows]
        axes[0].plot(x,[r['success_rate']*100 for r in rows],'-o',label=name)
        axes[1].plot(x,[r['mean_gates'] for r in rows],'-o',label=name)
    for ax in axes:ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Formal training wall h')
    axes[0].set_ylabel('Validation success (%)');axes[0].set_ylim(0,100)
    axes[1].set_ylabel('Mean gates');axes[1].set_ylim(0,8)
    fig.tight_layout();fig.savefig('figures/lwy_runtime2_learning_curve.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
