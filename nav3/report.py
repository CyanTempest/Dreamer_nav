"""Generate every numeric table from saved metrics, never from hand-entered scores."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    p=ROOT/'metrics'/name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x,digits=3):
    return '未完成' if x is None else f'{x:.{digits}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def restored_history(rows):
    """Keep raw logs, but do not plot updates rolled back by checkpoint restore."""
    result=[]
    for row in rows:
        if row.get('event')=='resume':
            result=[r for r in result if r.get('steps',0)<=row['steps']
                    and r.get('wall_seconds',0)<=row['wall_seconds']]
        result.append(row)
    return result


def normalized_iou(data):
    """Undefined masked IoU is null, including historical zero-union rows."""
    values=data.get('trav_iou')
    if values is None:return None
    rows=data.get('rows',[])
    counts=[r['trav_union'] for r in rows if r.get('trav_union') is not None]
    unions=np.sum(counts,axis=0) if counts else None
    return [None if (unions is not None and unions[k]==0) or (k==2 and not data['random_obstacles']) else v
            for k,v in enumerate(values)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary-output',default='metrics/evaluation_summary.json')
    parser.add_argument('--report-output',default='REPORT.md')
    parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args()
    metrics=ROOT/'metrics';figures=ROOT/'figures'
    evaluation=[]
    official={'planner','dreamer_s0','dreamer_s1','dreamer_s2','dreamer_random_s0','ppo_s0','ppo_s1','ppo_s2','dreamer_nohead','dreamer_amnesic','dreamer_direct'}
    for path in sorted(metrics.glob('eval_*.json')):
        data=json.loads(path.read_text());rows=data['rows']
        if data['method'] not in official:continue
        total_distance=sum(r['distance_m'] for r in rows)
        total_steps=sum(r['steps'] for r in rows)
        total_time=sum(r['duration_s'] for r in rows)
        summary=dict(method=data['method'],split=data['split'],obstacles=data['random_obstacles'],filter=data['filtered'],
            episodes=len(rows),mean_gates=float(np.mean([r['gates'] for r in rows])),success_rate=float(np.mean([r['success'] for r in rows])),
            fail_per_100m=sum(not r['success'] for r in rows)*100/max(total_distance,1e-6),
            takeover_rate=sum(r['takeovers'] for r in rows)/max(total_steps,1),mean_speed=total_distance/max(total_time,1),
            failures=data['failures'],trav_iou=normalized_iou(data),source=path.name)
        evaluation.append(summary)
    Path(args.summary_output).write_text(json.dumps(evaluation,indent=2))
    if args.tables_only:
        rows=[[e['method'],e['split'],int(e['obstacles']),int(e['filter']),e['episodes'],f"{e['success_rate']:.1%}",'/'.join('不适用' if v is None else fmt(v) for v in e['trav_iou']) if e['trav_iou'] is not None else '不适用'] for e in evaluation]
        Path(args.report_output).write_text(table(['方法','数据集','随机障碍','过滤','回合','全通过','IoU 全/遮挡/障碍邻域'],rows)+'\n')
        return
    train=[];curves={}
    checkpoint_progress=load('checkpoint_progress.json') or {}
    for p in sorted(metrics.glob('train_*.jsonl')):
        rows=restored_history([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
        if not rows:continue
        name=p.stem.removeprefix('train_');curves[name]=rows
        progress=ROOT/'checkpoints'/name/'progress.json'
        if progress.exists():checkpoint_progress[name]=json.loads(progress.read_text())
        end=checkpoint_progress.get(name,rows[-1])
        train.append([name,end.get('steps',0),end.get('updates',0),fmt(end.get('wall_seconds',0)/3600),
            next((r.get('prefill') for r in reversed(rows) if 'prefill' in r),0)])
    (metrics/'checkpoint_progress.json').write_text(json.dumps(checkpoint_progress,indent=2))
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for name,rows in curves.items():
            probes=[r for r in rows if r['event']=='probe']
            if not probes:continue
            axes[0].plot([r['steps'] for r in probes],[r['probe_return'] for r in probes],marker='.',label=name)
            axes[1].plot([r['wall_seconds']/3600 for r in probes],[r['probe_return'] for r in probes],marker='.',label=name)
            head=[r for r in probes if r.get('trav_iou')]
            if head:axes[2].plot([r['steps'] for r in head],[r['trav_iou'][0] for r in head],marker='.',label=name)
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[0].set(xlabel='Online environment transitions',ylabel='Fixed train-map probe return')
        axes[1].set(xlabel='Wall clock (hours)',ylabel='Fixed train-map probe return')
        axes[2].set(xlabel='Online environment transitions',ylabel='Traversability IoU')
        fig.savefig(figures/'training_curves.png',dpi=160);plt.close(fig)
    # Same axes for every mini-trail condition; show all trajectories, not a selected best run.
    for ob in (False,True):
        for sf in (False,True):
            paths=[]
            for path in metrics.glob('eval_*_test_*.json'):
                d=json.loads(path.read_text())
                if d['random_obstacles']==ob and d['filtered']==sf:paths.append(d)
            if not paths:continue
            fig,axes=plt.subplots(1,len(paths),figsize=(max(5,len(paths)*3),5),squeeze=False,layout='constrained')
            for ax,data in zip(axes[0],paths):
                for row in data['rows']:
                    xy=np.array(row['trajectory'])
                    if len(xy):ax.plot(xy[:,0],xy[:,1],alpha=.2,color='#176f79' if row['success'] else '#bd492f',lw=.8)
                ax.set(title=data['method'],xlabel='Map x (m)',ylabel='Map y (m)',aspect='equal');ax.grid(alpha=.15)
            fig.suptitle(f'Mini trail, random obstacles={ob}, safety filter={sf}')
            fig.savefig(figures/f'trajectories_mini_ob{int(ob)}_sf{int(sf)}.png',dpi=150);plt.close(fig)
    g1,g2,profile,noise,audit=[load(p) for p in ['g1.json','g2.json','profile.json','noise_comparison.json','audit.json']]
    report=['# nav3 第三轮导航实验报告','',
        '本报告由 `python -m nav3.report` 从 `metrics/` 生成。未完成和未通过不计作验收成功；预测、回报和对照数值均不手填。','',
        '## 实现与范围','',
        '实现了 JAX 批量 2.5D 快环境、标定双雷达极坐标遮挡近似、64×64 六通道 BEV、随机游走与累积衰减、延迟、局部路径点动作和 20 Hz 全向跟踪器。机体动力学参数统一放在配置中。地图复用已验收生成器，静态物体栅格化；随机石块与短倒木每回合重采样且不进入路线先验。',
        'Dreamer 使用官方 Agent、size12m、16×64 batch、百万容量 Replay 和 ratio 32。上游改动包括禁止 `priv_` 输入编码器，以及无损压缩 Replay、尾块回收和断点恢复工程修复；Agent 更新与原采样器不变。解码头通过外部适配器读出。PPO 使用 CNN+GRU，与 Dreamer 共用环境、观测和默认目标动作。','',
        '## 闸门与吞吐','']
    gates=[]
    for name,g in [('G1',g1),('G2',g2)]:
        gates.append([name,'未完成' if not g else ('通过' if g['passed'] else '未通过'),f'`metrics/{name.lower()}.json`'])
    report.append(table(['闸门','结论','证据'],gates));report.append('')
    if g1:report.append(f"G1：{g1['episodes']} 回合，平均 {g1['mean_gates']:.3f}/8 门，全通过率 {g1['success_rate']:.2%}；门槛 {g1['threshold']:.0%}。")
    if g2:
        rr=[]
        for name in ['dreamer_s0','ppo_s0']:
            d=g2[name];rr.append([name,fmt(d.get('initial_return')),fmt(d.get('final_return')),fmt(d.get('initial_iou')),fmt(d.get('final_iou')),d.get('online_steps','未完成')])
        report.extend(['',table(['方法','初始探针回报','最终探针回报','初始 IoU','最终 IoU','在线步数'],rr),'',
                       'G2 使用固定训练地图的 250 步探针，判定回报与 IoU 是否增加。它只验证学习信号，不替代整段留出评估。'])
    if profile:
        values=[r['env_steps_per_second'] for r in profile['trials']]
        report.append(f"\n吞吐：{profile['envs']} 并行，三次均值 {np.mean(values):,.0f} 环境步/秒，范围 {min(values):,.0f}–{max(values):,.0f}；JIT 编译 {profile['compile_seconds']:.1f} 秒。包含观测和特权标签，不包含神经网络训练。")
    report.extend(['','## 训练记账','',table(['运行','在线环境步','更新步','墙钟小时','预填步'],train) if train else '尚无完成的训练指标。','',
        '累计步数、更新数和墙钟以最近已保存检查点为准（`metrics/checkpoint_progress.json`）。墙钟包含该训练进程的初始化、探针和已记账存档时间。预填单独列出；不能把 replay 容量解释成实际已有百万步。','',
    ])
    resources=load('training_resources_summary.json')
    if resources:
        report.extend([table(['方法','环境数','显存观测峰值','RSS 观测峰值 GB','末次 RSS GB','交换峰值 B'],[
            [name,r['envs'],f"{r['gpu_peak_fraction']:.2%}",fmt(r['rss_observed_peak_gb']),fmt(r['rss_final_record_gb']),r['swap_peak_bytes']]
            for name,r in resources.items()]),'',
            '资源统计来自 `metrics/training_resources_summary.json`。RSS 包含恢复各进程的采样及内核高水位，显存来自每分钟遥测。Dreamer 最终定期归还分配器空闲页的进程单独记录峰值为 '+fmt(resources['dreamer_s0']['final_process_rss_peak_gb'])+' GB。',''])
    report.extend(['## 同条件评估',''])
    rows=[]
    for e in evaluation:
        iou=e['trav_iou']
        rows.append([e['method'],e['split'],int(e['obstacles']),int(e['filter']),e['episodes'],fmt(e['mean_gates']),f"{e['success_rate']:.1%}",fmt(e['fail_per_100m']),f"{e['takeover_rate']:.1%}",fmt(e['mean_speed']),('/'.join('不适用' if v is None else fmt(v) for v in iou) if iou else '不适用')])
    report.append(table(['方法','数据集','随机障碍','过滤','回合','平均门/8','全通过','失败/100m','接管率','速度 m/s','IoU 全/遮挡/障碍邻域'],rows) if rows else '尚无有效评估结果。')
    edge_cases=load('evaluation_edge_cases.json')
    if edge_cases and edge_cases['count']:
        report.extend(['',f"指标口径：8 门通过率沿用 `gates == 8`。本次有 {edge_cases['count']} 回合在达到 8 门的终止步同时记录了非成功终止原因，因此通过回合数与失败分类中的 success 数可能相差；原始记录保留在 `metrics/evaluation_edge_cases.json`，没有修改环境或评估定义。"])
    report.extend(['','失败分类及逐回合轨迹见 `metrics/evaluation_summary.json` 和各 `eval_*.json`。每 100 m 失败数以实际行驶总距离为分母；接管率以动作次数为分母。','',
        '## 达到规划器 90% 水平',''])
    efficiency=[]
    baseline=next((e for e in evaluation if e['method']=='planner' and e['split']=='heldout' and not e['obstacles'] and not e['filter']),None)
    if baseline:
        for path in sorted(metrics.glob('efficiency_*.jsonl')):
            series=[json.loads(line) for line in path.read_text().splitlines() if line]
            for metric in ['mean_gates','success_rate']:
                hits=[r for r in series if r[metric]>=.9*baseline[metric]]
                first=min(hits,key=lambda r:r['steps']) if hits else None
                efficiency.append([series[0]['method'],metric,first['steps'] if first else '未达到',fmt(first['wall_seconds']/3600) if first else '未达到',series[-1]['steps']])
    report.extend([table(['方法','阈值依据','首次达到的在线步数','训练墙钟小时','最近测量步数'],efficiency) if efficiency else '尚无整段中间检查点评估。',
        '这些是固定间隔评估首次观测到达到阈值的上界，非精确交叉时间；使用无随机障碍、无过滤的 100 回合留出集。','',
        '## 噪声模型与完整射线',''])
    if noise:
        ag=noise['aggregate'];report.append(table(['位姿数','射线未知率','快模型未知率','公共地面格高程 MAE(m)','未知阴影 IoU'],[[ag['poses'],fmt(ag['ray_unknown']),fmt(ag['fast_unknown']),fmt(ag['height_mae_m']),fmt(ag['shadow_iou'])]]))
        report.extend(['',table(['类别','参考目标格数','射线格检出率','快模型格检出率'],[[k,ag[k]['target_cells'],fmt(ag[k]['ray_cell_recall']),fmt(ag[k]['fast_cell_recall'])] for k in ['rail','tape','log']]),'',
            '参考每台 96×900 全分辨率射线，采用标定旋转和平移，base 离地暂设 0.54 m。高程误差只在双方共同观测到地面的位置计算，不把未知填零。细物体统计以栅格覆盖为单位，不能解释为实机语义识别召回率。'])
    else:report.append('完整射线核对尚未完成；不能声称噪声模型已校准。')
    report.extend(['','## 审计、限制与未完成项','',
        '地图 manifest、训练使用种子、特权信息替换检查、上游分支和只读目录文件统计由审计脚本核对。',
        f"审计状态：{'未完成' if audit is None else ('通过' if audit['passed'] else '未通过')}。",'',
        '已知限制：2.5D top 高度近似无法表达栏杆和悬空导引带下面的空隙，细杆保守栅格化有膨胀；快模型尚未实现机体俯仰与完整腿部自遮挡。静态射线对比未验证运动累积。支撑区来自路线场景定义；并不代表实机地表可通行真值。训练地图是同一生成器的随机特征排列，泛化范围有限。',''])
    expected=['planner','dreamer_s0','ppo_s0']
    completeness=[]
    for name in expected:
        entries=[e for e in evaluation if e['method']==name]
        complete=len(entries)==8 and all(e['episodes']>=(30 if e['split']=='test' else 100) for e in entries)
        completeness.append([name,'不适用' if name=='planner' else ('有日志' if name in curves else '未训练'),len(entries),'完整' if complete else '未完成'])
    report.append(table(['配置','训练记录','已评估条件/8','验收状态'],completeness))
    report.extend(['','本次队列仅含 Dreamer s0、PPO s0 各累计 5 小时及规划器对照评估。额外种子、随机预填对照和消融按用户要求暂缓，不属于本次队列缺项。',
        '消融结论只有完成对应对照后才成立；缺失时不推断记忆或可通行头有益。短探针不用于计算达到规划器 90% 水平的样本效率。',
        '原项目仍待完成：官方训练循环接入（当前为官方 Agent/Replay 加自定义批量调度）、独立进程安全过滤及失联停止、A2 慢环境完整闭环与标定、动作契约统一，以及 PyTorch/TensorRT 导出与逐步对齐。本次 OOM 恢复没有实现这些项目，也不代表整套最终设计验收完成。'])
    recovery=load('oom_recovery_summary.json')
    if recovery:
        report.extend(['','## OOM 恢复与内存验证','',
            f"Dreamer 保留 {recovery['dreamer_envs']} 个真实环境；无损压缩 Replay 与原 RAM 版已逐值对照。恢复源为 1,299,872 步检查点，旧 Replay 实际恢复 812,960 个采样起点，原始文件保留。",'',
            table(['实跑分钟','RSS 最小/最大 GB','末段 RSS GB','逻辑 Replay 最大 GB','显存峰值'],[[
                fmt(recovery['validation_seconds']/60),
                f"{recovery['rss_min_gb']:.3f} / {recovery['rss_max_gb']:.3f}",
                fmt(recovery['rss_last_gb']),fmt(recovery['replay_max_gb']),
                f"{recovery['gpu_peak_fraction']:.2%}"]]),'',
            '最终使用 LZ4 内存块，驻留内存与压缩数据字节分别统计；早期文件映射试验因磁盘换入频繁而停用。RSS 看门狗阈值 50 GB，超过时存档并以 75 退出。证据见 `metrics/oom_recovery_summary.json`、`memory_dreamer_s0.jsonl`、`queue_telemetry.jsonl` 和 `figures/oom_recovery_memory.png`。'])
    sessions=load('remote_status.json')
    if sessions:report.extend(['','远端最近记录的 tmux 会话：`'+sessions.get('tmux','').replace('\n','; ')+'`。'])
    Path(args.report_output).write_text('\n'.join(report)+'\n')
    print('Generated REPORT.md and evaluation_summary.json from saved metrics.')


if __name__=='__main__':main()
