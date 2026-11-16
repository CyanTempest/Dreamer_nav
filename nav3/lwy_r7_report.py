"""Append/update only round seven; tables always derive from saved artifacts."""
import json
from pathlib import Path
import pickle
import numpy as np

MARKER='\n<!-- LWY_R7_START -->\n'


def read(path):return json.loads(Path(path).read_text())


def main():
    state=read('metrics/lwy_r7_queue.json') if Path('metrics/lwy_r7_queue.json').exists() else {}
    lines=['# 第七轮','',f"状态：`{state.get('status','随机障碍闸门准备中')}`。仅在 Mac nav3 与指定远端 exp_nav3 内工作。",'',
        '## 随机障碍修复与闸门','',
        '第六轮正式训练/评估使用 SafeLwyEnv 的离散可行性目录，不是直接使用 LwyEnv 的原始提案。旧目录提供离散几何通路，却不足以保证带噪声规划器绕行通过。原版 ob1 的旧逐回合记录是全部已过 1 门再失败，任务描述中的“第 1 门之前”与原始数据不一致。','',
        '本轮入口 `LwyEnvR7` 默认 `obstacle_rule=shoulder`：障碍保留原尺寸、朝向、沿程位置和左右侧，仅向路肩外移至名义侵入路面 4 cm。不扩展支撑区；包络、栅格保守量、支撑/碰撞终止条件和运动学函数全部保持原样。对每套摆放，复核全部 0.1 m 离散见证位姿与新障碍矩形的分离轴，并保留原静态栅格证书；另报告障碍附近中心线路径的真实栅格余量。离散证书仍有采样误差限制，规划器闸门另外实测。','',
        '新评估入口 `python -m nav3.lwy_compare_r7` 默认修复规则，`--obstacle-rule legacy` 可复现第六轮规则；历史代码和目录不改。旧宽路 heldout/mini 保持原已检查的摆放规则。','']
    if Path('metrics/lwy_r7_gate.json').exists():
        g=read('metrics/lwy_r7_gate.json');lines += [f"随机障碍闸门：原版 {g['original_successes']}/{g['original_episodes']}；盲测 {g['blind_successes']}/{g['blind_episodes']}；通过={g['passed']}。",'']
    if Path('metrics/lwy_r7_obstacle_geometry.json').exists():
        geo=read('metrics/lwy_r7_obstacle_geometry.json')
        dimensions=[2*v for row in geo['rows'] for v in row['half_sizes_m']]
        lines += [f"实测栅格分辨率 {geo['resolution_m']:.3f} m，保守量 PAD={geo['conservative_pad_m']:.4f} m。支撑层存储 edge + 0.025 + PAD，而终止阈值是 0.025；两者抵消，阈值不能当作额外可驶路肩。",'',
            f"原版目录 8 套×3 障碍的矩形完整边长范围为 {min(dimensions):.3f}–{max(dimensions):.3f} m；沿程位置、原/新横向位置与尺寸逐项列在 `metrics/lwy_r7_obstacle_geometry.json`。",'',
            '| 摆放规则 | 原版障碍切片几何通道最小m | 可行机体中心区间宽度中位m | 最小m |', '| --- | --- | --- | --- |']
        for rule in ['legacy','shoulder']:
            v=[r[rule] for r in geo['rows']]
            lines.append(f"| {rule} | {min(x['geometric_passage_m'] for x in v):.3f} | {np.median([x['admissible_center_span_m'] for x in v]):.3f} | {min(x['admissible_center_span_m'] for x in v):.3f} |")
        lines += ['','切片只说明横向余量，不单独证明路径连通；连通性与噪声跟踪分别由完整离散证书和规划器闸门验证。','']
    lines += ['## 退化诊断','',
        '旧日志中 actor 熵字段为 `train/ent/action`，旧崩溃提取器漏掉了它。3–4 h：熵约 0.721→0.527；BEV 损失 238.41→209.15、priv_trav 损失 14.43→12.66、价值损失 1.019→0.917；梯度约 775→790。旧 replay 样本比例已经为 0。世界模型损失下降和有限梯度不能保证闭环策略稳定；旧数据耗尽与失败相关，但不能单凭相关性断言因果。','',
        '3 h 官方完整状态仍在。replay 清单中 4608 个无对应磁盘文件的记录长度全部为 0，恢复器按原逻辑重建空块，其余非空历史分块完整。新运行复制精确 3 h replay/优化器/随机数/采集器状态，不使用 4 h 的后续样本。为适应共享 GPU，仅保留固定的 2048 个采集器前缀，不同步重置；三条对照保持相同缩减。本轮 replay_old 指 3 h 快照已有数据，历史日志 replay_old 指最早 dreamer_s0 数据，二者口径不同。','',
        '短对照：原学习率 4e-5 + online=True；仅学习率降为 1e-5；仅 online=False。每条计划 12 min，验证 20 图×4 回合。有限预算和单次干预不足时，不宣称唯一根因。','']
    if Path('metrics/lwy_r7_return_scales.json').exists():
        lines += ['| 快照h | 回报归一化尺度 |','| --- | --- |']
        for row in read('metrics/lwy_r7_return_scales.json'):lines.append(f"| {row['hours']:.1f} | {row['scale']:.3f} |")
        lines += ['','三个快照的尺度下降约 8%，未显示大幅跳变；只有三个采样点，不能排除中间的瞬时波动。不能将归一化后的 ret 值误当成尺度。','']
    if Path('metrics/lwy_r7_historical_analysis.json').exists():
        history=read('metrics/lwy_r7_historical_analysis.json')
        old_ob=history['failures']['planner_original_ob1']
        lines += [f"旧 ob1 原版规划器失败都在第三区段 root_and_sign_bends，s={min(r['s'] for r in old_ob):.1f}–{max(r['s'] for r in old_ob):.1f} m，中位 {np.median([r['s'] for r in old_ob]):.1f} m；末态横向偏移绝对值中位 {np.median([abs(r['lateral_m']) for r in old_ob]):.3f} m。几何证书虽可行，带定位误差的绕行所需动态余量不足，不能将 0/100 简化为每套几何都绝对堵死。",'']
        lines += ['| 历史指标 | 约3.00h | 最后日志约4.03h |','| --- | --- | --- |']
        for key,series in history['metric_series'].items():
            if '/loss/' in key or key in ['train/ent/action','train/opt/grad_norm']:
                lines.append(f"| {key} | {series[0][1]:.6g} | {series[-1][1]:.6g} |")
        lines += ['', '| 历史验证快照 | 支撑失败总数 | s=970–990m | s≥990m | s=150–170m |', '| --- | --- | --- | --- | --- |']
        for name,label in [('best','3.0h'),('last','4.0h')]:
            failures=[r for r in history['previous_collapse'][name]['failures'] if r['failure']=='support_or_drop']
            lines.append(f"| {label} | {len(failures)} | {sum(970<=r['progress_m']<990 for r in failures)} | {sum(r['progress_m']>=990 for r in failures)} | {sum(150<=r['progress_m']<170 for r in failures)} |")
        lines += ['','上表是历史每图2回合协议，只用于分析同一历史验证集的退化；本轮选模统一使用每图4回合，不将两种样本数混作提升。','']
        lines += ['### 既有失败位置与动作','', '| 数据集 | 失败类型 | 区段 | 次数 | 横向偏移绝对值中位m | 请求速度均值m/s |', '| --- | --- | --- | --- | --- | --- |']
        for dataset in ['original','blind']:
            failures=history['failures']['dreamer_lwy_'+dataset+'_ob0'];groups={}
            for r in failures:
                section=r['section'] or {}
                label=str(section.get('index',-1)+1)+' '+str(section.get('label','unknown'))
                groups.setdefault((r['failure'],label),[]).append(r)
            for (failure,section),rows in groups.items():
                lines.append(f"| {dataset} | {failure} | {section} | {len(rows)} | {np.median([abs(r['lateral_m']) for r in rows]):.3f} | {np.mean([r['requested_speed_mps'] for r in rows]):.3f} |")
        lines += ['','旧轨迹每5秒采样，不能据此认定5Hz动作振荡；本轮补录保留失败前25秒逐策略步历史，原始目标点、速度档和横向偏移见对应分析 JSON。','']
    if state.get('diagnosis'):
        lines += ['| 对照 | 验证全通过率 | 平均门数 |','| --- | --- | --- |']
        for r in state['diagnosis']['arms']:lines.append(f"| {r['config']} | {r['success_rate']:.1%} | {r['mean_gates']:.3f} |")
        lines += ['',f"按验证结果选择续训配置 `{state['diagnosis']['selected']}`。",'']
    if Path('metrics/lwy_r7_diagnosis_analysis.json').exists():
        diagnostic=read('metrics/lwy_r7_diagnosis_analysis.json')
        lines += ['| 对照 | 实际更新数 | GPU进程秒（含恢复） | 相对未续训新增成功/新增失败 | 相对原学习率新增成功/新增失败 |',
            '| --- | --- | --- | --- | --- |']
        for arm in diagnostic['arms']:
            initial=arm['versus_initial'];control=arm['versus_baseline']
            lines.append(f"| {arm['config']} | {arm['updates']} | {arm['process_seconds']:.1f} | {initial['gained']}/{initial['lost']} | {control['gained']}/{control['lost']} |")
        lines += ['','各臂时间包含初始化与 replay 恢复，实际更新数见表；更新量差异是本次墙钟对照的限制。逐回合配对结果与探索性 McNemar 检验保存在 `metrics/lwy_r7_diagnosis_analysis.json`，单次验证和多个候选不能证明唯一因果。','']
        lines += ['| 对照末态 | actor熵 | 梯度范数 | 回报尺度 | 3h已有数据累计采样比例 |','| --- | --- | --- | --- | --- |']
        for arm in diagnostic['arms']:
            metrics=arm['last_metrics'];replay=arm['replay_reports'][-1]
            lines.append(f"| {arm['config']} | {metrics.get('ent/action')} | {metrics.get('opt/grad_norm')} | {arm.get('last_return_scale')} | {replay.get('replay_old_cumulative_fraction')} |")
        lines += ['']
        if diagnostic.get('skipped'):
            lines += ['因一小时诊断上限未执行：'+json.dumps(diagnostic['skipped'],ensure_ascii=False)+'。','']
        priority=[r['online_priority_sample_fraction'] for arm in diagnostic['arms'] for r in arm['replay_reports'] if r['online_priority_sample_fraction'] is not None]
        if priority and max(priority)==0:
            lines += ['短对照中实测在线优先采样比例均为 0；恢复后的短采集窗口尚未产生可用在线序列，因此 online 开关对照不能用于排除更长训练中的在线采样偏置。','']
        lines += ['本轮可确认的是继续优化后的闭环支撑稳定性退化。数值爆炸无证据；最早 s0 数据耗尽是历史事实，但不是由时间相关性就能确定的根因。回报尺度只有有限快照证据，奖励比例未做单变量干预，因此这两项仍是未排除因素，不为写出“唯一根因”而扩大结论。','']
    if Path('metrics/lwy_r7_failure_analysis.json').exists():
        full=read('metrics/lwy_r7_failure_analysis.json')['rows']
        lines += ['### 失败前 25 秒的完整动作检查','',
            '| 数据集 | 类型 | 数量 | 横向偏移绝对值中位m | 末目标横向绝对值中位m | 请求/实际速度中位m/s | 25秒进展中位m | 转向标准差中位° |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for dataset in ['original','blind']:
            for failure in sorted({r['failure'] for r in full}):
                rows=[r for r in full if r['dataset']==dataset and r['failure']==failure]
                if not rows:continue
                med=lambda key:np.median([r[key] for r in rows])
                lines.append(f"| {dataset} | {failure} | {len(rows)} | {np.median([abs(r['lateral_m']) for r in rows]):.3f} | {np.median([abs(r['terminal_target_body_m'][1]) for r in rows]):.3f} | {med('requested_speed_mps'):.3f}/{med('realized_speed_mps'):.3f} | {med('history_progress_m'):.3f} | {med('steer_std_degrees'):.1f} |")
        lines += ['','逐回合的区段、坐标、目标点、速度、树根/陡坡诊断、转向换符号比例与奖励分量见 `metrics/lwy_r7_failure_analysis.json`。地形标记是末态局部诊断；随机策略转向变化不能单独证明确定性极限环。不改变台阶和树根限速，避免用提高速度掩盖卡住问题。','']
    lines += ['## 改动、训练与选模','',
        '仅随机摆放规则和选中的优化/采样参数改变。Dreamer 与 PPO 仍共用原奖励；观测与动作规格不改，训练不开过滤，75% LWY / 25% 旧图的按回合混合保持不变。原始梯度超过 10¹² 或非有限立即停止。每 30 min 独立快照，验证 80 回合；连续两次低于最佳超过 30 个百分点或连续两次下降均停止，不自动反复重启。选模依次比较验证全通过率、平均门数、较早快照，测试集不参与。','']
    if state.get('memory'):lines += ['显存实测：`'+json.dumps(state['memory'])+'`。他人的 GPU 进程未更改。','']
    if state.get('rounds'):
        lines += ['| 新训练时长h | 验证通过率 | 平均门数 |','| --- | --- | --- |']
        for r in state['rounds']:lines.append(f"| {r['hours']:.2f} | {r['success_rate']:.1%} | {r['mean_gates']:.3f} |")
        lines += ['','![验证曲线](figures/lwy_r7_training_curve.png)','',f"早停原因：{state.get('early_stop','无')}。最终模型：`{state.get('best',{}).get('checkpoint','待选')}`。",'']
    if state.get('ppo_rounds'):
        lines += ['PPO 单次检查，学习率 3e-4→3e-5，熵系数保持原值；仅此一次预算，不因零成功追加训练。','',
            '| 检查时长h | 验证通过率 | 平均门数 |','| --- | --- | --- |']
        for r in state['ppo_rounds']:lines.append(f"| {r['hours']:.2f} | {r['success_rate']:.1%} | {r['mean_gates']:.3f} |")
        lines += ['']
    if state.get('ppo_memory'):
        lines += ['PPO 128 环境、全部训练图、完整 rollout 和优化器实测：`'+json.dumps(state['ppo_memory'])+'`。BEV rollout 张量和优化器/XLA 工作区决定了大量显存开销，不能按环境数与 Dreamer 作线性比较；保持原 ppo_s0 的 128 环境规模。该探测计入 PPO 一小时上限。','']
    files=sorted(Path('metrics').glob('lwy_r7_final_*.json'))
    lines += ['## 最终比较','',f'已完成 {len(files)}/64 个正式条件。原版/盲测/旧留出各100回合，mini30回合。所有方法使用 autotune=1、deterministic_ops、独立回合 PRNG、单环境 vmap、固定100策略流。','',
        '| 方法 | 数据集 | ob/sf | 全通过率 [Wilson95%] | 平均门 | 失败分类 | 接管率 | 速度m/s | IoU |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in files:
        d=read(p);ci=d['success_wilson95'];lines.append(f"| {d['method']} | {d['dataset']} | {int(d['random_obstacles'])}/{int(d['safety_filter'])} | {d['success_rate']:.1%} [{ci[0]:.1%},{ci[1]:.1%}] | {d['mean_gates']:.3f} | {json.dumps(d['failures'])} | {d['takeover_rate']:.2%} | {d['mean_speed_mps']:.3f} | {d['trav_iou']} |")
    lines += ['','## 可选支撑余量过滤','',
        '仅评估：从公共 BEV 支撑/路线通道和速度状态预测 0.8 s 足迹，在物理包络外采样检查 8 cm 余量，不读取特权地图。受 0.125 m BEV 栅格和离散足迹采样限制，不是连续几何安全保证。触发时由原规划器接管；默认 margin=0，原过滤器不变。','',
        '| 数据集 | 障碍 | 全通过率 | 接管率 |','| --- | --- | --- | --- |']
    for p in sorted(Path('metrics').glob('lwy_r7_support_*.json')):
        d=read(p);lines.append(f"| {d['dataset']} | {int(d['random_obstacles'])} | {d['success_rate']:.1%} | {d['takeover_rate']:.2%} |")
    lines += ['','## 验证、视频与未解决事项','',
        '视频：`videos/lwy_r7_dreamer.npz/.json` 为远端 CPU 实际推理，`videos/lwy_r7_dreamer.mp4` 在 Mac 渲染。CPU 与 GPU 不承诺逐位一致，视频不能替代正式 GPU 评估；MuJoCo 仅回放快环境位姿，不代表轮足底层验证。','']
    for name in ['check_diag','check_lwy','unit_checks']:
        p=Path(f'metrics/lwy_r7_{name}.json');lines += [f'`{p}`：'+('通过' if p.exists() and read(p).get('passed') else '待完成')+'。']
    lines += ['','执行路径变更：Mac 的整个 ATEC_NAV 目录在任务期间移至 `/Users/amiao/ATEC_NAV/`，后续继续使用其中同一份 nav3；没有在 dreamerv3 中重建或修改文件。','']
    log=Path('logs/lwy_r7_checks.log')
    if log.exists() and "No module named 'jax'" in log.read_text():
        lines += ["新增回归测试首次使用几何专用 venv，原始错误：`ModuleNotFoundError: No module named 'jax'`。随后改用已有 `$NAV3_PYTHON` 并设置 `JAX_PLATFORMS=cpu`；未向共享环境安装包。原失败日志保留。",'']
    lines += ['','GPU 进程墙钟记账：'+f"{(state.get('gate_gpu_seconds_upper_bound',0)+sum(r.get('seconds',0) for r in state.get('jobs',[]) if r.get('gpu')))/3600:.3f} h。包括显存探测、诊断、训练、验证、最终评估及初始障碍闸门；闸门按 CPU 目录生成完成至最后闸门指标写出的时间差保守计入（{state.get('gate_gpu_seconds_upper_bound',0):.1f} s）。CPU 克隆/几何分析/测试/录像和 Mac 渲染不计 GPU；进程初始化与保存均计入，不以纯 kernel 时间缩减预算。",'',
        '限制：仅一个训练种子；地图为视频近似，未实测；没有修改真实碰撞/支撑阈值；离散通路并非连续动力学保证；短对照不一定足以识别所有相互作用。未完成步骤仍明确标为待完成。','',
        '代码清单：`lwy_obstacles_r7.py` 摆放与证书；`lwy_env_r7.py` 默认/旧规则切换；`lwy_mixed_r7.py` 新训练入口；`lwy_compare_r7.py` 同协议评估；`lwy_support_filter_r7.py` 可选支撑检查；`lwy_train_r7.py` 官方循环续训；`lwy_train_ppo_r7.py` / `lwy_ppo_update_r7.py` PPO检查；`lwy_r7_queue.py` 闸门/预算/早停；`lwy_r7_analysis.py` / `lwy_r7_obstacle_analysis.py` 只读证据；`lwy_r7_clone.py` 精确克隆；`lwy_r7_report.py` 从指标生成本节；`lwy_video_record_r7.py` CPU固定100策略流录像。','']
    report=Path('LWY_REPORT.md');old=report.read_text().split(MARKER)[0];report.write_text(old+MARKER+'\n'.join(lines))
    if state.get('rounds'):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8));r=state['rounds'];ax=axes[0,0]
        ax.plot([x['hours'] for x in r],[x['success_rate'] for x in r],'-o')
        ax.set(xlabel='Scheduled continuation hours',ylabel='Validation success (80 episodes)',ylim=(0,1))
        path=Path('metrics/train_dreamer_lwy_r7.jsonl')
        events=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        events=[x for x in events if x.get('event')=='train']
        def curve(ax,key,label,metric=True):
            points=[(x['wall_seconds']/3600,(x.get('metrics',{}) if metric else x).get(key)) for x in events]
            points=[(x,y) for x,y in points if y is not None]
            if points:ax.plot(*zip(*points),label=label)
        ax=axes[0,1]
        for key in ['bev','priv_trav','dyn','rew','value']:curve(ax,'train/loss/'+key,key)
        ax.set(xlabel='Logged continuation wall hours',ylabel='Training loss',yscale='log')
        if ax.lines:ax.legend(fontsize=8)
        ax=axes[1,0];curve(ax,'train/ent/action','Actor entropy');ax.set(xlabel='Logged continuation wall hours',ylabel='Actor entropy')
        other=ax.twinx();curve(other,'train/opt/grad_norm','Gradient norm');other.set_ylabel('Raw gradient norm')
        for line in other.lines:line.set_color('tab:orange')
        ax=axes[1,1]
        curve(ax,'replay_old_sample_fraction','Data present at 3 h',False)
        curve(ax,'online_priority_sample_fraction','Online priority',False)
        ax.set(xlabel='Logged continuation wall hours',ylabel='Replay sampling fraction',ylim=(0,1))
        if ax.lines:ax.legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig('figures/lwy_r7_training_curve.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
