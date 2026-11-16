"""Append current round evidence while preserving the previous LWY report."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from .lwy_comparison_runtime import flags as evaluation_flags,suffix as evaluation_suffix,level as evaluation_level

MARKER='\n<a id="lwy-v2-update"></a>\n'

def read(path,default=None):
    p=Path(path);return json.loads(p.read_text()) if p.exists() else default

def link(path):return f'[{Path(path).name}]({path})'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    report=Path('LWY_REPORT.md');old=report.read_text().split(MARKER)[0]
    # Remove only the status preface created by this continuation; original body survives.
    if old.startswith('> **第六轮续做'):old=old[old.index('\n# nav3 第六轮'):].lstrip('\n')
    gate=read('metrics/lwy_v2_g1.json',{});manifest=read('metrics/lwy_v2_split_manifest.json',{})
    queue=read('metrics/lwy_v2_queue.json',{});selection=read('metrics/lwy_v2_model_selection.json',{})
    completed=queue.get('status') in ['complete','evaluations_complete'];status='训练及最终评估完成' if completed else ('训练队列运行中' if queue.get('status')=='running' and queue.get('training') else '训练前检查进行中')
    if queue.get('status')=='failed':status='队列已停止：'+queue.get('error','未注明错误')
    if queue.get('status') in ['interrupted','diagnosis']:status='训练已暂停，正在诊断数值异常'
    numerics=read('metrics/lwy_v2_training_numerics_diagnosis.json')
    decision=read('metrics/lwy_v2_evaluation_runtime_decision.json')
    candidate=read('metrics/lwy_v2_runtime_candidate_protocol.json')
    ready=(set(queue.get('training',{}))=={'dreamer_lwy','ppo_lwy'}
        and all(x.get('smoke_collectors_complete') for x in queue['training'].values()) and candidate and candidate['passed'])
    if numerics and not decision:
        status=('训练前检查及短冒烟已完成；等待评估协议决定，正式训练未开始' if ready else '正式训练未开始；评估数值异常待协议决定，训练运行时正在复核')
    preface=f'> **第六轮续做更新**：G1 {"通过（原版 10/10，盲测 20/20）" if gate.get("passed") else "尚未通过"}；{status}。本轮证据见[续做结果](#lwy-v2-update)。下方原报告保留为上一轮记录。\n\n'
    text=['# LWY v2 续做结果',f'更新时间：{datetime.now(timezone.utc).isoformat()}。',
        f'**G1：{"通过" if gate.get("passed") else "未通过"}；{status}。** 本轮没有修改原版 1 km 几何、碰撞判定、1.03×0.58 m 包络或旧地图。',
        '## 验收决定与实际改动',
        '原来的第八门落在路面尽头，机器人中心越过门就一定出界。这是任务设定的问题，不是规划器的问题。完整路线前七门保持 125、250、375、500、625、750、875 m，第八门改为 998.5 m；通过即正常结束。`--terminal-offset` 默认 1.5 m，允许更大的后退量。训练裁片为 250 m，起点取 0/249/498/747 m；最末裁片结束于源路线 997 m，其后仍有 3 m 真路面，所以其末门保留裁片 s=250。没有源路面延续时，末门后退至少 1.5 m。',
        'v2 对生成完成的整个世界几何施加 diag(1,−1,1)，不是只镜像控制点。所有 road/bank 碰撞棱柱、坡面网格、护栏、台阶侧墙、石块、树根、树干、标牌一起反射，三角面绕序同时翻转；植物实例同时反射位置、偏航与局部 Y 手性，以支持非对称素材。原 `maps_lwy/` 保持不动。',
        table(['改动','默认 / v2 采用','独立证据'],[
            ['末门位置','旧导入器默认行为不变；v2 L−1.5 m','只改门，原版 gain=2 已 10/10；gain=4 同为 10/10'],
            ['跟踪器航向增益','旧行为 2；v2 配置 4','旧图2→998 m完整历史诊断 9/20→12/20，改善4图、退化1图；原版无退化'],
            ['规划器预测时域','默认70×0.1 s；v2配置30×0.1 s','同一失败现场 0/2→2/2；新盲测完整18/20→20/20'],
            ['候选数量、前视、速度预测、边界权重','15、1.2 m、0.5 m/s、300 保持','未改善的单变量设置不采用'],
            ['几何筛选','0.2 m，最小原始净空0.10 m，可配置','原始多边形检查；阻挡石块固定规则重采样，仍不合格则丢弃种子'],
            ['判定、限速、观测、奖励','不改','旧环境指纹、默认跟踪哈希、包络回归和保护清单核对']]),
        '## s≈980 m 专项诊断',
        '先用旧完整路线与完整观测历史复现：11 个内部失败的动作/位姿哈希与上一轮完全一致；四个 50 ms 物理子步的回放终点误差为 0。然后对题述8图核查原始几何与首次失败足迹。'+link('metrics/lwy_v2_diagnosis_summary.json')+'。']
    geo=read('metrics/lwy_v2_geometry_diagnosis.json',{}).get('rows',[])
    text += [table(['种子','失败 s(m)','几何','路宽(m)','区段边界距离(m)','有符号横移(cm)'],[
        [r['seed'],f"{r['s']:.3f}",'/'.join(f.get('name','final_descent') for f in r['flights']) or '台阶邻域',
         f"{r['width_range_m'][1]:.3f}",f"{r['zone_transition_distance_m']:.2f}",f"{100*r['actual_lateral_m']:+.2f}"] for r in geo]),
        '这些位置是最后一段下行台阶的入口、出口或近邻弯折，邻域±2 m路宽不变；不是区段过渡或宽度突变。按路线法向 (−t_y,t_x)，正横移是左偏：8图中7图左偏，只有630014右偏，因此不能把原先的“全部右偏”当作已确认事实。台阶前后方向变化、跟踪延迟与保守边界共同压缩余量。',
        table(['假设','实验 / 证据','结论'],[
            ['台阶限速下跟踪误差','航向增益2→4；完整旧图9→12/20；记录cmd/vel/航向误差','有贡献，单独不足以解锁G1；0.28 m/s限速未改'],
            ['定位漂移','完整旧图关闭漂移9→10/20，含退化图','漂移不是充分解释；正式设置保留原漂移'],
            ['候选太稀','短段15→57个候选均12/20','没有证据支持单纯加密；未采用'],
            ['预测时域过长','保留完全相同PRNG/记忆/延迟队列的现场；70→30步0/2→2/2','缩短时域减少远端预测误差；完整G1确认有效'],
            ['支撑栅格在台阶处出错','首次坏子步对原始多边形复核，多个失败仍有约5–8 cm真净空','达到既有保守栅格界限；未发现应通过放宽判定修复的导入错误'],
            ['侧墙碰撞层','630014首次碰撞足迹距原墙约4.70 cm；投影与原始侧墙一致','保守层确实触发，但不能删除侧墙或降低包络；不合格变体由几何筛选丢弃']]),
        '旧图冷启动970 m短段结果：基线12/20；候选57为12；前视0.6 m为11；预测速度0.28 m/s为12；无漂移11；扩大规划预测余量11；航向增益4为13、6为12。短段分数不替代完整G1。'+link('figures/lwy_v2_parameter_regressions.png')+'。',
        '首轮新G1剩余两处为630005的703.667 m与630021的985.179 m，均为支撑失败。现场恢复基线0/2，单独恢复增益2为1/2，提高边界权重300→1000为0/2，仅时域70→30为2/2；两处首次坏足迹的原始支撑净空分别6.93、5.93 cm。'+link('metrics/lwy_v2_round0_replay_baseline_analysis.json')+'、'+link('metrics/lwy_v2_round0_replay_horizon30_analysis.json')+'。',
        '## 导入、净空与数据划分',
        '原版的0.05 m世界坐标分块栅格、43×25足迹、5.277 cm保守距离与几何误差量化仍见上轮记录；原始内部中心线最小净空20.73 cm，3000探针不安全误放行0、保守误拒293。v2原版地图除门数组外逐数组与旧原版一致。BEV保持64×64×6、0.125 m/格，状态13维、动作3维，旧权重可续训；此次没有足够证据要求修改观测规格。',
        '变体可行性审计检查可执行中心域 s=1.5…998.5 m，步长0.2 m并包含末点，使用完整1.03×0.58 m矩形。两端1.5 m端帽明确排除：有限包络不可能以中心位于真实端点，且出生在2 m、终门在998.5 m。该约定写入每张图的元数据；不能把端帽排除隐称为全闭区间通过。']
    if manifest:
        records=manifest['maps'];clear=[r['clearance_check']['minimum_m'] for r in records if r['split']!='test']
        text += [f"版本 `{manifest['dataset_version']}`；训练200、验证20、盲测20、原版测试1。丢弃候选 **{manifest['discarded_variants']}** 个，累计重采样石块 **{manifest['resampled_objects']}** 次；接受变体最小净空 **{min(clear)*100:.3f} cm**，中位数 **{np.median(clear)*100:.3f} cm**。筛选只读取原始几何，未用规划器成绩筛种子。完整接受、拒绝和重采样清单："+link('metrics/lwy_v2_split_manifest.json')+'。',
            '种子从610000/620000/630000分别连续递增，接受各自首批满足净空的200/20/20图；不跨各自10000种子带。原版862428、mini-trail871002、合成30000–30003及旧留出集均不进入训练。旧训练种子10000–10199。',
            table(['划分','接受种子'],[[s,', '.join(str(r['seed']) for r in records if r['split']==s)] for s in ['train','val','blind','test']])]
        rejects=[r for r in manifest['rejected_candidates'] if r['seed'] in [630006,630007]]
        text += [table(['拒绝种子','最小净空(m)','不合格采样数','阻挡项示例'],[[r['seed'],r['clearance']['minimum_m'],len(r['clearance']['bad_stations_m']),', '.join(r['clearance']['blocking_names'][:5])] for r in rejects]),
            '检查中−1 m是“矩形未完全被支撑域覆盖”的哨兵值，不代表实际穿出1 m。固定侧墙、护栏及支撑不足不通过移动道路修复；只允许固定规则重采样阻挡石块。草稿目录 `maps_lwy_v2_stride_draft/` 与中断候选不在最终清单中；最终集合按连续种子规则生成。']
    else:text.append('最终清单仍在生成。已完成的各图位于远端 `maps_lwy_v2/`，尚不据此宣称200图齐全。')
    text += ['## G1 重跑',table(['轮次','原版','盲测','结论'],[
        [i,f"{r['original_successes']}/10",f"{int(round(r['blind_success_rate']*20))}/20",'通过' if r['passed'] else '未通过']
        for i in ['v2_round0','v2_round1','v2_round2'] if (r:=read(f'metrics/lwy_{i}_g1.json'))]),
        '首轮18/20后只使用了一轮额外完整修复迭代，达到原版10/10与盲测≥95%的双重条件。没有降低阈值。G1先在指向最终20张盲测与原版的只读资格视图执行，训练入口还要求最终训练/验证清单齐全并匹配G1哈希。'+link('metrics/lwy_v2_g1.json')+'。',
        '## 评估协议与训练前检查',
        f'当前正式比较配置为XLA autotune={evaluation_level()}、deterministic_ops=true，TF_CUDNN_DETERMINISTIC=1；每回合独立噪声PRNG，按单环境vmap执行。策略另用噪声种子+1,000,000,000。最初发现卷积批量形状不同仍会导致数值差异，因此神经策略统一以100个独立流执行，不足100的部分仅补齐策略计算，不进入物理环境、统计或PRNG；旧的未固定批量探针保留为诊断资料，不混入最终对比。原level0批量一致性证据：'+link('metrics/lwy_v2_policy_protocol.json')+'。',
        '按每次回合重置独立抽取来源：LWY 75%、旧maps/train 25%，随后在对应训练集合中均匀抽图；并非把75%常驻worker永久绑到LWY。不同来源保留各自原物理判定。训练日志记录实际回合数与每图暴露次数。',
        '随机障碍沿用原尺寸、侧向和站点提议分布，组合静态障碍后用0.2 m路线/0.04 m横移格点寻找通路，同时检查连接边的中点，再以0.1 m间距、完整原足迹和原碰撞距离复核通路。不通过即重采样。每图保留8组通过检查的三障碍布局，运行时随机选择；所有方法使用同一目录与协议。这是离散几何通路检查，不能证明动态跟踪器必然通过；随机场景多样性限于该目录。',
        '训练前修复了两项实现问题：混合采样器在vmap内选择整个地图对象会广播大型图集，64环境曾请求95 GB设备内存；改为共享图集、只选择状态与观测后，7775次回合开始中LWY为5858、旧图1917（75.344%）。通路搜索原先只检查节点、未在搜索中约束连接中点，且额外原始几何筛选会拒绝原物理判定允许的绕行；现统一使用原物理判定并检查节点和中点，额外1 cm仅用于搜索见证。legacy/train_10001单例8组/10次提议、lwy/train_610043单例8组/277次提议通过；最终目录全部按v3检查器生成。这些改动没有放宽运行时碰撞判定。',
        '固定100策略流还通过了1回合与最终100回合首轨迹逐位比较。'+link('metrics/lwy_v2_policy_protocol_final_batch.json')+'。物理步分组加速实验虽在100步状态、观测和额外信息上逐位一致，但仅为直接vmap速度的0.247倍，未采用；正式评估仍执行原单环境vmap。']
    obs=read('metrics/lwy_v2_random_feasibility.json')
    text += [f"200回合分层抽查（150 LWY+50旧图，每张图独立提议种子）：首提议未找到合格通路 **{obs['initial_unpassable']}/200={obs['initial_unpassable_rate']:.1%}**，重采样后 **0/200**；整个目录拒绝组合{obs['total_catalog_rejections']}次。该比例是保守离散搜索拒绝率，不等同于几何上绝对不可通过率。"+link('metrics/lwy_v2_random_feasibility.json')+'。' if obs else '200回合抽查尚未完成；不会据此提前启动训练。']
    mem=read('metrics/lwy_v2_memory_selection.json')
    if mem:
        memory=[read(p) for p in mem['dreamer_profiles']+[mem['ppo_profile']]]
        text += [table(['方法','环境数','分配器 / 池上限','实测峰值MiB','占总显存','Agent+优化器完成'],[[r['method'],r['envs'],r.get('jax_allocator','default')+' / '+('不适用' if r.get('jax_allocator')=='platform' else str(r.get('jax_pool_fraction',.95))),r['peak_used_mib'],f"{r['peak_fraction']:.1%}",r['completed']] for r in memory]),
            f"实际选用环境数：{mem['selected_envs']}。测量包括全部200张LWY裁片、200张旧图、真实Agent、采集、回放批次及优化器；没有用纯环境占用估算。"+link('metrics/lwy_v2_memory_selection.json')+'。']
        text.append('默认分配器下4608实测OOM，4096/3584/3072虽运行成功，保留内存池后仍占95.9%；2560–2944约72.3%。池上限0.86试验无法满足3072的连续分配请求，原始失败均保留。随后使用JAX已支持的platform分配器释放不再使用的分配，重新测量Dreamer；PPO与评估保留默认分配器。选中的训练配置与实测一致。表中峰值是整张24,564 MiB卡的实际占用，包含约911 MiB桌面基线；冒烟继续检查实际峰值。若原定4608已可完整运行且低于85%，保留原环境数，不为填满显存额外扩大训练配置。')
    else:text.append('完整地图+Agent显存实测尚未完成；不把旧报告的纯环境显存当作本轮实测。')
    text += ['## 训练、快照与模型选择',
        '观测动作规格不变，因此从原始健康 `dreamer_s0`、`ppo_s0` 复制到新名字，保留优化器；没有从已崩溃的ft快照启动。Dreamer使用未改的 `embodied.run.train.train` 主循环，接入GPU混合集合Driver。保留原replay的100,000个序列起点及其依赖chunk，旧数据只读复制到新replay；记录实际抽到的新旧样本比例。PPO保留原配置。两者奖励一致，均不开安全过滤。',
        '修复运行时后各做6分钟冒烟和5分钟恢复检查，实际作业合计须≤15分钟；此前失败初始化与异常数值尝试单独保留为诊断，不计作通过的冒烟，全部GPU时间仍计入12小时总预算。随后各计划8段30分钟正式训练。每段保存新快照并做验证20图×2回合，不开随机障碍/过滤。选择规则为验证全通过率最高，其次平均门数，完全相同取更早快照；原版、盲测、旧留出和mini不参与选择。连续两次低于此前最佳超过30个百分点即停止该方法，保留快照并生成只读崩溃诊断；不重启刷分。',
        'Dreamer首次冒烟在0步/0次更新时因Chunk.saved标记不一致而失败。失败初始化被隔离到checkpoints/replay下的lwy_v2_failed_initialization，原始日志保留；修复为将旧chunk文件复制到新replay目录，同时保留两个保存标记。未更改上游replay源码。另先采集66步以形成新的65步序列，随后启用官方Ratio=32，避免最初长时间只抽到旧replay；保留旧样本并记录实际比例。初始化异常不再写出可能不完整的最终检查点。'+link('metrics/lwy_v2_replay_initialization_failure.json')+'。',
        '检查点保存混合集合状态、策略状态、PRNG、replay和官方Ratio进度，段间恢复环境而非同步重置。训练采样预取队列遵循官方循环行为，未宣称跨进程训练位级重现。',
        '恢复冒烟另发现NumPy直接保存bfloat16时将dtype记成|V2，导致JAX加载失败；失败发生在加载阶段，未执行新更新。LWY专用序列化改为保存uint16原始位并记录bfloat16叶子索引，兼容读取已保存的两片旧carry。实际721次更新检查点的43个叶子全部经CPU验证：原文件位模式完全一致，新格式往返后的dtype、形状及所有位均一致。未改旧通用检查点工具或原始权重。'+link('metrics/lwy_v2_checkpoint_serialization.json')+'。',
        '另遵循本次续做的“验证成绩连续下降就诊断”要求：全通过率相邻两次都下降时，即使尚未触及上述30个百分点规则，也停止该条训练并做3.5诊断，不自动重启。原TASK的强制早停条件同时保留。',
        '队列：`nav3/lwy_queue_v2.py`，状态：'+link('metrics/lwy_v2_queue.json')+'；每个子作业带超时且短于3小时，预算累计并记录。出现代码异常、OOM、非有限损失或预算耗尽会明确失败并停止，不能把失败算完成。']
    if queue:
        charged=queue.get('gpu_job_wall_seconds',0)+queue.get('preceding_g1_and_diagnostics_allowance_seconds',0)
        text.append(f"累计GPU占用记账（含前置与外部诊断预留）约{charged/3600:.3f}小时，12小时预算尚余约{(queue.get('gpu_budget_seconds',43200)-charged)/3600:.3f}小时；完整正式训练及最终评估是否完成以队列状态为准。")
    parallel=read(f'metrics/lwy_v2_parallel_protocol{evaluation_suffix()}.json')
    if parallel:
        text.append('最终评估最多并行两个独立进程，训练与显存测量保持单作业。两种旧模型分别在完整20图盲测上运行两个100回合的并行副本；400个完整轨迹哈希均与串行基线相同，同时检验两份Dreamer的显存。评估的每回合PRNG、单环境vmap与固定100策略流不变；GPU预算按这张卡被占用的墙钟并集累计，另保留每个进程耗时。'+link(f'metrics/lwy_v2_parallel_protocol{evaluation_suffix()}.json')+'。')
    curve=read('metrics/lwy_v2_learning_curve.json',{})
    if curve:
        text += [table(['方法','快照(h)','验证通过率','平均门数','停止'],[[name,r['label'],f"{r['success_rate']:.1%}",f"{r['mean_gates']:.3f}",x['stopped']] for name,x in curve.items() for r in x['rounds']]),link('figures/lwy_v2_learning_curve.png')]
    else:text.append('尚无正式训练曲线或已选模型；不会填入预期数字。')
    if selection:text.append('选择结果：'+link('metrics/lwy_v2_model_selection.json')+'。')
    current_training=queue.get('training',{})
    if current_training:
        text.append(table(['方法','纯训练冒烟完成','完整冒烟含验证','恢复前/后更新数','冒烟实际秒数'],[
            [name,run.get('smoke_collectors_complete',False),run.get('smoke_complete',False),
             str(run.get('resume_verified',{}).get('before',{}).get('updates','待完成'))+' / '+str(run.get('resume_verified',{}).get('after',{}).get('updates','待完成')),
             round(run.get('smoke_job_wall_seconds',0),1)] for name,run in current_training.items()]))
    diagnoses=sorted(Path('metrics').glob('lwy_v2_collapse_*.json'))
    if numerics:
        text += ['## 训练运行时数值异常与单变量诊断',
            '异常冒烟累计313,344步、289次更新，随后在验证阶段人工停止队列，未进入正式训练。健康原始权重与同一冻结旧replay批次，在autotune=0时首个损失5,585.917、原始梯度范数2.023×10¹⁵；唯一改动为恢复原训练autotune=1后，损失236.285、梯度830.305。7个返回指标的更新末尾损失229.797、梯度764.318。冻结新批次也由损失5,640.120/梯度5.127×10¹⁵恢复到2,318.631/9,786.664，并继续下降。'+link('metrics/lwy_v2_training_numerics_diagnosis.json')+'、'+link('figures/lwy_v2_training_numerics.png')+'。',
            '这是把评估设置错误带入训练导致的运行时问题。采用Dreamer运行时声明的默认autotune=1；PPO训练恢复原来的库默认XLA设置。所有原始学习率、优化器、奖励及地图判定不变。异常新权重与replay隔离于`checkpoints/lwy_v2_autotune0_diagnostic/dreamer_lwy/`和`replay/lwy_v2_autotune0_diagnostic/dreamer_lwy/`，异常0.10h快照保留且不参与选模型。修复后的运行另用`_runtime2`快照及作业标识，从原始健康权重开始。显存也按修复后的训练运行时重新实测，旧测量保留。',
            '历史兼容性证据其实已存在于'+link('metrics/fix_numeric_dreamer_level0.json')+'和'+link('metrics/fix_numeric_dreamer_level1.json')+'，第5轮正式`nav3/fix_eval.py`也已在JAX初始化前使用level1。上一轮LWY的level0只跑了规则规划器。本轮错误地将它推广到神经训练/推理，现用冻结批次再次复现并记录；不能把该问题归咎于地图或新的训练分布。',
            '逐项排查：旧replay本身即可复现异常，因此新障碍惩罚不是该数值问题的原因；确实保留100,000个旧序列起点，并非全新replay。在线队列只优先最近完成的65步序列，冒烟结束时队列为0，不能据初始旧样本比例0就断言旧数据永久不参与。冻结旧批次在上下文后没有重置标记，仍出现异常，排除同步初始重置作为这一数值问题的解释。首个更新前损失和原始梯度已经不同，学习率不可能解释该差异；没有据此调小学习率。上述结论不排除后续导航行为仍受奖励和分布变化影响。',
            '推理复核同样发现问题：固定100流、相同健康模型与观测，autotune=0的可通行预测整幅恒为0.48828125；autotune=1输出范围约0…1，平均绝对差0.48544，动作最大绝对差2.65569。故现有旧模型零样本0/100不能直接解释为泛化失败。'+link('metrics/lwy_v2_forward_runtime_comparison.json')+'。'+('评估协议决定：'+link('metrics/lwy_v2_evaluation_runtime_decision.json') if decision else '用户明确要求沿用旧设置，已提出协议修正问题；在答复前暂停依赖该决定的训练验证及最终比较。'),
            '队列对外部数值诊断预留并计入900秒GPU占用，低于3.5节1小时上限。新训练增加原始梯度>10¹²立即停止保护，防止再次把有限但失真的数值当作正常损失。']
    text += ['## 崩溃诊断状态',('\n'.join(link(str(p)) for p in diagnoses) if diagnoses else '尚未触发本轮崩溃诊断。旧ft崩溃快照与原有诊断材料保持只读。触发后报告损失、熵/方差、回报尺度、价值损失、梯度范数、replay旧样本比例、重置与失败动作/位置；没有干预证据时不声称唯一根因。'),
        '## 最终比较与视频']
    candidate=read('metrics/lwy_v2_runtime_candidate_protocol.json')
    if candidate:
        text.append('level1候选协议试验已完成，是否正式采用以用户决定及配置为准。两种旧模型各3个独立进程、每个100回合，合计600个完整轨迹哈希逐位一致；单回合与100回合的首轨迹也一致。规划器资格复核原版10/10、盲测20/20，全部30条轨迹与level0逐位相同，未调整规划器或几何。'+link('metrics/lwy_v2_runtime_candidate_protocol.json')+f"。候选协议试验基于数值错误诊断，不以测试成绩调参；加上外部诊断预留，数值诊断GPU占用上界为{candidate['diagnostic_gpu_upper_bound_seconds']:.1f}秒，低于3.5节1小时上限。")
        if not decision:
            text.append('当前唯一待定项：本次第1.5条要求沿用上一轮确定性设置，而旧LWY报告写为level0；该设置使Dreamer神经内核数值异常。已向用户提出是否恢复level1的决定，尚未收到答复，因此未启动依赖可信验证的各4小时正式训练，也未将候选结果当作正式最终比较。允许后只需记录决定、激活配置并运行独立队列；不需要重新生成数据或重新初始化已通过冒烟的模型。')
    evaluations=[(p,r) for p in sorted(Path('metrics').glob('lwy_v2_final_*.json')) if (r:=read(p))['runtime']['xla_flags']==evaluation_flags()]
    text.append(f'已产生 **{len(evaluations)}/80** 个最终条件结果。完整要求为5方法×4数据集×随机障碍/过滤4条件；原版、盲测、旧留出各100回合，mini30回合。未完成项不以旧协议结果代替。')
    if evaluations:
        if numerics and evaluation_level()==0:
            text.append('注意：下表神经模型结果使用已确认存在数值异常的level0推理，不可据此单独判断模型泛化能力；决定前仅保留为诊断记录。规划器不使用该神经内核；其100/100结果与G1首10回合轨迹逐位一致。')
        text += [table(['方法','数据集','障碍/过滤','N','成功率 [Wilson95%]','平均门数','接管率','速度m/s','IoU 全/遮挡/障碍'],[
            [r['method'],r['dataset'],f"{int(r['random_obstacles'])}/{int(r['safety_filter'])}",r['episodes'],
             f"{r['success_rate']:.1%} [{r['success_wilson95'][0]:.1%},{r['success_wilson95'][1]:.1%}]",f"{r['mean_gates']:.3f}",f"{r['takeover_rate']:.2%}",f"{r['mean_speed_mps']:.3f}",
             '不适用' if r['trav_iou'] is None else '/'.join('无定义' if x is None else f'{x:.3f}' for x in r['trav_iou'])] for p,r in evaluations]),
            '失败分类、每回合s位置/对应原始区段、接管数、动作、轨迹哈希、模型哈希与奖励分量在各 `metrics/lwy_v2_final_*.json`。IoU每20策略步采样；无预测头或空并集不报伪造的0。']
    video=read(f'metrics/lwy_v2_dreamer_vs_ppo{evaluation_suffix()}.json')
    text.append('同噪声940000、同策略种子1000940000，无过滤对比视频：'+link(video['path'])+'。使用原版包XML，回放快环境位姿并显示包络；不是MuJoCo腿部动力学验证。' if video else '最终Dreamer/PPO对比视频尚未生成；`lwy_v2_renderer_probe.mp4` 若存在，仅是原版XML渲染检查，不冒充模型推理视频。')
    text += ['## 检查、产物与未解决事项']
    for p in ['metrics/lwy_v2_check_diag_final.json','metrics/lwy_v2_checks_final.json','metrics/lwy_v2_check_diag.json','metrics/lwy_v2_checks_legacy.json','metrics/lwy_v2_checks.json','metrics/lwy_v2_sampler_probe.json','metrics/lwy_v2_mac_integrity_check.json','metrics/lwy_v2_cleanup.json']:
        d=read(p)
        text.append(link(p)+('：'+json.dumps({k:v for k,v in d.items() if k in ['passed','protected_files','protected_changed','nav3_sessions','owned_background_processes','queue_pid']},ensure_ascii=False) if d else '：尚未完成。'))
    text += ['地图、checkpoint、replay及大型栅格留在远端 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`；Mac同步报告、代码、JSON、图、日志和小视频。源包两处校验均为137文件/642,046,794字节，旧物理指纹保持 `a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d`。',
        '较早的`lwy_v2_checks.json`顶层`g1_passed=false`引用上一轮旧G1；其`v2.passed=true`为新版几何回归通过。最终`lwy_v2_checks_final.json`已将上一轮状态单列为`legacy_g1_passed`，顶层G1对应本轮，并核对当前环境哈希。G1资格原始证据仍以`lwy_v2_g1.json`为准。',
        '剩余限制：原始几何为视频近似，未实测标定；中心线静态净空不能保证任意偏航/侧移轨迹；离散通路见证不等于动力学保证；保守栅格仍会拒绝有数厘米原始净空的位姿。未完成训练/评估必须以队列状态为准。',
        '进度命令：`ssh hpf@100.108.114.64 "cd ~/wang-sm/ATEC_NAV_WSM/research/exp_nav3 && cat metrics/lwy_v2_queue.json"`；实时日志以`active_log`为准，并行评估时看`active_logs`。手动同步最新报告和小产物：`python3 scripts/lwy_v2_fetch.py`。',
        '## 本轮代码清单',table(['文件','作用'],[
            ['nav3/lwy_dataset_v2.py','末门偏移、全几何反射、10 cm净空筛选、固定重采样与新划分。'],
            ['nav3/lwy_control_v2.py','显式规划参数与时域选项，旧默认保留。'],
            ['nav3/lwy_env_v2.py','可配置航向增益，物理判定原样保留；子步诊断。'],
            ['nav3/lwy_eval_v2.py','独立PRNG、单环境vmap的G1与训练闸门。'],
            ['nav3/lwy_probe_v2.py','短段、完整历史、精确现场保存/回放和单变量实验。'],
            ['nav3/lwy_v2_geometry_diag.py / lwy_v2_trace_analysis.py / lwy_summary_v2.py','失败几何、子步净空、受控实验汇总与图。'],
            ['nav3/lwy_obstacles_v2.py / lwy_safe_obstacles.py','静态+随机障碍通路检查、拒绝采样目录与运行时选择。'],
            ['nav3/lwy_mixed.py / lwy_sampler_probe_v2.py','真正按回合75/25混合与重置回归。'],
            ['nav3/lwy_memory_v2.py','全部地图+真实Agent+优化器的显存实测。'],
            ['nav3/lwy_training_runtime.py','将原训练运行时与确定性评估设置分开，恢复Dreamer level1及PPO原库默认。'],
            ['nav3/lwy_checkpoint_v2.py','无损保存bfloat16策略状态，兼容恢复本轮早期的NumPy |V2状态文件。'],
            ['nav3/lwy_gradient_diagnosis.py / lwy_forward_diagnosis.py / lwy_numerics_report.py','冻结数据的单变量数值诊断、推理核对和四项原因排查。'],
            ['nav3/lwy_train_dreamer.py / nav3/train_ppo.py','官方Dreamer循环GPU驱动、PPO显式LWY入口及状态恢复。'],
            ['nav3/lwy_ppo_update.py','保持PPO损失与更新，补充梯度范数、策略方差和回报尺度日志。'],
            ['nav3/lwy_grouped_eval.py','未采用的分组速度与逐位一致性实验，正式评估不导入。'],
            ['nav3/dreamer_adapter.py','可选JAX确定性配置覆盖，默认不变。'],
            ['nav3/lwy_policy_v2.py / lwy_compare_v2.py','独立策略PRNG、固定推理形状、统一验证/最终比较。'],
            ['nav3/lwy_comparison_runtime.py','正式评估默认保留level0；只有用户决定和候选证据齐备才可启用level1，结果使用独立_rt1文件名。'],
            ['nav3/lwy_queue_v2.py / lwy_collapse_v2.py','限时预算队列、半小时快照验证、早停与证据诊断。'],
            ['nav3/lwy_video_v2.py / lwy_report_v2.py','原版XML推理对比视频、增量报告和曲线。'],
            ['configs/lwy_v2_*.json','独立参数及失败现场配置，旧配置不改。'],
            ['scripts/lwy_v2_*.sh','可脱离客户端执行的数据生成、诊断、G1、协议与零样本任务。'],
            ['scripts/lwy_v2_fetch.py','只同步本轮报告与小产物；可有界后台等待远端完成，完成后审计进程并退出。'],
            ['tests/check_lwy.py','新增末门偏移、完整反射/双反射、阈值、种子隔离与旧文件保护检查。']])]
    report.write_text(preface+old.rstrip()+MARKER+'\n\n'.join(text)+'\n')
    if curve:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for name,data in curve.items():
            x=[float(r['label']) for r in data['rounds']]
            axes[0].plot(x,[100*r['success_rate'] for r in data['rounds']],'-o',label=name)
            axes[1].plot(x,[r['mean_gates'] for r in data['rounds']],'-o',label=name)
        axes[0].set(ylabel='Validation success (%)',xlabel='Nominal training wall h',ylim=(0,100));axes[1].set(ylabel='Mean gates',xlabel='Nominal training wall h',ylim=(0,8))
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig('figures/lwy_v2_learning_curve.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(report=str(report),status=status,final_conditions=len(evaluations))),flush=True)


if __name__=='__main__':main()
