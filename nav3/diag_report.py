"""Generate the fourth-round report and all comparison numbers from JSON."""
import json
import math
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'metrics'


def read(name):return json.loads((M/(name+'.json')).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]


def pct(x):return f'{x:.1%}'


def comparison(old,new):
    before={(r['map'],r['episode']):r for r in old['rows']};pairs=[(before[r['map'],r['episode']],r) for r in new['rows']]
    wins=sum(not a['success'] and b['success'] for a,b in pairs);losses=sum(a['success'] and not b['success'] for a,b in pairs)
    discordant=wins+losses
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    groups={}
    for a,b in pairs:groups.setdefault(a['map'],[]).append(b['gates']-a['gates'])
    cluster=[statistics.mean(v) for v in groups.values()];rng=random.Random(9042)
    boots=sorted(statistics.mean(rng.choices(cluster,k=len(cluster))) for _ in range(5000))
    return dict(improved=wins,regressed=losses,mcnemar_exact_p=p,mean_gate_delta=new['mean_gates']-old['mean_gates'],
        gate_delta_map_bootstrap_95=[boots[125],boots[4874]] if len(cluster)>1 else None,
        success_wilson_95=wilson(sum(r['success'] for r in new['rows']),len(new['rows'])))


def main():
    g=read('diag_geometry');a=read('diag_attribution');t=a['aggregate'];c=read('diag_training_census');physical=read('diag_physical_clearance')
    checks=read('diag_checks');cf=[]
    for tag,label in [('start150','原场景'),('remove_log','仅移除倒木及树皮'),('remove_opposing_rocks','仅移除后方右侧两石块'),
                      ('shift_log_left_030','仅倒木向路线左移 0.30 m'),('rotate_log_30','仅倒木绕中心转 30°'),('oracle_obstacles','仅输入障碍高度完整化')]:
        runs=[read(f'diag_{tag}_{seed}') for seed in [9137,9141]];n=sum(r['episodes'] for r in runs);k=sum(r['failures'].get('collision',0) for r in runs)
        ci=wilson(k,n);cf.append(dict(tag=tag,label=label,n=n,collisions=k,rate=k/n,collision_wilson_95=ci,
            seconds=[r['wall_seconds'] for r in runs],by_seed=[r['failures'] for r in runs]))
    comparisons=[];paired=[]
    for split in ['heldout','test']:
        for ob in [0,1]:
            for sf in [0,1]:
                old=read(f'eval_dreamer_s0_{split}_ob{ob}_sf{sf}');new=read(f'diag_eval_margin_{split}_ob{ob}_sf{sf}')
                stat=comparison(old,new);stat.update(split=split,random_obstacles=ob,filtered=sf);paired.append(stat)
                rate=lambda d:sum(r['takeovers'] for r in d['rows'])/sum(r['steps'] for r in d['rows'])
                speed=lambda d:sum(r['distance_m'] for r in d['rows'])/sum(r['duration_s'] for r in d['rows'])
                comparisons.append([split,ob,sf,new['episodes'],pct(old['success_rate']),pct(new['success_rate']),
                    f"{old['mean_gates']:.3f} → {new['mean_gates']:.3f}",f'{pct(rate(old))} → {pct(rate(new))}',
                    f'{speed(old):.3f} → {speed(new):.3f}',f"{stat['improved']}/{stat['regressed']}"])
    synthetic=[];synthetic_by_map=[]
    for method in ['dreamer_s0','ppo_s0','planner']:
        for kind in ['base','rocks']:
            d=read(f'diag_synthetic_local_{kind}_{method}');f=d['failures']
            synthetic.append([method,kind,d['episodes'],f.get('segment_pass',0),f.get('collision',0),f.get('support_or_drop',0)])
            for name in sorted(set(r['map'] for r in d['rows'])):
                rr=[r for r in d['rows'] if r['map']==name]
                synthetic_by_map.append(dict(method=method,kind=kind,map=name,episodes=len(rr),collisions=sum(r['failure']=='collision' for r in rr)))
    gaps=[r['envelope_raw_gap_terminal_m'] for r in a['collisions']];first=[r['envelope_raw_gap_first_m'] for r in a['collisions']]
    collision_progress=[r['s'] for r in a['collisions']]
    diagnostic_runs=[]
    for p in sorted(M.glob('diag_*.json')):
        d=json.loads(p.read_text())
        if isinstance(d,dict) and 'arguments' in d and 'wall_seconds' in d:diagnostic_runs.append(dict(source=p.name,seconds=d['wall_seconds']))
    repair_seconds=sum(read(f'diag_eval_margin_{s}_ob{o}_sf{f}')['total_wall_seconds'] for s in ['heldout','test'] for o in [0,1] for f in [0,1])
    summary=dict(counterfactuals=cf,paired_comparisons=paired,synthetic_by_map=synthetic_by_map,
        diagnostic_run_wall_seconds=sum(r['seconds'] for r in diagnostic_runs),diagnostic_run_max_seconds=max(r['seconds'] for r in diagnostic_runs),
        repair_evaluation_wall_seconds=repair_seconds,training_seconds=0,runs=diagnostic_runs)
    (M/'diag_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# nav3 第四轮：mini-trail 集中碰撞诊断','',
        '本报告由 `python -m nav3.diag_report` 从 `metrics/diag_*.json` 和原评估生成。所有距离单位为 m；原始评估、模型权重和生产环境几何均未修改。',
        '', '## 结论','',
        f"原 Dreamer 的 {len(a['collisions'])}/30 次碰撞已逐回合精确重现，终止进度为 {min(collision_progress):.3f}–{max(collision_progress):.3f} m，与旧记录最大差值 {a['old_collision_progress_max_difference_m']:.1f} m。根因是**保守栅格下的绕行余量不足（c/e），并受到训练分布缺少倒木后方右侧石块组合（d）的影响**。解码边界误差（b）有量化证据；‘输入完全看不到倒木’（a）不是主要解释。",'',
        f"必须区分栅格碰撞和实体接触：{len(gaps)}/{len(gaps)} 次碰撞在包络采样点上真实命中障碍栅格，但终止包络与原始倒木/树皮 XY 几何仍相隔 {min(gaps)*100:.1f}–{max(gaps)*100:.1f} cm，中位 {statistics.median(gaps)*100:.1f} cm。不能将这批结果称为已经发生实体撞木。",'',
        '已验证一个**通用安全过滤候选**：原几何、原 checkpoint、原 8 条件下，mini-trail 开过滤成绩提高，留出总体通过率没有下降；代价是约八成动作改由规划器执行。**无过滤学习策略没有修复，没有微调，也没有把 mini-trail 或本轮诊断地图送入训练。**',
        '', '## 1. 现场与几何核对','',
        f"主路线 s=150–170 的宽度为 {min(r['width'] for r in g['stations']):.2f}–{max(r['width'] for r in g['stations']):.2f}；设计纵坡 {g['route_grade_percent_min']:.3f}%–{g['route_grade_percent_max']:.3f}%（高度场另有毫米级粗糙度）。这段没有台阶、栏杆、导引带或导引带支架。第 7 门位于 s=164.8。无导引带的右岔路从主线 s=148 延伸至 s=162，横移最终 4.6，宽度由 2.6 过渡至 1.65；此处并非窄路段。",'',
        '下表列出范围内所有道路相关 prop；树皮球逐个列出。树干等旁侧物体、每 0.2 m 中心线坐标/宽度/航向、岔路坐标均保存在 `metrics/diag_geometry.json`。尺寸沿用原 `props.json`：box/ellipsoid 为半尺寸，圆柱为半径/半高，sphere 为半径；fromto 为轴端点。', '',
    ]
    rows=[]
    for p in g['props']:
        if p['kind']=='tree':continue
        coords=' → '.join('('+', '.join(f'{v:.3f}' for v in q)+')' for q in [p['fromto'][:3],p['fromto'][3:]]) if 'fromto' in p else '('+', '.join(f'{v:.3f}' for v in p['pos'])+')'
        rows.append([p['index'],p['kind'],f"{p['route_s']:.2f}",coords,', '.join(f'{v:.3f}' for v in p['size']),f"{math.degrees(p.get('yaw',0)):.2f}°" if 'fromto' not in p else '轴线定义'])
    lines.extend([table(['索引','类别','s','中心/轴端点 xyz','原始 size','yaw'],rows),'',
        '倒木轴长 1.60、半径 0.14，树皮半径 0.145；与主路线成 90°，轴心在中心线左侧 0.86，内端在左侧 0.06。后方右侧石块在 s=165.4 和 166.8，横向分别约 −1.306、−1.270。右岔路、门与这些石块均保留在现场图内。','',
        '![现场俯视、29 次首次碰撞足迹与三方法轨迹](figures/diag_scene.png)','',
        f"从序列化 props 重建 {g['raster_rebuild_local_cells']:,} 个现场格，障碍占据位差异为 {g['raster_rebuild_collision_bit_mismatch']}。倒木使用 fromto 端点计算旋转，未发现轴线旋转/尺寸错误。地图原点和最近格采样均按 `floor((xy-origin)/0.1+0.5)`；先以半格对角线 {g['raster_pad_m']:.5f} 膨胀，再按最近格查询。每 50 ms 检查 1.03×0.58 包络的 5×5 点，29 次首次命中的 kind 均为 2（倒木/树皮）。",'',
        f"首次命中时与原始几何的间隙为 {min(first)*100:.1f}–{max(first)*100:.1f} cm。该间隙来自保守栅格覆盖及查询量化，超出了名义包络本身的 5 cm 单侧裕量；`top` 还将球/圆柱投影内高度近似为顶高，按高于地面 0.22 判障碍。这是已记录的 2.5D 保守近似，并非证明原始实体在这些位置占据空间。",'',
        '判定分两层：**对当前栅格协议，碰撞判定正确；对实体接触语义，这些是保守提前终止。** 本轮未改生产地图、阈值或足迹，也不以放松标准冒充策略修复。旧评估仍可比较原协议下的安全裕量，但不能直接作为真实物理碰撞概率。若以后把判定改为几何精确接触，会影响所有地图、方法与训练奖励，必须重建地图指纹并重新评估；不能只替换 mini-trail 的分数。',
        '', '## 2. 归因证据','',
        f"29 个失败回合各保存碰撞前 26 个策略帧（5.2 s），共 {t['frames']} 帧。`metrics/diag_full_dreamer.npz` 包含六通道 BEV、priv_trav、解码概率、公共向量、漂移、原始动作、位姿，以及每帧四个 50 ms 的跟踪命令/实际速度/足迹命中。诊断动力学重放与环境终点最大误差为 0。目标点和速度统计均按环境实际的 [-1,1] 裁剪计算。",'',
        table(['归因','可复现数字','判断'],[
            ['(a) 观测/噪声',f"倒木格直接检出 {t['log_seen']}/{t['log_cells']}={pct(t['log_recall'])}；容许一格邻域对齐 {pct(t['log_recall_with_one_pixel_tolerance'])}；完整障碍高度输入仍 52/60 碰撞",'存在稀疏/时延偏移；不是充分解释'],
            ['(b) 世界模型',f"危险边界 {t['hazard_false_free']}/{t['hazard_cells']}={pct(t['hazard_false_free_rate'])} 被误解码为可通行；局部窗口整体 IoU {t['overall_iou']:.3f}；原始倒木格误判可通行 {pct(t['log_pred_false_free_rate'])}",'边界误差参与，未见整体抹掉倒木'],
            ['(c) 策略',f"{pct(t['dreamer_unsafe_target_rate'])} 目标位于真值禁行区；{pct(t['dreamer_known_unsafe_target_rate'])} 在真值和解码图中都禁行；平均速度档 {t['dreamer_speed_mean']:.3f} m/s",'明确存在未满足栅格余量的动作；解码头并非动作硬约束'],
            ['(d) 分布差异','倒木尺寸/朝向/宽度同类 200/200；邻近右侧石块组合 0/200；独立新图增加该组合后 Dreamer 碰撞 0/60→11/60','组合缺口有干预证据，不是倒木本身从未见过'],
            ['(e) 栅格保守性',f"原 29 次终止均未触及原始几何；仅取消倒木占据膨胀后 30/30 通过该段，20 Hz 最小原始几何间隙 {physical['all_min_gap_m']*100:.2f} cm",'栅格安全余量是实际触发边界；不等于实体接触'],
        ]),'',
        '![训练图特征频次](figures/diag_training_features.png)','',
        '输入检出率以当前真值位置投影的倒木格为分母、BEV 已知且高度 >0.22 为检出；输入本身延迟一帧，故同时给出 3×3 邻域容差，不能把 32.3% 的精确格不一致全部解释成漏检。危险边界定义为距内端 2 m 内、障碍距离 <0.6、支撑边距 >0.4、priv_trav 不可通行的格；同一序列帧高度相关，以上帧比例是描述统计，未作独立样本显著性检验。', '',
        '![第 0 回合的六通道、真值与解码时序](figures/diag_bev_sequence_ep00.png)', '',
        '![第 0 回合目标与 20 Hz 跟踪速度](figures/diag_controls_ep00.png)','',
        '其余每个失败回合都有 `figures/diag_bev_sequence_epNN.png` 和 `diag_controls_epNN.png`。通道图展示 −5.2、−3.2、−2.0、−1.0、−0.2 s 五个时刻；完整 5 Hz / 20 Hz 数据保存在 NPZ，可由同一脚本重画。灰度/颜色范围：相对高程 −0.5～0.5 m、障碍高度 0～0.5 m，其余 0～1。','',
        table(['同一 Dreamer 观测历史上的动作','目标真值禁行比例','速度档 m/s'],[
            ['Dreamer',pct(t['dreamer_unsafe_target_rate']),f"{t['dreamer_speed_mean']:.3f}"],
            ['PPO shadow',pct(t['ppo_shadow_unsafe_target_rate']),f"{t['ppo_shadow_speed_mean']:.3f}"],
            ['规则规划器',pct(t['planner_unsafe_target_rate']),f"{t['planner_speed_mean']:.3f}"],
        ]),'',
        'PPO shadow 的 GRU 顺序读取相同公共观测；它不是 PPO 自己执行动作形成的闭环轨迹。目标点落在禁行区也不是完整轨迹碰撞的充分条件。闭环局部出生试验为 Dreamer 27/30、PPO 3/30 碰撞、规划器 0/30；俯视叠加显示后两者在倒木内端下方留出更多横向余量。PPO 速度也高，不能只凭速度差就把根因归成“太快”。','',
        '训练统计为全部 200 张原始训练图的 props、路线和 metadata，未用小样本替代。判定“邻近右石块”为 |Δs|<3.5 且 −1.6<横向偏移<0；生成器 `repair_bypass()` 正好把此类石块移至外路肩。', '',
        table(['匹配特征','训练图频次 /200'],[[k,v] for k,v in c['counts'].items()]),
        '', '## 3. 反事实验证','',
        '以下均为推理，关闭随机障碍与过滤；两个独立批次种子 9137、9141，每批 30 环境，从 s=150 出生至 s≥171 即记为段通过。冷启动 GRU/RSSM 与全路线历史不同，故单列完整路线复现。置信区间为同一地图条件下回合噪声的 Wilson 95% 区间，不能解释为地图泛化区间。','',
        table(['单变量干预','碰撞/回合','碰撞率','95% 区间'],[[r['label'],f"{r['collisions']}/{r['n']}",pct(r['rate']),'–'.join(pct(v) for v in r['collision_wilson_95'])] for r in cf]),'',
        '![单变量反事实](figures/diag_counterfactuals.png)','',
        '移除/平移/转向均只作用于诊断地图副本。完整障碍高度输入仅替换观测障碍通道并将对应障碍标成已知，其他通道仍保留原噪声；它使用真值，只是诊断 oracle，不能部署。', '',
        f"栅格敏感性对照从路线起点完整运行：只将倒木/树皮的占据 mask 改为按格中心落入原始半径判断，保留原 top/BEV 和其他物体；原 {len(gaps)}/30 碰撞变成 0/30，全部通过 s=171。进一步逐个检查这些轨迹 s=162–166 内的 20 Hz 包络，原始几何投影重叠 {physical['physical_xy_projection_overlap']}/30，最小净空 {physical['all_min_gap_m']*100:.2f} cm。这是阈值敏感性证据，**未作为环境修复采用**，也不是 8 门完整成绩。",'',
        '独立随机新图使用 seeds 30000–30003，经 `TrainingScene(seed, mini=False)` 生成，不读 mini-trail 的路线/高度场/props。每图生成一对版本：原随机生成器的通路修复结果，以及在其自身倒木后添加两块随机采样的右肩石块（Δs 为 1.1–1.7 / 2.5–3.1，横向 −1.4～−1.2，半尺寸和 yaw 独立采样）。它们仅是诊断留出集，没有加入训练 manifest。每图 15 回合，从本图倒木前 6 m 至后 7 m：','',
        table(['方法','新图版本','回合','段通过','碰撞','支撑失败'],synthetic),'',
        '较早的 −14 m 出生对照也保留在 `diag_synthetic_base_*` / `diag_synthetic_rocks_*`：Dreamer 基线有 30/60 个支撑失败，混入了上游路段问题，故随后将实验缩短到 −6 m 隔离倒木组合。没有删除这些失败，也没有把它们算成倒木碰撞。分地图结果见 `diag_summary.json`；样本只有 4 张独立新图，不能外推为普遍失效率。',
        '', '## 4. 最小防护实验与原协议对照','',
        '没有修改网络、奖励、训练地图或 checkpoint。新增可选 `nav3.diag_candidate.margin_filter`：已观测高度 >0.21 m 的障碍仍处于 x∈(−0.6,3.5)、|y|<1.4 时，继续使用原规则规划器；并保留旧过滤器的目标风险、走廊和未知触发条件。它只读公共 BEV，不读取 s、地图名、priv_trav 或解码图。后向范围覆盖机体后半段，避免绕行尚未结束便交还策略。', '',
        '候选在原生产几何上评估；未采用任何删障碍或放松碰撞的反事实地图。每条件留出 100 回合、mini-trail 30 回合，共 520 回合。原方法为旧过滤器，新方法为此候选；sf=0 两者动作逻辑相同，完整重跑核对。所有新结果写入 `metrics/diag_eval_margin_*.json`。','',
        table(['数据集','随机障碍','过滤','N','旧全通过','候选全通过','平均门 旧→新','接管率 旧→新','速度 旧→新','成败翻转 改善/退化'],comparisons),'',
        '留出图没有总体通过率回退。开启随机障碍和过滤时，有 7 个旧成功回合变失败，也有 7 个旧失败变成功，配对 McNemar 精确 p=1；因此不能声称每个回合都无退化。平均门数变化及按 20 张地图分组重采样的区间如下（5,000 次，固定 seed=9042）：','',
        table(['留出条件 ob/sf','平均门差','地图 bootstrap 95% 区间','改善/退化'],[[f"{r['random_obstacles']}/{r['filtered']}",f"{r['mean_gate_delta']:+.3f}",'–'.join(f'{v:+.3f}' for v in r['gate_delta_map_bootstrap_95']),f"{r['improved']}/{r['regressed']}"] for r in paired if r['split']=='heldout']),'',
        f"防护评估累计墙钟 {repair_seconds/60:.2f} 分钟（不含首次 Agent 加载；日志可审计），训练时间为 0，远低于 2 小时 GPU 修复预算。单个诊断运行最长 {summary['diagnostic_run_max_seconds']/60:.2f} 分钟，均低于 30 分钟。该候选保留为显式选择，未替换原默认过滤器，便于旧协议复现。高接管率说明主要收益来自规划器；后续应训练出足够的障碍余量，而不是把本表当作 Dreamer 已学会。",'',
        '未进行微调：本轮已经找到无需训练、在原协议下有效的最小防护候选；学习策略层的最小充分修复仍需区分贴边奖励、组合增广与解码边界监督。当前证据不支持承诺任选一种短训必然改善，故没有硬训，也未创建 `dreamer_s0_ft` 或修改原训练目录。',
        '', '留出次要指标并非全部无回退：开过滤后整体、遮挡和随机障碍邻域 IoU 均下降，按地图配对 bootstrap 区间不含 0（固定 seed=9042，5,000 次）。权重没有变化，新的接管轨迹改变了观测分布与帧权重；这不能证明模型权重退化，但确实是候选系统按当前评估协议测到的回退。故只把它保留为可选防护实验，未宣称全指标优于旧系统。','',
        table(['留出 ob/sf','IoU 项','旧 → 新','差值 95% 地图 bootstrap'],[
            [f"{r['obstacles']}/{r['filtered']}",r['channel'],f"{r['old']:.4f} → {r['new']:.4f}",'–'.join(f'{v:+.4f}' for v in r['delta_95_map_bootstrap'])]
            for r in read('diag_iou_comparison') if r['filtered']]),
        '', '## 5. 无障碍邻域 IoU 修正','',
        '修正 `evaluate.py` 的新结果：分母为零时写 JSON null。修正 `report.py`：从历史逐回合 trav_union 判断未定义值，并把无随机障碍时第三项输出为“不适用”。其他两项沿用原值，不把未定义当作 0。旧 JSON 逐字节保留；修正后的派生汇总为 `metrics/diag_evaluation_summary.json`，表为 `metrics/diag_evaluation_table.md`；`REPORT.md` 的同条件表已重生成。',
        '', '## 6. 未解决事项与限制','',
        '1. 无过滤 Dreamer 的原协议表现没有改善；防护只是隔离危险动作。随机障碍下候选仍有失败，不能宣称安全保证。',
        '2. 保守栅格造成的额外有效包络应在后续设计中明确。建议先用独立合成 primitive 场景比较高分辨率实体接触与当前 mask，量化裕量，再决定保留标准并训练适应，还是整体升级碰撞检测；不得仅为 mini-trail 放宽判定。',
        '3. 原训练修复函数应改为“验证通路连通后保留可绕行的右肩石块”，并在独立随机训练图上增广倒木+门+石块组合；本轮只在诊断图验证，没有把这些地图塞入原训练集。',
        '4. 只有一个训练种子；本轮完整评估的筛选仍使用已有 heldout 和 mini-trail，不能替代新盲测集。辅助解码头的边界错误与动作错误有共同原因，尚无直接潜变量干预证明两者的因果方向。',
        '5. PPO 的本轮全路线补录仅 22/30 通过 s=171（旧整段为 24/30）。第二次补录仍为 22/30；没有声称 PPO 逐位复现旧评估。原记录不动，现场图使用明确标注的本轮实跑轨迹。无插桩原评估器重复结果见下方补充；差异原因未确定，不能推测成已证实的数值原因。',
        '', '## 7. 复现、文件与完整性','',
        '远端工作目录始终为 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`。下列命令从该目录执行，先 `source scripts/env.sh`；GPU 命令放入 `nav3_` 前缀 tmux，重跑时请改 `--tag` 或另设结果根目录，脚本拒绝覆盖同名 rollout JSON。', '',
        '```bash\n# 有意报 AssertionError：确认原始故障仍存在\n"$NAV3_PYTHON" -m nav3.diag_geometry repro\n# CPU：现场、200 图统计、测试地图副本\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_geometry geometry\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_geometry census\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_geometry variants\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_synthetic\n# 补录 30 回合；PPO/planner 另用 --method 与新 --tag\n"$NAV3_PYTHON" -m nav3.diag_rollout --tag full_dreamer_new --record --shadow\n# 全部已执行命令在 scripts/diag_counterfactuals.sh、diag_validation.sh、diag_candidate.sh\n# 独立安全候选 8 条件（已存在结果自动跳过）\n"$NAV3_PYTHON" -m nav3.diag_candidate --all\n# 从已保存记录重画全部 29 回合图\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_analysis --all-episodes\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" -m nav3.diag_clearance\n"$NAV3_PYTHON" -m nav3.report --tables-only --summary-output metrics/diag_evaluation_summary.json --report-output metrics/diag_evaluation_table.md\nJAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py\n"$NAV3_PYTHON" -m nav3.diag_report\n```','',
        table(['代码文件','改动说明'],[
            ['nav3/evaluate.py','零分母 IoU 写 null；新增可选 filter_fn，默认行为不变。'],
            ['nav3/report.py','从原始 union 恢复未定义 IoU；支持另存派生汇总/表，避免覆盖旧 JSON。'],
            ['nav3/diag_rollout.py','只读模型补录、同输入动作、oracle 与短段实验；核对 20 Hz 动力学镜像终点。'],
            ['nav3/diag_geometry.py','几何清单、200 图统计、故障命令与独立反事实地图根目录。'],
            ['nav3/diag_synthetic.py','重新随机生成独立 holdout 风格配对地图，绝不进入训练。'],
            ['nav3/diag_analysis.py','29 回合归因数字、现场图、六通道/真值/解码图与速度图。'],
            ['nav3/diag_statistics.py','按地图配对 bootstrap 核对留出 IoU 的显著下降。'],
            ['nav3/diag_plots.py','从派生统计生成反事实与训练特征汇总图。'],
            ['nav3/diag_clearance.py','按 20 Hz 包络核对原始倒木实体投影净空。'],
            ['nav3/diag_candidate.py','显式可选的通用观测障碍防护及原 8 条件评估。'],
            ['nav3/diag_report.py','从保存 JSON 生成本报告、配对统计及修正旧报告表。'],
            ['tests/check_diag.py','验证 IoU、旧结果哈希、地图隔离、原指纹、checkpoint 哈希及诊断完整性。'],
            ['scripts/diag_counterfactuals.sh / diag_validation.sh / diag_candidate.sh','按顺序运行各组短实验，保存日志。'],
            ['scripts/diag_repeat_ppo.py','用原 evaluate 逻辑独立复核 PPO 差异。'],
        ]),'',
        f"完整性检查通过：{checks['protected_original_files']} 个原指标文件哈希不变；两个 latest.pkl 与首次补录/已有视频记录的 SHA256 一致；生产环境指纹仍为 `{checks['environment_fingerprint']}`。训练/留出种子不相交，mini 和 seeds 30000–30003 均不在训练 manifest。原 dreamerv3 工作树和上游 main 未编辑、未提交、未 push。",'',
        '失败/中断记录：Mac 的系统 Python 缺少 numpy，两个 uv 包源尝试超时，因此计算转到远端已有环境；没有往共享环境装包。首次 CPU 分析反复解压 NPZ 较慢，主动中断后改为一次加载，原 KeyboardInterrupt 保留在 `logs/diag_analysis.log`。取消倒木膨胀的首版诊断地图重建时带入 2 个远处非倒木占据位变化，已弃作单变量证据；修正版 `diag_log_center_occupancy_full_v2` 强制保留全部非倒木占据位并重跑，观察层完全不变，旧试验记录保留。原始碰撞断言仍报错是预期结果，见 `logs/diag_repro.log`。没有隐去失败实验。',
    ])
    lines.extend(['','本轮保留的故障输出（原策略不变，因此断言失败是预期）：','', '```text\nREPLAY: collision= True footprint_hit_samples= 1 kind= [2.0] s= 163.44058227539062\nAssertionError: Reproduced Dreamer collision at 163.44 m\n```', '', '本地依赖失败输出：`ModuleNotFoundError: No module named numpy`；uv 两个源均为 `Request failed after 3 retries` / `operation timed out`。被主动中断的旧分析日志末尾为 `KeyboardInterrupt`。'])
    lines.extend(['','附加复现：`python -m nav3.diag_statistics` 生成 IoU 配对区间；`python -m nav3.diag_plots` 生成两张汇总图。文档修改为 `DIAG_REPORT.md`、`REPORT.md` 的 IoU 表和 `README.md` 的诊断入口。'])
    repeat=M/'diag_ppo_original_evaluator_repeat.json'
    if repeat.exists():
        d=json.loads(repeat.read_text());lines.extend(['',f"PPO 无插桩原评估器复跑：全通过 {sum(r['success'] for r in d['rows'])}/{d['episodes']}={pct(d['success_rate'])}，失败分类 `{json.dumps(d['failures'],ensure_ascii=False)}`。"])
    status=M/'diag_remote_final.json'
    if status.exists():
        d=json.loads(status.read_text());lines.extend(['',f"收尾状态：{d['summary']}；详细原始输出见 `metrics/diag_remote_final.json`。"])
    (ROOT/'DIAG_REPORT.md').write_text('\n'.join(lines)+'\n')
    # Only the presentation table changes; old evaluation JSONs remain immutable.
    ev=read('diag_evaluation_summary');rows=[]
    for e in ev:
        iou='不适用' if e['trav_iou'] is None else '/'.join('不适用' if v is None else f'{v:.3f}' for v in e['trav_iou'])
        rows.append([e['method'],e['split'],int(e['obstacles']),int(e['filter']),e['episodes'],f"{e['mean_gates']:.3f}",pct(e['success_rate']),f"{e['fail_per_100m']:.3f}",pct(e['takeover_rate']),f"{e['mean_speed']:.3f}",iou])
    replacement=table(['方法','数据集','随机障碍','过滤','回合','平均门/8','全通过','失败/100m','接管率','速度 m/s','IoU 全/遮挡/障碍邻域'],rows)
    path=ROOT/'REPORT.md';text=path.read_text();start=text.index('| 方法 | 数据集 |');end=text.index('\n\n',start)
    text=text[:start]+replacement+text[end:]
    note='\n第 4 轮勘误：无随机障碍的障碍邻域 IoU 为不适用；原评估 JSON 保留，派生值见 `metrics/diag_evaluation_summary.json`。现场诊断与可选防护实验见 [DIAG_REPORT.md](DIAG_REPORT.md)。\n'
    if '第 4 轮勘误' not in text:text+=note
    path.write_text(text)
    print('Generated DIAG_REPORT.md, diag_summary.json, and corrected REPORT.md table.')

if __name__=='__main__':main()
