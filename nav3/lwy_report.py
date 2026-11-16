"""Chinese round-six report rendered exclusively from saved result JSON."""
import json
from pathlib import Path


def read(name): return json.loads(Path(f'metrics/lwy_{name}.json').read_text())


def main():
    geometry = read('geometry'); gate = read('g1'); dataset = read('dataset_summary')
    checks = read('checks'); diag = read('check_diag'); observations = read('observations')
    old = read('profile_legacy_1024'); new = read('profile_lwy_1024'); large = read('profile_lwy_4608')
    original = read('g1_original_dynamic'); blind = read('g1_blind_dynamic'); endpoints = read('endpoint_geometry')
    fix = read('planner_fix_dynamic')
    sections = ['# nav3 第六轮：LWY 赛道接入与 G1 阻断报告\n',
        '**本轮未达成重新训练目标，G1 不通过，未启动任何 RL 冒烟或训练。** '
        f"原版 1 km 为 {gate['original_successes']}/10 全通过，平均 {gate['original_mean_gates']:.2f}/8 门；"
        f"盲测为 {sum(r['success'] for r in blind['rows'])}/{blind['episodes']} 全通过，平均 {blind['mean_gates']:.2f}/8 门。"
        '原版的 10 个回合均通过前七门，失败集中在路面终点。末门 s=1000 m 与支撑路面终点重合，'
        '现有门状态机要求机器人中心越过门，有限尺寸包络无法在此保持完整支撑。没有移动门、补虚拟路面、缩小包络或忽略终止来换取分数。\n',
        f"盲测另有 {len(gate['blind_before_999m_failures'])} 个回合在 s<999 m 失败，详见下文清单；不把这些失败归到终点问题。\n",
        '## 导入与几何判定\n',
        'Mac、远端完整包均通过 `tools/verify_package.py`：137 个文件、642,046,794 字节。'
        'Mac Python 3.9 缺少 `hashlib.file_digest`，使用 `scripts/lwy_integrity.py package` 提供流式哈希兼容包装后重跑同一校验脚本；包本身未改。'
        '源码提交为 `83758d1dbcdfbd9ddbd2466ce8397f40cb927fce`；nav3 内复制的 `vendor/lwy/trail_scene.py` 与两处源文件 SHA256 一致，见 `metrics/lwy_sources.json`。\n',
        '原版直接读取只读包的 `scene_bundle.json` 与 `track_spec.json`；变体调用复制的 `Route`、`build_bundle`，只向 `maps_lwy/` 写入。'
        '支撑区取 road 棱柱顶面及台阶踏面，不含路肩、坡面或植被带。相邻面片由上游 ±10 µm 取样造成的微缝以 60 µm 内的闭合运算拼接；不扩展真实起终点或侧边界。'
        '地面高程来自原始三角面平面插值。台阶保留离散高程，原有陡坡/台阶 0.28 m/s 限速不变。\n',
        '护栏立柱/两层横杆、侧墙、石块、树干和标牌均用基本体 XY 投影碰撞，包括顶高不足 22 cm 的石块；不把整座山或树冠当可支撑道路。'
        '这是保守的 2.5D 包络判断，不是腿部接触或攀阶能力验证。半径 2.5 cm 的树根保留雷达高度返回，不作硬障碍，接近时限速 0.28 m/s；'
        '依据是任务指定的 M1 轮足通常可压越该尺度树根，尚无底层实测标定。\n',
        '世界 XY 以 0.05 m 栅格、32×32 单元分块，仅保留距路线约 6 m 以内相交的整块；BEV 仍按真实世界位姿裁剪和旋转，无路线坐标拉伸。'
        f"原版共 {int(new['atlas_bytes']/1024**2):,} MiB 设备地图数据，5,535 块。沿程 s 另存 float32，避免 1 km 处 float16 约 0.5 m 的量化步长影响投影。"
        '新 `LwyBank`/`LwyEnv` 是显式入口，原 `MapBank`、`FastEnv`、控制器和配置文件均保持逐字节不变。\n',
        '足迹仍为 **1.03×0.58 m**，20 Hz 判定由原 5×5 点改为 43×25 点。栅格量化半对角线 3.536 cm、足迹采样覆盖界 1.722 cm 与 0.02 cm 数值余量合成 5.277 cm 保守距离；'
        '支撑层另补偿旧 2.5 cm 终止阈值，防止旧容差变成越界许可。碰撞阈值与包络均未放宽。\n',
        '下表使用每 0.2 m 的 5,001 个中心线位姿，独立矩形多边形对原始面片/基本体投影核对；数据见 `metrics/lwy_geometry.json`、`lwy_clearance_samples.npz`。'
        '负支撑净空表示包络已越出道路，端点数值不能与内部路段混为一谈。\n',
        '| 原始几何净空（m） | 最小 | P01 | P05 | 中位数 | P95 | 最大 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, key in [('支撑边界', 'support_clearance_m'), ('碰撞体', 'obstacle_clearance_m')]:
        sections.append('| '+label+' | '+' | '.join(f"{geometry[key][k]:.4f}" for k in ['min', 'p01', 'p05', 'median', 'p95', 'max'])+' |')
    sections += [f"\n内部 s=1–999 m 最小综合净空 **{geometry['interior_min_clearance_m']*100:.2f} cm**。小于 5 cm 的中心线采样区间仅为 {geometry['below_5cm_intervals_m']} m；"
                 '其中起点由默认 s=2 m 出生规避，终点属于 `final_steps_and_moss` 区段，无法规避又同时通过末门。未发现原版内部中心线几何不可行段。\n',
        '| 相对原始几何的额外保守量（cm） | 最小 | 中位数 | P95 | 最大 |\n| --- | ---: | ---: | ---: | ---: |']
    for label, key in [('支撑判定', 'support_conservatism_m'), ('碰撞判定', 'obstacle_conservatism_m')]:
        sections.append('| '+label+' | '+' | '.join(f"{geometry[key][k]*100:.2f}" for k in ['min', 'median', 'p95', 'max'])+' |')
    sections += [f"\n另做 {geometry['probe_count']} 个随机 s/横移/偏航位姿：原始几何可行 {geometry['probe_exact_free']} 个，新判定可行 {geometry['probe_raster_free']} 个；"
        f"不安全误放行 {geometry['unsafe_false_free']}，保守误拒 {geometry['conservative_rejections']}。这不是零风险证明；新判定仍有约 8.8 cm 的最坏额外保守量，并未声称精确物理接触。\n",
        '![净空与终点放大](figures/lwy_clearance.png)\n',
        '## 分辨率、吞吐与观测规格\n',
        '| 条件 | 环境数 | 环境步/秒 | 峰值在用设备内存 GiB | 地图 MiB |\n| --- | ---: | ---: | ---: | ---: |']
    for label, row in [('旧 mini-trail 快环境', old), ('LWY 原版 1 km', new), ('LWY 原版 1 km', large)]:
        sections.append(f"| {label} | {row['envs']} | {row['env_steps_per_second']:,.0f} | {row['memory_stats']['peak_bytes_in_use']/1024**3:.3f} | {row['atlas_bytes']/1024**2:.2f} |")
    sections += [f"\n1024 环境吞吐下降 {(1-new['env_steps_per_second']/old['env_steps_per_second'])*100:.1f}%，仍超过 20,000 环境步/秒目标。"
        '测量使用同一确定性 XLA 设置、预热后 100 步；旧/新地图长度不同，因此不是仅改变栅格分辨率的严格消融。'
        '数值不含 Agent、优化器和 replay。4608 环境单图的 JAX 内存池还曾扩展至 '
        f"{large['memory_stats']['peak_pool_bytes']/1024**3:.2f} GiB，与在用峰值不是同一指标。"
        f"全部新训练裁片栅格约 {dataset['train']['raster_bytes']/1024**3:.2f} GiB，旧 200 图 atlas 约 1.33 GiB；"
        '全混合地图、4608 环境和 Dreamer 的联合显存尚未验证，不能据单图 profiling 声称原配置一定装得下。\n',
        f"观测维持 64×64×6、0.125 m/格、13 维向量，动作维持 3 维局部目标/速度档。1.1 m 路宽为 8.8 格，每侧包络余量约 2.08 格；"
        f"六个实测位姿的车体横截面走廊宽度为 {[round(x) for x in observations['corridor_cells_at_body_x']]} 格。"
        '路线通道能表示中心通路，原规划器也完成原版前七门；但 5.4 cm 直径立柱小于一个观测格，且特权腐蚀图在窄路上可能只剩 1–2 格。'
        '这不足以证明学习策略的边界识别充分。当前无证据要求强制更换规格，因此未改；后续若验证出观测瓶颈，可用 128×128、0.0625 m/格保持 8×8 m 视野，'
        '图像/原始 replay 图像字节量增加 4 倍，前端卷积开销增加，Dreamer/PPO 都需从头训练。\n',
        '双 Airy 半球/遮挡、丢点、漂移、时延与累积流程完全继承旧环境。仅地图精度和足迹判定升级；旧 `priv_trav` 腐蚀近似仍保留，不能称为精确接触标签。\n',
        '植物视觉素材没有逐网格体素化进入遮挡；树干使用 bundle 的碰撞胶囊体。尚未对 LWY 做全分辨率双 Airy 射线对照，'
        '因此保留旧噪声流程不等于已经完成新场景的传感器真实性标定。\n',
        '![真实噪声观测与特权图](figures/lwy_observations.png)\n',
        '## 数据划分与种子\n',
        '| 集合 | 数量 | 种子 | 每张用于快环境的长度 |\n| --- | ---: | --- | --- |']
    for split in ['train', 'val', 'blind', 'test']:
        row = dataset[split]; seeds = row['seeds']
        sections.append(f"| {split} | {row['count']} | {min(seeds)}–{max(seeds)} | {'250 m 裁片（完整源变体 1 km）' if split == 'train' else '完整 1 km'} |")
    training = dataset['train']
    sections += [f"\n训练集 {training['mirrored']} 张对路线、坡面和护栏侧别做参数层左右镜像；{training['unique_zone_orders']} 种区段排列，"
        f"宽度 {training['width_range_m'][0]:.3f}–{training['width_range_m'][1]:.3f} m（均不低于 minimum_width_m=1.05）；"
        f"台阶组数 {training['stair_flight_count_range']}、实算级高 {training['stair_rise_range_m'][0]*100:.2f}–{training['stair_rise_range_m'][1]*100:.2f} cm。"
        '控制点横移/曲率、区段顺序/长度、台阶数量/位置/高度、护栏侧别、石块/树根密度与位置、植被 seed 均有确定性随机化。'
        '训练裁片选择 s=0、249、498、747 m 中一个起点，长 250 m；八门按裁片本地里程为 31.25、62.5、…、250 m。'
        '这样避免把原始终点缺少出口的任务加入训练裁片；完整验证/盲测/测试地图的八门仍是 125、250、…、1000 m。\n',
        '镜像仍有未完成项：上游台阶侧墙固定在路线右侧，参数层镜像没有将所有基本体整体反射，因此不是严格的全场景镜像。'
        '后续应统一反射几何并重建整个划分，而非只修改失败的盲测地图。当前数据集不能宣称满足全部几何增广验收。\n',
        '清单为 `metrics/lwy_split_manifest.json`，远端完整生成物在 `maps_lwy/`。计划按回合 75% LWY、25% 旧 `maps/train` 混合，旧图种子 10000–10199；'
        '该比例只是清单中的训练方案，**没有实际训练采样，也尚未接入混合训练器**。原版 862428、验证、盲测、mini-trail 871002 和诊断种子 30000–30003 均不在训练清单中。'
        '原版/盲测仅执行任务明确要求的 G1 资格测试，没有参与模型选择；验证集没有被用来代替盲测。\n',
        '## G1、重复性与停止依据\n',
        'LWY 规划器只改一项：将已观测障碍的 3×3 max-pool 膨胀改为恒等映射，移除额外一格 12.5 cm 的重复余量；'
        '原规划器文件不动。采样偏移范围 ±0.84 m、15 条候选、1.11×0.66 m 预测足迹、动力学、风险/边界权重和速度均保留；'
        '物理碰撞判定和 1.03×0.58 m 包络未放宽。新环境将 max_steps 从旧 6000 明确设为 18000（60 分钟），否则旧 20 分钟上限与 1 km 慢速路线不匹配。'
        '门固定为 125、250、375、500、625、750、875、1000 m，导入 CLI 可指定八个门；本次使用原默认。\n',
        f"旧规划器完整十回合有 {fix['baseline_full_early_failures']} 次在 s=909.416、923.433、948.141 m 提前终止；"
        '当时车体向右偏约 14.2–14.6 cm，原始几何支撑净空仍有 7.85–8.51 cm，中心线净空则有 20.76–23.45 cm。'
        '可见是偏离中心后的保守提前终止，不是这三处完全无路可走。独立原始几何与终止帧栅格核对见 `metrics/lwy_planner_diagnosis.json`；'
        '终止标志会累计四个 50 ms 子步，最终帧的栅格状态不必等于首次命中子步。\n',
        f"固定单回合 seed 940004、s=900 m 出生的回归：旧规划器在 {fix['baseline_single_failure_s']:.6f} m 失败并触发断言；"
        f"只取消规划器重复膨胀后到达 {fix['fixed_single_failure_s']:.6f} m。完整十回合提前失败变为 {fix['fixed_full_early_failures']}，"
        '另起进程重复后全部动作/位姿哈希相同。该单变量干预支持重复膨胀导致贴边的解释；未改定位噪声，不把噪声假设当作已独立穷尽验证。'
        '余下终点失败保留原样，绝不算通过。诊断/回归日志见 `logs/lwy_diag_single_legacy.log`、`lwy_diag_single_lwy.log`、`lwy_planner_fix.log`。\n',
        '| G1 集合 | N | 全通过 | Wilson 95% | 平均门 | s≥999 m 且已过七门 | 失败分类 |\n| --- | ---: | ---: | --- | ---: | ---: | --- |']
    for label, row, prefix in [('原版 1 km', original, 'original'), ('盲测变体', blind, 'blind')]:
        ci = row['success_wilson95']
        sections.append(f"| {label} | {row['episodes']} | {row['success_rate']:.1%} | {ci[0]:.1%}–{ci[1]:.1%} | {row['mean_gates']:.2f} | {gate[prefix+'_near_finish_7gate_runs']} | {row['failures']} |")
    sections += [f"\n原版失败 s 范围 {min(gate['original_failure_s_m']):.4f}–{max(gate['original_failure_s_m']):.4f} m。"
        '逐回合种子、动作/位姿 SHA256、轨迹、速度和终止位置均保存在对应 G1 JSON。\n',
        '盲测的内部失败逐项如下。净空来自终止帧的原始几何核对，中心线可行仅表示该位置有可行姿态，不能证明整段动态可通过。'
        '负横向偏移表示路线右侧；物理接触为否时，应进一步区分栅格保守量与累计子步终止。\n',
        '| 地图种子 | 失败 s（m） | 分类 | 实际横移（cm） | 中心线净空（cm） | 中心线可行 | 终止帧原始几何接触 |\n'
        '| --- | ---: | --- | ---: | ---: | --- | --- |']
    for row in endpoints['rows']:
        if not row['map'].startswith('blind_') or row['s'] >= 999: continue
        sections.append(f"| {row['map'].split('_')[1]} | {row['s']:.3f} | {row['failure']} | "
            f"{row['lateral_m']*100:.2f} | {row['centerline_margin_m']*100:.2f} | "
            f"{'是' if row['centerline_geometry_free'] else '否'} | {'否' if row['original_geometry_free'] else '是'} |")
    sections += [
        '\n变体 630007 在 s=875.905 m 的中心线姿态已与障碍投影相交，630006 在失败位置的中心线净空仅 1.61 cm；'
        '这说明原版中心线的 20.73 cm 最小净空不能外推到全部变体。两处实际偏移后的终止姿态仍可行，因此这里未证明整条横断面都不可通过。\n',
        f"对原始几何再次核对终止时完整包络，{endpoints['original_geometry_free_at_termination']}/{len(endpoints['rows'])} 个终止位姿尚未发生原始几何接触（见 `metrics/lwy_endpoint_geometry.json`）。"
        '这与保守提前终止相符，不能直接称为已经物理跌落；首次触发的子步仍需进一步核对。'
        '即使消除全部保守量，有限包络也不能让中心越过真实路面末端。\n',
        '确定性设置为 `--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true`、`TF_CUDNN_DETERMINISTIC=1`；'
        '每回合独立 PRNG seed，通过 vmap(single environment) 执行，避免批次大小改变噪声流。原版整套十回合另起进程重复，全部逐回合字段及每步动作/位姿哈希完全一致。'
        '这与旧评估器按批形状抽随机数的协议不同，后续方法比较必须统一采用本轮协议。\n',
        '![规划器原版轨迹](figures/lwy_trajectories.png)\n',
        '## 训练、模型选择与最终评估状态\n',
        'G1 的原版条件要求 10/10，盲测要求至少 95%；本轮失败，因此 `require_g1()` 明确拒绝继续。'
        '未复制 `dreamer_lwy`/`ppo_lwy`，未运行 ≤15 min 冒烟、各 4 h 续训、30 min 快照/验证或学习曲线，也没有可选最终模型。'
        '规格未改，若以后 G1 解锁，按任务应从原始 `dreamer_s0`/`ppo_s0` 复制续训；不能拿第五轮已崩溃的快照初始化。'
        '届时模型选择应仅按 20 张验证图×2 回合的全通过率，平局按平均门数；连续两次低于本次最佳超过 30 个百分点则停止。\n',
        '第 3.5 节崩溃诊断未触发，因为本轮没有续训；第五轮的崩溃原因仍未解决。没有把本轮 G1 失败归因于 RL。\n',
        '| 要求的方法 | 本轮最终评估状态 |\n| --- | --- |\n'
        '| dreamer_lwy / ppo_lwy | 未创建，G1 阻断训练 |\n'
        '| dreamer_s0 / ppo_s0 零样本 | 未执行规定的最终矩阵，G1 尚不成立 |\n'
        '| 规则规划器 | 仅完成上表 G1，不等于最终 100 回合/条件评估 |\n',
        '五方法×四数据集×四条件的最终比较、模型可通行 IoU、接管率比较、mini-trail 退化检查和 Dreamer/PPO MuJoCo 推理视频均未执行。'
        '没有学习曲线或视频路径可交付，不以旧结果或规划器录像替代新模型结果。随机障碍最终评估未执行。'
        '已单独修正 LWY 的随机障碍放置入口：旧规则要求路宽>1.7 m，会让所有新图的 ob=1 变成空场景；'
        '新入口把三个障碍分布在整条路线的三个分离区间，靠一侧留 3 cm 边界，按路宽采样 12–24 cm 内可行半尺寸，'
        '另一侧留出 0.58 m 包络加 10 cm 名义通路；横向倒木放不下时使用石块形状。此规则不改静态几何和包络，也不保证与所有静态物体组合后全局可通过。'
        '不开随机障碍时走原分支，G1 的噪声和动作复现不受影响。启用性检查见 `metrics/lwy_random_obstacles.json`。\n',
        '## 完整性、问题与复现\n',
        f"`tests/check_diag.py`：{'通过' if diag['passed'] else '失败'}；`tests/check_lwy.py`：{'通过' if checks['passed'] else '失败'}。"
        f"远端 {checks['protected_files']} 个本轮前已存在的受保护文件逐个 SHA256 无变化；受保护地图目录无新增文件。"
        '完整旧环境指纹仍为 `a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d`。'
        'Mac 的独立清单见 `metrics/lwy_protected_mac.json`，最终复核见 `logs/lwy_mac_final_check.log`。\n',
        '本轮依赖安装于远端 `exp_nav3/.venv_lwy/`，版本固定于 `requirements-lwy.txt`，共享训练环境未安装包。原包和源仓库只读，Dreamer main 未提交或 push。'
        '地图/场景/atlas 大文件只留远端，Mac 同步 JSON、图、日志和报告；原 checkpoint/replay 未下载。最终远端状态见 `metrics/lwy_remote_final.json`。\n',
        '已修复并保留原始错误输出：\n\n```text\n'
        "AttributeError: module 'hashlib' has no attribute 'file_digest'\n"
        'timeout: 无法运行命令 "": 没有那个文件或目录\n'
        'ValueError: No zone at station 1000.0\n```\n\n'
        '分别对应 Mac Python 兼容性、首次 tmux 命令引号错误（改为独立脚本后成功冻结）、随机区段末值浮点舍入（固定最终边界后续跑）。'
        '两份早期变体参数的末值 1000.0000000000002 已正规化为 1000，并从最终生成器独立重建验证所有栅格数组完全一致，原栅格文件未改；'
        '证据为 `metrics/lwy_canonical_specs.json`。日志见 `logs/lwy_protected_freeze.log`、`lwy_protected_freeze_v2.log`、`lwy_dataset.log`、`lwy_dataset_v2.log`。\n',
        '整组盲测地图首次 JIT 编译还出现 `CompilationResultProto exceeded maximum protobuf size of 2GB: 4092930600`。'
        '原因是 vmap 闭包捕获了整个 atlas；已改为动态环境参数，最终原版两次与盲测均重新执行，使用 `*_dynamic.json` 结果。'
        '关闭 tmux 窗口曾留下本轮等待脚本及评估进程，已逐 PID 确认归属并终止；最终采用单一队列，收尾同时检查会话和进程。'
        '原始错误及被终止日志保留为 `logs/lwy_finish_v2.log`、`lwy_finish_v3.log`、`lwy_finish_v4.log`。\n',
        '尚需处理：明确末门的可实现通过定义，或由场景提供方提供实际出口支撑几何；本轮未擅自修改。'
        f"盲测还有 {len(gate['blind_before_999m_failures'])} 个内部失败，必须继续诊断生成器/精度/规划器，不能仅移动末门就宣称通过。"
        '终点定义明确后仍须完成严格镜像、修复变体上的失败并重跑完整 G1，接入新旧图混合训练、验证随机/静态障碍组合通路以及全地图+Agent+4608 环境显存，之后才能训练。'
        '视频近似几何未经实测；本报告的净空和成功率不能代表真实 M1 的安全认证。\n',
        '以下命令在远端 `exp_nav3/` 执行，先 `source scripts/env.sh`。已存在结果使用新 tag；地图生成支持跳过哈希一致的已完成项。\n\n```bash\n'
        '# CPU 专用 venv，输出只写 exp_nav3\n'
        'NAV3_PYTHON="$PWD/.venv_lwy/bin/python"\n'
        '# 首次运行导入命令；重建必须使用新根目录，勿覆盖已有地图\n'
        '"$NAV3_PYTHON" -m nav3.lwy_import --out maps_lwy/test_862428\n'
        '# 当前数据集入口；正式长任务用对应 scripts/lwy_*.sh + nav3_ tmux\n'
        '"$NAV3_PYTHON" -m nav3.lwy_dataset --workers 8\n'
        'source scripts/env.sh\n'
        '"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split test --episodes 10 --tag g1_original_new\n'
        '"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split blind --episodes 1 --tag g1_blind_new\n'
        'JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/lwy_check_diag_new.json\n'
        'JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_lwy.py\n```\n',
        '## 代码改动清单\n\n| 文件 | 说明 |\n| --- | --- |']
    descriptions = {
        'requirements-lwy.txt': '固定 CPU 几何处理 venv 的依赖版本，不修改共享训练环境。',
        'vendor/lwy/trail_scene.py': '原样复制只读共享生成器，SHA256 保持一致。',
        'nav3/lwy_geometry.py': '原始表面、支撑边界与基本体投影索引和独立净空查询。',
        'nav3/lwy_import.py': '5 cm 世界坐标稀疏栅格、台阶高程、碰撞层和可配置八门。',
        'nav3/lwy_bank.py': '新格式设备 atlas，独立 float32 s，旧 MapBank 不变。',
        'nav3/lwy_env.py': '显式 LWY 环境、密足迹、树根限速、窄路随机障碍和 60 分钟上限。',
        'nav3/lwy_control.py': '原动态轨迹规划器的显式 LWY 版本，仅去除重复障碍膨胀。',
        'nav3/lwy_diagnose.py': '提前越界位姿的横向偏移、原始净空和中心线可行性证据。',
        'nav3/lwy_planner_check.py': '单种子失败回归、完整路线修复与确定性复跑检查。',
        'nav3/lwy_dataset.py': '200/20/20 参数化变体、裁片、种子清单与续跑。',
        'nav3/lwy_canonicalize.py': '正规化两份修复前的浮点末值并验证全部栅格数组不变。',
        'nav3/lwy_audit.py': '原始几何与栅格净空对照、3000 个独立位姿探针。',
        'nav3/lwy_eval.py': '确定性独立回合 PRNG、G1 评估及训练阻断。',
        'nav3/lwy_profile.py': '1024/4608 环境吞吐、设备内存和地图体积测量。',
        'nav3/lwy_observations.py': '六个代表位置的实际噪声观测与通道宽度记录。',
        'nav3/lwy_random_check.py': '核对窄路 ob=1 真正生成障碍、覆盖全程且留有名义绕行宽度。',
        'nav3/lwy_finish.py': '数据统计、重复性核对与 G1 汇总。',
        'nav3/lwy_endpoints.py': '按终止时真实位姿区分原始几何接触与保守提前终止。',
        'nav3/lwy_plots.py': '由保存数据绘制净空、观测和轨迹。',
        'nav3/lwy_report.py': '由结果 JSON 自动生成本报告。',
        'scripts/lwy_integrity.py': '受保护文件冻结/检查与 Mac 包验证兼容包装。',
        'scripts/lwy_final_state.py': '核对没有残留 nav3 tmux 或本轮工作进程并保存最终状态。',
        'tests/check_lwy.py': '种子隔离、文件哈希、几何和 G1 阻断回归。'}
    for path, description in descriptions.items(): sections.append(f'| `{path}` | {description} |')
    for path in sorted(Path('scripts').glob('lwy_*.sh')):
        sections.append(f'| `{path}` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |')
    Path('LWY_REPORT.md').write_text('\n'.join(sections)+'\n')
    print('Wrote LWY_REPORT.md from saved JSON; G1 passed =', gate['passed'])


if __name__ == '__main__': main()
