> **第六轮续2最新状态**：complete；G1通过；有效最终评估 80/80。见[最新结果](#lwy-runtime2)。以下历史记录以最新协议修正为准。

# nav3 第六轮：LWY 赛道接入与 G1 阻断报告

**本轮未达成重新训练目标，G1 不通过，未启动任何 RL 冒烟或训练。** 原版 1 km 为 0/10 全通过，平均 7.00/8 门；盲测为 0/20 全通过，平均 6.30/8 门。原版的 10 个回合均通过前七门，失败集中在路面终点。末门 s=1000 m 与支撑路面终点重合，现有门状态机要求机器人中心越过门，有限尺寸包络无法在此保持完整支撑。没有移动门、补虚拟路面、缩小包络或忽略终止来换取分数。

盲测另有 11 个回合在 s<999 m 失败，详见下文清单；不把这些失败归到终点问题。

## 导入与几何判定

Mac、远端完整包均通过 `tools/verify_package.py`：137 个文件、642,046,794 字节。Mac Python 3.9 缺少 `hashlib.file_digest`，使用 `scripts/lwy_integrity.py package` 提供流式哈希兼容包装后重跑同一校验脚本；包本身未改。源码提交为 `83758d1dbcdfbd9ddbd2466ce8397f40cb927fce`；nav3 内复制的 `vendor/lwy/trail_scene.py` 与两处源文件 SHA256 一致，见 `metrics/lwy_sources.json`。

原版直接读取只读包的 `scene_bundle.json` 与 `track_spec.json`；变体调用复制的 `Route`、`build_bundle`，只向 `maps_lwy/` 写入。支撑区取 road 棱柱顶面及台阶踏面，不含路肩、坡面或植被带。相邻面片由上游 ±10 µm 取样造成的微缝以 60 µm 内的闭合运算拼接；不扩展真实起终点或侧边界。地面高程来自原始三角面平面插值。台阶保留离散高程，原有陡坡/台阶 0.28 m/s 限速不变。

护栏立柱/两层横杆、侧墙、石块、树干和标牌均用基本体 XY 投影碰撞，包括顶高不足 22 cm 的石块；不把整座山或树冠当可支撑道路。这是保守的 2.5D 包络判断，不是腿部接触或攀阶能力验证。半径 2.5 cm 的树根保留雷达高度返回，不作硬障碍，接近时限速 0.28 m/s；依据是任务指定的 M1 轮足通常可压越该尺度树根，尚无底层实测标定。

世界 XY 以 0.05 m 栅格、32×32 单元分块，仅保留距路线约 6 m 以内相交的整块；BEV 仍按真实世界位姿裁剪和旋转，无路线坐标拉伸。原版共 97 MiB 设备地图数据，5,535 块。沿程 s 另存 float32，避免 1 km 处 float16 约 0.5 m 的量化步长影响投影。新 `LwyBank`/`LwyEnv` 是显式入口，原 `MapBank`、`FastEnv`、控制器和配置文件均保持逐字节不变。

足迹仍为 **1.03×0.58 m**，20 Hz 判定由原 5×5 点改为 43×25 点。栅格量化半对角线 3.536 cm、足迹采样覆盖界 1.722 cm 与 0.02 cm 数值余量合成 5.277 cm 保守距离；支撑层另补偿旧 2.5 cm 终止阈值，防止旧容差变成越界许可。碰撞阈值与包络均未放宽。

下表使用每 0.2 m 的 5,001 个中心线位姿，独立矩形多边形对原始面片/基本体投影核对；数据见 `metrics/lwy_geometry.json`、`lwy_clearance_samples.npz`。负支撑净空表示包络已越出道路，端点数值不能与内部路段混为一谈。

| 原始几何净空（m） | 最小 | P01 | P05 | 中位数 | P95 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 支撑边界 | -0.5150 | 0.2288 | 0.2318 | 0.2678 | 0.3329 | 0.3350 |
| 碰撞体 | 0.2073 | 0.2150 | 0.2660 | 2.0285 | 4.2394 | 5.8522 |

内部 s=1–999 m 最小综合净空 **20.73 cm**。小于 5 cm 的中心线采样区间仅为 [[0.0, 0.4], [999.6, 1000.0]] m；其中起点由默认 s=2 m 出生规避，终点属于 `final_steps_and_moss` 区段，无法规避又同时通过末门。未发现原版内部中心线几何不可行段。

| 相对原始几何的额外保守量（cm） | 最小 | 中位数 | P95 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| 支撑判定 | 4.39 | 7.98 | 8.50 | 8.75 |
| 碰撞判定 | 2.33 | 6.64 | 8.16 | 8.76 |

另做 3000 个随机 s/横移/偏航位姿：原始几何可行 800 个，新判定可行 507 个；不安全误放行 0，保守误拒 293。这不是零风险证明；新判定仍有约 8.8 cm 的最坏额外保守量，并未声称精确物理接触。

![净空与终点放大](figures/lwy_clearance.png)

## 分辨率、吞吐与观测规格

| 条件 | 环境数 | 环境步/秒 | 峰值在用设备内存 GiB | 地图 MiB |
| --- | ---: | ---: | ---: | ---: |
| 旧 mini-trail 快环境 | 1024 | 66,150 | 2.516 | 6.75 |
| LWY 原版 1 km | 1024 | 59,378 | 2.614 | 97.54 |
| LWY 原版 1 km | 4608 | 53,626 | 10.412 | 97.54 |

1024 环境吞吐下降 10.2%，仍超过 20,000 环境步/秒目标。测量使用同一确定性 XLA 设置、预热后 100 步；旧/新地图长度不同，因此不是仅改变栅格分辨率的严格消融。数值不含 Agent、优化器和 replay。4608 环境单图的 JAX 内存池还曾扩展至 18.25 GiB，与在用峰值不是同一指标。全部新训练裁片栅格约 4.90 GiB，旧 200 图 atlas 约 1.33 GiB；全混合地图、4608 环境和 Dreamer 的联合显存尚未验证，不能据单图 profiling 声称原配置一定装得下。

观测维持 64×64×6、0.125 m/格、13 维向量，动作维持 3 维局部目标/速度档。1.1 m 路宽为 8.8 格，每侧包络余量约 2.08 格；六个实测位姿的车体横截面走廊宽度为 [9, 9, 8, 9, 8, 9] 格。路线通道能表示中心通路，原规划器也完成原版前七门；但 5.4 cm 直径立柱小于一个观测格，且特权腐蚀图在窄路上可能只剩 1–2 格。这不足以证明学习策略的边界识别充分。当前无证据要求强制更换规格，因此未改；后续若验证出观测瓶颈，可用 128×128、0.0625 m/格保持 8×8 m 视野，图像/原始 replay 图像字节量增加 4 倍，前端卷积开销增加，Dreamer/PPO 都需从头训练。

双 Airy 半球/遮挡、丢点、漂移、时延与累积流程完全继承旧环境。仅地图精度和足迹判定升级；旧 `priv_trav` 腐蚀近似仍保留，不能称为精确接触标签。

植物视觉素材没有逐网格体素化进入遮挡；树干使用 bundle 的碰撞胶囊体。尚未对 LWY 做全分辨率双 Airy 射线对照，因此保留旧噪声流程不等于已经完成新场景的传感器真实性标定。

![真实噪声观测与特权图](figures/lwy_observations.png)

## 数据划分与种子

| 集合 | 数量 | 种子 | 每张用于快环境的长度 |
| --- | ---: | --- | --- |
| train | 200 | 610000–610199 | 250 m 裁片（完整源变体 1 km） |
| val | 20 | 620000–620019 | 完整 1 km |
| blind | 20 | 630000–630019 | 完整 1 km |
| test | 1 | 862428–862428 | 完整 1 km |

训练集 106 张对路线、坡面和护栏侧别做参数层左右镜像；200 种区段排列，宽度 1.050–1.350 m（均不低于 minimum_width_m=1.05）；台阶组数 [3, 5]、实算级高 12.00–15.99 cm。控制点横移/曲率、区段顺序/长度、台阶数量/位置/高度、护栏侧别、石块/树根密度与位置、植被 seed 均有确定性随机化。训练裁片选择 s=0、249、498、747 m 中一个起点，长 250 m；八门按裁片本地里程为 31.25、62.5、…、250 m。这样避免把原始终点缺少出口的任务加入训练裁片；完整验证/盲测/测试地图的八门仍是 125、250、…、1000 m。

镜像仍有未完成项：上游台阶侧墙固定在路线右侧，参数层镜像没有将所有基本体整体反射，因此不是严格的全场景镜像。后续应统一反射几何并重建整个划分，而非只修改失败的盲测地图。当前数据集不能宣称满足全部几何增广验收。

清单为 `metrics/lwy_split_manifest.json`，远端完整生成物在 `maps_lwy/`。计划按回合 75% LWY、25% 旧 `maps/train` 混合，旧图种子 10000–10199；该比例只是清单中的训练方案，**没有实际训练采样，也尚未接入混合训练器**。原版 862428、验证、盲测、mini-trail 871002 和诊断种子 30000–30003 均不在训练清单中。原版/盲测仅执行任务明确要求的 G1 资格测试，没有参与模型选择；验证集没有被用来代替盲测。

## G1、重复性与停止依据

LWY 规划器只改一项：将已观测障碍的 3×3 max-pool 膨胀改为恒等映射，移除额外一格 12.5 cm 的重复余量；原规划器文件不动。采样偏移范围 ±0.84 m、15 条候选、1.11×0.66 m 预测足迹、动力学、风险/边界权重和速度均保留；物理碰撞判定和 1.03×0.58 m 包络未放宽。新环境将 max_steps 从旧 6000 明确设为 18000（60 分钟），否则旧 20 分钟上限与 1 km 慢速路线不匹配。门固定为 125、250、375、500、625、750、875、1000 m，导入 CLI 可指定八个门；本次使用原默认。

旧规划器完整十回合有 3 次在 s=909.416、923.433、948.141 m 提前终止；当时车体向右偏约 14.2–14.6 cm，原始几何支撑净空仍有 7.85–8.51 cm，中心线净空则有 20.76–23.45 cm。可见是偏离中心后的保守提前终止，不是这三处完全无路可走。独立原始几何与终止帧栅格核对见 `metrics/lwy_planner_diagnosis.json`；终止标志会累计四个 50 ms 子步，最终帧的栅格状态不必等于首次命中子步。

固定单回合 seed 940004、s=900 m 出生的回归：旧规划器在 948.088318 m 失败并触发断言；只取消规划器重复膨胀后到达 999.424194 m。完整十回合提前失败变为 0，另起进程重复后全部动作/位姿哈希相同。该单变量干预支持重复膨胀导致贴边的解释；未改定位噪声，不把噪声假设当作已独立穷尽验证。余下终点失败保留原样，绝不算通过。诊断/回归日志见 `logs/lwy_diag_single_legacy.log`、`lwy_diag_single_lwy.log`、`lwy_planner_fix.log`。

| G1 集合 | N | 全通过 | Wilson 95% | 平均门 | s≥999 m 且已过七门 | 失败分类 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 原版 1 km | 10 | 0.0% | 0.0%–27.8% | 7.00 | 10 | {'support_or_drop': 10} |
| 盲测变体 | 20 | 0.0% | 0.0%–16.1% | 6.30 | 9 | {'collision': 2, 'support_or_drop': 18} |

原版失败 s 范围 999.4146–999.4278 m。逐回合种子、动作/位姿 SHA256、轨迹、速度和终止位置均保存在对应 G1 JSON。

盲测的内部失败逐项如下。净空来自终止帧的原始几何核对，中心线可行仅表示该位置有可行姿态，不能证明整段动态可通过。负横向偏移表示路线右侧；物理接触为否时，应进一步区分栅格保守量与累计子步终止。

| 地图种子 | 失败 s（m） | 分类 | 实际横移（cm） | 中心线净空（cm） | 中心线可行 | 终止帧原始几何接触 |
| --- | ---: | --- | ---: | ---: | --- | --- |
| 630001 | 982.190 | support_or_drop | 14.34 | 30.33 | 是 | 否 |
| 630006 | 982.083 | support_or_drop | 16.26 | 1.61 | 是 | 否 |
| 630007 | 875.905 | support_or_drop | 13.37 | 0.00 | 否 | 否 |
| 630008 | 986.604 | support_or_drop | 10.22 | 20.76 | 是 | 否 |
| 630011 | 985.588 | support_or_drop | 16.75 | 22.62 | 是 | 否 |
| 630013 | 980.130 | support_or_drop | 13.76 | 26.21 | 是 | 否 |
| 630014 | 979.836 | collision | -7.59 | 5.69 | 是 | 否 |
| 630016 | 980.042 | support_or_drop | 10.49 | 19.88 | 是 | 否 |
| 630017 | 17.423 | collision | 0.65 | 9.51 | 是 | 否 |
| 630018 | 20.362 | support_or_drop | -12.38 | 17.57 | 是 | 否 |
| 630019 | 981.606 | support_or_drop | 11.24 | 26.20 | 是 | 否 |

变体 630007 在 s=875.905 m 的中心线姿态已与障碍投影相交，630006 在失败位置的中心线净空仅 1.61 cm；这说明原版中心线的 20.73 cm 最小净空不能外推到全部变体。两处实际偏移后的终止姿态仍可行，因此这里未证明整条横断面都不可通过。

对原始几何再次核对终止时完整包络，30/30 个终止位姿尚未发生原始几何接触（见 `metrics/lwy_endpoint_geometry.json`）。这与保守提前终止相符，不能直接称为已经物理跌落；首次触发的子步仍需进一步核对。即使消除全部保守量，有限包络也不能让中心越过真实路面末端。

确定性设置为 `--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true`、`TF_CUDNN_DETERMINISTIC=1`；每回合独立 PRNG seed，通过 vmap(single environment) 执行，避免批次大小改变噪声流。原版整套十回合另起进程重复，全部逐回合字段及每步动作/位姿哈希完全一致。这与旧评估器按批形状抽随机数的协议不同，后续方法比较必须统一采用本轮协议。

![规划器原版轨迹](figures/lwy_trajectories.png)

## 训练、模型选择与最终评估状态

G1 的原版条件要求 10/10，盲测要求至少 95%；本轮失败，因此 `require_g1()` 明确拒绝继续。未复制 `dreamer_lwy`/`ppo_lwy`，未运行 ≤15 min 冒烟、各 4 h 续训、30 min 快照/验证或学习曲线，也没有可选最终模型。规格未改，若以后 G1 解锁，按任务应从原始 `dreamer_s0`/`ppo_s0` 复制续训；不能拿第五轮已崩溃的快照初始化。届时模型选择应仅按 20 张验证图×2 回合的全通过率，平局按平均门数；连续两次低于本次最佳超过 30 个百分点则停止。

第 3.5 节崩溃诊断未触发，因为本轮没有续训；第五轮的崩溃原因仍未解决。没有把本轮 G1 失败归因于 RL。

| 要求的方法 | 本轮最终评估状态 |
| --- | --- |
| dreamer_lwy / ppo_lwy | 未创建，G1 阻断训练 |
| dreamer_s0 / ppo_s0 零样本 | 未执行规定的最终矩阵，G1 尚不成立 |
| 规则规划器 | 仅完成上表 G1，不等于最终 100 回合/条件评估 |

五方法×四数据集×四条件的最终比较、模型可通行 IoU、接管率比较、mini-trail 退化检查和 Dreamer/PPO MuJoCo 推理视频均未执行。没有学习曲线或视频路径可交付，不以旧结果或规划器录像替代新模型结果。随机障碍最终评估未执行。已单独修正 LWY 的随机障碍放置入口：旧规则要求路宽>1.7 m，会让所有新图的 ob=1 变成空场景；新入口把三个障碍分布在整条路线的三个分离区间，靠一侧留 3 cm 边界，按路宽采样 12–24 cm 内可行半尺寸，另一侧留出 0.58 m 包络加 10 cm 名义通路；横向倒木放不下时使用石块形状。此规则不改静态几何和包络，也不保证与所有静态物体组合后全局可通过。不开随机障碍时走原分支，G1 的噪声和动作复现不受影响。启用性检查见 `metrics/lwy_random_obstacles.json`。

## 完整性、问题与复现

`tests/check_diag.py`：通过；`tests/check_lwy.py`：通过。远端 3811 个本轮前已存在的受保护文件逐个 SHA256 无变化；受保护地图目录无新增文件。完整旧环境指纹仍为 `a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d`。Mac 的独立清单见 `metrics/lwy_protected_mac.json`，最终复核见 `logs/lwy_mac_final_check.log`。

本轮依赖安装于远端 `exp_nav3/.venv_lwy/`，版本固定于 `requirements-lwy.txt`，共享训练环境未安装包。原包和源仓库只读，Dreamer main 未提交或 push。地图/场景/atlas 大文件只留远端，Mac 同步 JSON、图、日志和报告；原 checkpoint/replay 未下载。最终远端状态见 `metrics/lwy_remote_final.json`。

已修复并保留原始错误输出：

```text
AttributeError: module 'hashlib' has no attribute 'file_digest'
timeout: 无法运行命令 "": 没有那个文件或目录
ValueError: No zone at station 1000.0
```

分别对应 Mac Python 兼容性、首次 tmux 命令引号错误（改为独立脚本后成功冻结）、随机区段末值浮点舍入（固定最终边界后续跑）。两份早期变体参数的末值 1000.0000000000002 已正规化为 1000，并从最终生成器独立重建验证所有栅格数组完全一致，原栅格文件未改；证据为 `metrics/lwy_canonical_specs.json`。日志见 `logs/lwy_protected_freeze.log`、`lwy_protected_freeze_v2.log`、`lwy_dataset.log`、`lwy_dataset_v2.log`。

整组盲测地图首次 JIT 编译还出现 `CompilationResultProto exceeded maximum protobuf size of 2GB: 4092930600`。原因是 vmap 闭包捕获了整个 atlas；已改为动态环境参数，最终原版两次与盲测均重新执行，使用 `*_dynamic.json` 结果。关闭 tmux 窗口曾留下本轮等待脚本及评估进程，已逐 PID 确认归属并终止；最终采用单一队列，收尾同时检查会话和进程。原始错误及被终止日志保留为 `logs/lwy_finish_v2.log`、`lwy_finish_v3.log`、`lwy_finish_v4.log`。

尚需处理：明确末门的可实现通过定义，或由场景提供方提供实际出口支撑几何；本轮未擅自修改。盲测还有 11 个内部失败，必须继续诊断生成器/精度/规划器，不能仅移动末门就宣称通过。终点定义明确后仍须完成严格镜像、修复变体上的失败并重跑完整 G1，接入新旧图混合训练、验证随机/静态障碍组合通路以及全地图+Agent+4608 环境显存，之后才能训练。视频近似几何未经实测；本报告的净空和成功率不能代表真实 M1 的安全认证。

以下命令在远端 `exp_nav3/` 执行，先 `source scripts/env.sh`。已存在结果使用新 tag；地图生成支持跳过哈希一致的已完成项。

```bash
# CPU 专用 venv，输出只写 exp_nav3
NAV3_PYTHON="$PWD/.venv_lwy/bin/python"
# 首次运行导入命令；重建必须使用新根目录，勿覆盖已有地图
"$NAV3_PYTHON" -m nav3.lwy_import --out maps_lwy/test_862428
# 当前数据集入口；正式长任务用对应 scripts/lwy_*.sh + nav3_ tmux
"$NAV3_PYTHON" -m nav3.lwy_dataset --workers 8
source scripts/env.sh
"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split test --episodes 10 --tag g1_original_new
"$NAV3_PYTHON" -m nav3.lwy_eval --planner lwy --split blind --episodes 1 --tag g1_blind_new
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/lwy_check_diag_new.json
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_lwy.py
```

## 代码改动清单

| 文件 | 说明 |
| --- | --- |
| `requirements-lwy.txt` | 固定 CPU 几何处理 venv 的依赖版本，不修改共享训练环境。 |
| `vendor/lwy/trail_scene.py` | 原样复制只读共享生成器，SHA256 保持一致。 |
| `nav3/lwy_geometry.py` | 原始表面、支撑边界与基本体投影索引和独立净空查询。 |
| `nav3/lwy_import.py` | 5 cm 世界坐标稀疏栅格、台阶高程、碰撞层和可配置八门。 |
| `nav3/lwy_bank.py` | 新格式设备 atlas，独立 float32 s，旧 MapBank 不变。 |
| `nav3/lwy_env.py` | 显式 LWY 环境、密足迹、树根限速、窄路随机障碍和 60 分钟上限。 |
| `nav3/lwy_control.py` | 原动态轨迹规划器的显式 LWY 版本，仅去除重复障碍膨胀。 |
| `nav3/lwy_diagnose.py` | 提前越界位姿的横向偏移、原始净空和中心线可行性证据。 |
| `nav3/lwy_planner_check.py` | 单种子失败回归、完整路线修复与确定性复跑检查。 |
| `nav3/lwy_dataset.py` | 200/20/20 参数化变体、裁片、种子清单与续跑。 |
| `nav3/lwy_canonicalize.py` | 正规化两份修复前的浮点末值并验证全部栅格数组不变。 |
| `nav3/lwy_audit.py` | 原始几何与栅格净空对照、3000 个独立位姿探针。 |
| `nav3/lwy_eval.py` | 确定性独立回合 PRNG、G1 评估及训练阻断。 |
| `nav3/lwy_profile.py` | 1024/4608 环境吞吐、设备内存和地图体积测量。 |
| `nav3/lwy_observations.py` | 六个代表位置的实际噪声观测与通道宽度记录。 |
| `nav3/lwy_random_check.py` | 核对窄路 ob=1 真正生成障碍、覆盖全程且留有名义绕行宽度。 |
| `nav3/lwy_finish.py` | 数据统计、重复性核对与 G1 汇总。 |
| `nav3/lwy_endpoints.py` | 按终止时真实位姿区分原始几何接触与保守提前终止。 |
| `nav3/lwy_plots.py` | 由保存数据绘制净空、观测和轨迹。 |
| `nav3/lwy_report.py` | 由结果 JSON 自动生成本报告。 |
| `scripts/lwy_integrity.py` | 受保护文件冻结/检查与 Mac 包验证兼容包装。 |
| `scripts/lwy_final_state.py` | 核对没有残留 nav3 tmux 或本轮工作进程并保存最终状态。 |
| `tests/check_lwy.py` | 种子隔离、文件哈希、几何和 G1 阻断回归。 |
| `scripts/lwy_audit.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_canonical.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_dataset.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_eval.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_final_queue.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_freeze.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_import.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_measure.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_planner_fix.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
| `scripts/lwy_profile.sh` | 独立可脱离的限时阶段脚本，日志留存；不调用训练。 |
<a id="lwy-v2-update"></a>
# LWY v2 续做结果

更新时间：2026-11-11T20:33:28.261215+00:00。

**G1：通过；训练前检查及短冒烟已完成；等待评估协议决定，正式训练未开始。** 本轮没有修改原版 1 km 几何、碰撞判定、1.03×0.58 m 包络或旧地图。

## 验收决定与实际改动

原来的第八门落在路面尽头，机器人中心越过门就一定出界。这是任务设定的问题，不是规划器的问题。完整路线前七门保持 125、250、375、500、625、750、875 m，第八门改为 998.5 m；通过即正常结束。`--terminal-offset` 默认 1.5 m，允许更大的后退量。训练裁片为 250 m，起点取 0/249/498/747 m；最末裁片结束于源路线 997 m，其后仍有 3 m 真路面，所以其末门保留裁片 s=250。没有源路面延续时，末门后退至少 1.5 m。

v2 对生成完成的整个世界几何施加 diag(1,−1,1)，不是只镜像控制点。所有 road/bank 碰撞棱柱、坡面网格、护栏、台阶侧墙、石块、树根、树干、标牌一起反射，三角面绕序同时翻转；植物实例同时反射位置、偏航与局部 Y 手性，以支持非对称素材。原 `maps_lwy/` 保持不动。

| 改动 | 默认 / v2 采用 | 独立证据 |
| --- | --- | --- |
| 末门位置 | 旧导入器默认行为不变；v2 L−1.5 m | 只改门，原版 gain=2 已 10/10；gain=4 同为 10/10 |
| 跟踪器航向增益 | 旧行为 2；v2 配置 4 | 旧图2→998 m完整历史诊断 9/20→12/20，改善4图、退化1图；原版无退化 |
| 规划器预测时域 | 默认70×0.1 s；v2配置30×0.1 s | 同一失败现场 0/2→2/2；新盲测完整18/20→20/20 |
| 候选数量、前视、速度预测、边界权重 | 15、1.2 m、0.5 m/s、300 保持 | 未改善的单变量设置不采用 |
| 几何筛选 | 0.2 m，最小原始净空0.10 m，可配置 | 原始多边形检查；阻挡石块固定规则重采样，仍不合格则丢弃种子 |
| 判定、限速、观测、奖励 | 不改 | 旧环境指纹、默认跟踪哈希、包络回归和保护清单核对 |

## s≈980 m 专项诊断

先用旧完整路线与完整观测历史复现：11 个内部失败的动作/位姿哈希与上一轮完全一致；四个 50 ms 物理子步的回放终点误差为 0。然后对题述8图核查原始几何与首次失败足迹。[lwy_v2_diagnosis_summary.json](metrics/lwy_v2_diagnosis_summary.json)。

| 种子 | 失败 s(m) | 几何 | 路宽(m) | 区段边界距离(m) | 有符号横移(cm) |
| --- | --- | --- | --- | --- | --- |
| 630001 | 982.190 | final_descent | 1.197 | 113.41 | +14.34 |
| 630006 | 982.083 | final_descent | 1.188 | 8.27 | +16.26 |
| 630008 | 986.604 | final_descent | 1.050 | 100.01 | +10.22 |
| 630011 | 985.588 | final_descent | 1.148 | 80.99 | +16.75 |
| 630013 | 980.130 | final_descent | 1.117 | 119.80 | +13.76 |
| 630014 | 979.836 | final_descent | 1.268 | 163.52 | -7.59 |
| 630016 | 980.042 | final_descent | 1.050 | 100.64 | +10.49 |
| 630019 | 981.606 | final_descent | 1.169 | 14.04 | +11.24 |

这些位置是最后一段下行台阶的入口、出口或近邻弯折，邻域±2 m路宽不变；不是区段过渡或宽度突变。按路线法向 (−t_y,t_x)，正横移是左偏：8图中7图左偏，只有630014右偏，因此不能把原先的“全部右偏”当作已确认事实。台阶前后方向变化、跟踪延迟与保守边界共同压缩余量。

| 假设 | 实验 / 证据 | 结论 |
| --- | --- | --- |
| 台阶限速下跟踪误差 | 航向增益2→4；完整旧图9→12/20；记录cmd/vel/航向误差 | 有贡献，单独不足以解锁G1；0.28 m/s限速未改 |
| 定位漂移 | 完整旧图关闭漂移9→10/20，含退化图 | 漂移不是充分解释；正式设置保留原漂移 |
| 候选太稀 | 短段15→57个候选均12/20 | 没有证据支持单纯加密；未采用 |
| 预测时域过长 | 保留完全相同PRNG/记忆/延迟队列的现场；70→30步0/2→2/2 | 缩短时域减少远端预测误差；完整G1确认有效 |
| 支撑栅格在台阶处出错 | 首次坏子步对原始多边形复核，多个失败仍有约5–8 cm真净空 | 达到既有保守栅格界限；未发现应通过放宽判定修复的导入错误 |
| 侧墙碰撞层 | 630014首次碰撞足迹距原墙约4.70 cm；投影与原始侧墙一致 | 保守层确实触发，但不能删除侧墙或降低包络；不合格变体由几何筛选丢弃 |

旧图冷启动970 m短段结果：基线12/20；候选57为12；前视0.6 m为11；预测速度0.28 m/s为12；无漂移11；扩大规划预测余量11；航向增益4为13、6为12。短段分数不替代完整G1。[lwy_v2_parameter_regressions.png](figures/lwy_v2_parameter_regressions.png)。

首轮新G1剩余两处为630005的703.667 m与630021的985.179 m，均为支撑失败。现场恢复基线0/2，单独恢复增益2为1/2，提高边界权重300→1000为0/2，仅时域70→30为2/2；两处首次坏足迹的原始支撑净空分别6.93、5.93 cm。[lwy_v2_round0_replay_baseline_analysis.json](metrics/lwy_v2_round0_replay_baseline_analysis.json)、[lwy_v2_round0_replay_horizon30_analysis.json](metrics/lwy_v2_round0_replay_horizon30_analysis.json)。

## 导入、净空与数据划分

原版的0.05 m世界坐标分块栅格、43×25足迹、5.277 cm保守距离与几何误差量化仍见上轮记录；原始内部中心线最小净空20.73 cm，3000探针不安全误放行0、保守误拒293。v2原版地图除门数组外逐数组与旧原版一致。BEV保持64×64×6、0.125 m/格，状态13维、动作3维，旧权重可续训；此次没有足够证据要求修改观测规格。

变体可行性审计检查可执行中心域 s=1.5…998.5 m，步长0.2 m并包含末点，使用完整1.03×0.58 m矩形。两端1.5 m端帽明确排除：有限包络不可能以中心位于真实端点，且出生在2 m、终门在998.5 m。该约定写入每张图的元数据；不能把端帽排除隐称为全闭区间通过。

版本 `lwy-v2.1`；训练200、验证20、盲测20、原版测试1。丢弃候选 **258** 个，累计重采样石块 **0** 次；接受变体最小净空 **10.007 cm**，中位数 **15.106 cm**。筛选只读取原始几何，未用规划器成绩筛种子。完整接受、拒绝和重采样清单：[lwy_v2_split_manifest.json](metrics/lwy_v2_split_manifest.json)。

种子从610000/620000/630000分别连续递增，接受各自首批满足净空的200/20/20图；不跨各自10000种子带。原版862428、mini-trail871002、合成30000–30003及旧留出集均不进入训练。旧训练种子10000–10199。

| 划分 | 接受种子 |
| --- | --- |
| train | 610002, 610006, 610008, 610009, 610010, 610014, 610018, 610020, 610022, 610023, 610026, 610031, 610035, 610036, 610038, 610042, 610043, 610044, 610045, 610049, 610052, 610053, 610054, 610055, 610058, 610060, 610062, 610063, 610064, 610065, 610066, 610068, 610072, 610075, 610077, 610079, 610081, 610082, 610083, 610085, 610088, 610089, 610090, 610094, 610096, 610105, 610106, 610109, 610111, 610112, 610113, 610114, 610121, 610123, 610124, 610126, 610127, 610132, 610134, 610135, 610137, 610138, 610141, 610142, 610145, 610146, 610147, 610148, 610152, 610157, 610159, 610160, 610163, 610165, 610169, 610171, 610172, 610174, 610176, 610178, 610179, 610180, 610181, 610185, 610186, 610188, 610189, 610190, 610191, 610192, 610193, 610194, 610195, 610197, 610199, 610200, 610202, 610203, 610204, 610208, 610210, 610212, 610214, 610216, 610218, 610219, 610220, 610223, 610224, 610225, 610230, 610232, 610233, 610235, 610236, 610238, 610240, 610243, 610244, 610245, 610247, 610249, 610250, 610252, 610253, 610257, 610259, 610260, 610261, 610263, 610265, 610266, 610267, 610269, 610272, 610273, 610274, 610275, 610277, 610278, 610279, 610280, 610282, 610284, 610285, 610286, 610287, 610288, 610293, 610294, 610296, 610297, 610298, 610307, 610313, 610315, 610317, 610319, 610325, 610328, 610329, 610330, 610331, 610333, 610334, 610335, 610337, 610342, 610343, 610346, 610349, 610350, 610351, 610352, 610353, 610355, 610357, 610359, 610363, 610368, 610369, 610374, 610375, 610377, 610381, 610382, 610384, 610390, 610392, 610393, 610395, 610397, 610399, 610401, 610403, 610404, 610407, 610409, 610411, 610412 |
| val | 620000, 620002, 620005, 620008, 620010, 620012, 620013, 620016, 620017, 620021, 620022, 620023, 620026, 620027, 620030, 620031, 620032, 620035, 620036, 620037 |
| blind | 630000, 630001, 630002, 630003, 630004, 630005, 630009, 630010, 630012, 630015, 630021, 630023, 630027, 630029, 630031, 630037, 630039, 630044, 630045, 630046 |
| test | 862428 |

| 拒绝种子 | 最小净空(m) | 不合格采样数 | 阻挡项示例 |
| --- | --- | --- | --- |
| 630006 | 0.0 | 10 | rail_bar_9_1_5_0, rail_bar_9_1_5_1, stair_wall_1_0 |
| 630007 | 0.0 | 9 | rail_bar_8_-1_54_0, rail_bar_8_-1_54_1, stair_wall_1_15 |

检查中−1 m是“矩形未完全被支撑域覆盖”的哨兵值，不代表实际穿出1 m。固定侧墙、护栏及支撑不足不通过移动道路修复；只允许固定规则重采样阻挡石块。草稿目录 `maps_lwy_v2_stride_draft/` 与中断候选不在最终清单中；最终集合按连续种子规则生成。

## G1 重跑

| 轮次 | 原版 | 盲测 | 结论 |
| --- | --- | --- | --- |
| v2_round0 | 10/10 | 18/20 | 未通过 |
| v2_round1 | 10/10 | 20/20 | 通过 |

首轮18/20后只使用了一轮额外完整修复迭代，达到原版10/10与盲测≥95%的双重条件。没有降低阈值。G1先在指向最终20张盲测与原版的只读资格视图执行，训练入口还要求最终训练/验证清单齐全并匹配G1哈希。[lwy_v2_g1.json](metrics/lwy_v2_g1.json)。

## 评估协议与训练前检查

当前正式比较配置为XLA autotune=0、deterministic_ops=true，TF_CUDNN_DETERMINISTIC=1；每回合独立噪声PRNG，按单环境vmap执行。策略另用噪声种子+1,000,000,000。最初发现卷积批量形状不同仍会导致数值差异，因此神经策略统一以100个独立流执行，不足100的部分仅补齐策略计算，不进入物理环境、统计或PRNG；旧的未固定批量探针保留为诊断资料，不混入最终对比。原level0批量一致性证据：[lwy_v2_policy_protocol.json](metrics/lwy_v2_policy_protocol.json)。

按每次回合重置独立抽取来源：LWY 75%、旧maps/train 25%，随后在对应训练集合中均匀抽图；并非把75%常驻worker永久绑到LWY。不同来源保留各自原物理判定。训练日志记录实际回合数与每图暴露次数。

随机障碍沿用原尺寸、侧向和站点提议分布，组合静态障碍后用0.2 m路线/0.04 m横移格点寻找通路，同时检查连接边的中点，再以0.1 m间距、完整原足迹和原碰撞距离复核通路。不通过即重采样。每图保留8组通过检查的三障碍布局，运行时随机选择；所有方法使用同一目录与协议。这是离散几何通路检查，不能证明动态跟踪器必然通过；随机场景多样性限于该目录。

训练前修复了两项实现问题：混合采样器在vmap内选择整个地图对象会广播大型图集，64环境曾请求95 GB设备内存；改为共享图集、只选择状态与观测后，7775次回合开始中LWY为5858、旧图1917（75.344%）。通路搜索原先只检查节点、未在搜索中约束连接中点，且额外原始几何筛选会拒绝原物理判定允许的绕行；现统一使用原物理判定并检查节点和中点，额外1 cm仅用于搜索见证。legacy/train_10001单例8组/10次提议、lwy/train_610043单例8组/277次提议通过；最终目录全部按v3检查器生成。这些改动没有放宽运行时碰撞判定。

固定100策略流还通过了1回合与最终100回合首轨迹逐位比较。[lwy_v2_policy_protocol_final_batch.json](metrics/lwy_v2_policy_protocol_final_batch.json)。物理步分组加速实验虽在100步状态、观测和额外信息上逐位一致，但仅为直接vmap速度的0.247倍，未采用；正式评估仍执行原单环境vmap。

200回合分层抽查（150 LWY+50旧图，每张图独立提议种子）：首提议未找到合格通路 **74/200=37.0%**，重采样后 **0/200**；整个目录拒绝组合5987次。该比例是保守离散搜索拒绝率，不等同于几何上绝对不可通过率。[lwy_v2_random_feasibility.json](metrics/lwy_v2_random_feasibility.json)。

| 方法 | 环境数 | 分配器 / 池上限 | 实测峰值MiB | 占总显存 | Agent+优化器完成 |
| --- | --- | --- | --- | --- | --- |
| dreamer | 4608 | platform / 不适用 | 19651 | 80.0% | True |
| ppo | 128 | default / 0.95 | 17717 | 72.1% | True |

实际选用环境数：{'dreamer': 4608, 'ppo': 128}。测量包括全部200张LWY裁片、200张旧图、真实Agent、采集、回放批次及优化器；没有用纯环境占用估算。[lwy_v2_memory_selection.json](metrics/lwy_v2_memory_selection.json)。

默认分配器下4608实测OOM，4096/3584/3072虽运行成功，保留内存池后仍占95.9%；2560–2944约72.3%。池上限0.86试验无法满足3072的连续分配请求，原始失败均保留。随后使用JAX已支持的platform分配器释放不再使用的分配，重新测量Dreamer；PPO与评估保留默认分配器。选中的训练配置与实测一致。表中峰值是整张24,564 MiB卡的实际占用，包含约911 MiB桌面基线；冒烟继续检查实际峰值。若原定4608已可完整运行且低于85%，保留原环境数，不为填满显存额外扩大训练配置。

## 训练、快照与模型选择

观测动作规格不变，因此从原始健康 `dreamer_s0`、`ppo_s0` 复制到新名字，保留优化器；没有从已崩溃的ft快照启动。Dreamer使用未改的 `embodied.run.train.train` 主循环，接入GPU混合集合Driver。保留原replay的100,000个序列起点及其依赖chunk，旧数据只读复制到新replay；记录实际抽到的新旧样本比例。PPO保留原配置。两者奖励一致，均不开安全过滤。

修复运行时后各做6分钟冒烟和5分钟恢复检查，实际作业合计须≤15分钟；此前失败初始化与异常数值尝试单独保留为诊断，不计作通过的冒烟，全部GPU时间仍计入12小时总预算。随后各计划8段30分钟正式训练。每段保存新快照并做验证20图×2回合，不开随机障碍/过滤。选择规则为验证全通过率最高，其次平均门数，完全相同取更早快照；原版、盲测、旧留出和mini不参与选择。连续两次低于此前最佳超过30个百分点即停止该方法，保留快照并生成只读崩溃诊断；不重启刷分。

Dreamer首次冒烟在0步/0次更新时因Chunk.saved标记不一致而失败。失败初始化被隔离到checkpoints/replay下的lwy_v2_failed_initialization，原始日志保留；修复为将旧chunk文件复制到新replay目录，同时保留两个保存标记。未更改上游replay源码。另先采集66步以形成新的65步序列，随后启用官方Ratio=32，避免最初长时间只抽到旧replay；保留旧样本并记录实际比例。初始化异常不再写出可能不完整的最终检查点。[lwy_v2_replay_initialization_failure.json](metrics/lwy_v2_replay_initialization_failure.json)。

检查点保存混合集合状态、策略状态、PRNG、replay和官方Ratio进度，段间恢复环境而非同步重置。训练采样预取队列遵循官方循环行为，未宣称跨进程训练位级重现。

恢复冒烟另发现NumPy直接保存bfloat16时将dtype记成|V2，导致JAX加载失败；失败发生在加载阶段，未执行新更新。LWY专用序列化改为保存uint16原始位并记录bfloat16叶子索引，兼容读取已保存的两片旧carry。实际721次更新检查点的43个叶子全部经CPU验证：原文件位模式完全一致，新格式往返后的dtype、形状及所有位均一致。未改旧通用检查点工具或原始权重。[lwy_v2_checkpoint_serialization.json](metrics/lwy_v2_checkpoint_serialization.json)。

另遵循本次续做的“验证成绩连续下降就诊断”要求：全通过率相邻两次都下降时，即使尚未触及上述30个百分点规则，也停止该条训练并做3.5诊断，不自动重启。原TASK的强制早停条件同时保留。

队列：`nav3/lwy_queue_v2.py`，状态：[lwy_v2_queue.json](metrics/lwy_v2_queue.json)；每个子作业带超时且短于3小时，预算累计并记录。出现代码异常、OOM、非有限损失或预算耗尽会明确失败并停止，不能把失败算完成。

累计GPU占用记账（含前置与外部诊断预留）约1.832小时，12小时预算尚余约10.168小时；完整正式训练及最终评估是否完成以队列状态为准。

最终评估最多并行两个独立进程，训练与显存测量保持单作业。两种旧模型分别在完整20图盲测上运行两个100回合的并行副本；400个完整轨迹哈希均与串行基线相同，同时检验两份Dreamer的显存。评估的每回合PRNG、单环境vmap与固定100策略流不变；GPU预算按这张卡被占用的墙钟并集累计，另保留每个进程耗时。[lwy_v2_parallel_protocol.json](metrics/lwy_v2_parallel_protocol.json)。

尚无正式训练曲线或已选模型；不会填入预期数字。

| 方法 | 纯训练冒烟完成 | 完整冒烟含验证 | 恢复前/后更新数 | 冒烟实际秒数 |
| --- | --- | --- | --- | --- |
| dreamer_lwy | True | False | 721 / 1873 | 788.8 |
| ppo_lwy | True | False | 15232 / 26624 | 666.9 |

## 训练运行时数值异常与单变量诊断

异常冒烟累计313,344步、289次更新，随后在验证阶段人工停止队列，未进入正式训练。健康原始权重与同一冻结旧replay批次，在autotune=0时首个损失5,585.917、原始梯度范数2.023×10¹⁵；唯一改动为恢复原训练autotune=1后，损失236.285、梯度830.305。7个返回指标的更新末尾损失229.797、梯度764.318。冻结新批次也由损失5,640.120/梯度5.127×10¹⁵恢复到2,318.631/9,786.664，并继续下降。[lwy_v2_training_numerics_diagnosis.json](metrics/lwy_v2_training_numerics_diagnosis.json)、[lwy_v2_training_numerics.png](figures/lwy_v2_training_numerics.png)。

这是把评估设置错误带入训练导致的运行时问题。采用Dreamer运行时声明的默认autotune=1；PPO训练恢复原来的库默认XLA设置。所有原始学习率、优化器、奖励及地图判定不变。异常新权重与replay隔离于`checkpoints/lwy_v2_autotune0_diagnostic/dreamer_lwy/`和`replay/lwy_v2_autotune0_diagnostic/dreamer_lwy/`，异常0.10h快照保留且不参与选模型。修复后的运行另用`_runtime2`快照及作业标识，从原始健康权重开始。显存也按修复后的训练运行时重新实测，旧测量保留。

历史兼容性证据其实已存在于[fix_numeric_dreamer_level0.json](metrics/fix_numeric_dreamer_level0.json)和[fix_numeric_dreamer_level1.json](metrics/fix_numeric_dreamer_level1.json)，第5轮正式`nav3/fix_eval.py`也已在JAX初始化前使用level1。上一轮LWY的level0只跑了规则规划器。本轮错误地将它推广到神经训练/推理，现用冻结批次再次复现并记录；不能把该问题归咎于地图或新的训练分布。

逐项排查：旧replay本身即可复现异常，因此新障碍惩罚不是该数值问题的原因；确实保留100,000个旧序列起点，并非全新replay。在线队列只优先最近完成的65步序列，冒烟结束时队列为0，不能据初始旧样本比例0就断言旧数据永久不参与。冻结旧批次在上下文后没有重置标记，仍出现异常，排除同步初始重置作为这一数值问题的解释。首个更新前损失和原始梯度已经不同，学习率不可能解释该差异；没有据此调小学习率。上述结论不排除后续导航行为仍受奖励和分布变化影响。

推理复核同样发现问题：固定100流、相同健康模型与观测，autotune=0的可通行预测整幅恒为0.48828125；autotune=1输出范围约0…1，平均绝对差0.48544，动作最大绝对差2.65569。故现有旧模型零样本0/100不能直接解释为泛化失败。[lwy_v2_forward_runtime_comparison.json](metrics/lwy_v2_forward_runtime_comparison.json)。用户明确要求沿用旧设置，已提出协议修正问题；在答复前暂停依赖该决定的训练验证及最终比较。

队列对外部数值诊断预留并计入900秒GPU占用，低于3.5节1小时上限。新训练增加原始梯度>10¹²立即停止保护，防止再次把有限但失真的数值当作正常损失。

## 崩溃诊断状态

尚未触发本轮崩溃诊断。旧ft崩溃快照与原有诊断材料保持只读。触发后报告损失、熵/方差、回报尺度、价值损失、梯度范数、replay旧样本比例、重置与失败动作/位置；没有干预证据时不声称唯一根因。

## 最终比较与视频

level1候选协议试验已完成，是否正式采用以用户决定及配置为准。两种旧模型各3个独立进程、每个100回合，合计600个完整轨迹哈希逐位一致；单回合与100回合的首轨迹也一致。规划器资格复核原版10/10、盲测20/20，全部30条轨迹与level0逐位相同，未调整规划器或几何。[lwy_v2_runtime_candidate_protocol.json](metrics/lwy_v2_runtime_candidate_protocol.json)。候选协议试验基于数值错误诊断，不以测试成绩调参；加上外部诊断预留，数值诊断GPU占用上界为1175.0秒，低于3.5节1小时上限。

当前唯一待定项：本次第1.5条要求沿用上一轮确定性设置，而旧LWY报告写为level0；该设置使Dreamer神经内核数值异常。已向用户提出是否恢复level1的决定，尚未收到答复，因此未启动依赖可信验证的各4小时正式训练，也未将候选结果当作正式最终比较。允许后只需记录决定、激活配置并运行独立队列；不需要重新生成数据或重新初始化已通过冒烟的模型。

此前产生的神经网络结果已全部作废，不能计入正式最终条件。完整要求为5方法×4数据集×随机障碍/过滤4条件；原版、盲测、旧留出各100回合，mini30回合。未完成项不以旧协议结果代替。

注意：此前level0神经模型结果已作废并从下表移除，原始JSON仅作故障记录。规划器不使用该神经内核；其100/100结果与G1首10回合轨迹逐位一致。

| 方法 | 数据集 | 障碍/过滤 | N | 成功率 [Wilson95%] | 平均门数 | 接管率 | 速度m/s | IoU 全/遮挡/障碍 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner | original | 0/0 | 100 | 100.0% [96.3%,100.0%] | 8.000 | 0.00% | 0.352 | 不适用 |

失败分类、每回合s位置/对应原始区段、接管数、动作、轨迹哈希、模型哈希与奖励分量在各 `metrics/lwy_v2_final_*.json`。IoU每20策略步采样；无预测头或空并集不报伪造的0。

最终Dreamer/PPO对比视频尚未生成；`lwy_v2_renderer_probe.mp4` 若存在，仅是原版XML渲染检查，不冒充模型推理视频。

## 检查、产物与未解决事项

[lwy_v2_check_diag_final.json](metrics/lwy_v2_check_diag_final.json)：{"passed": true, "protected_changed": []}

[lwy_v2_checks_final.json](metrics/lwy_v2_checks_final.json)：{"passed": true, "protected_files": 3840, "protected_changed": []}

[lwy_v2_check_diag.json](metrics/lwy_v2_check_diag.json)：{"passed": true, "protected_changed": []}

[lwy_v2_checks_legacy.json](metrics/lwy_v2_checks_legacy.json)：{"passed": true, "protected_files": 3840, "protected_changed": []}

[lwy_v2_checks.json](metrics/lwy_v2_checks.json)：{"passed": true, "protected_files": 3840, "protected_changed": []}

[lwy_v2_sampler_probe.json](metrics/lwy_v2_sampler_probe.json)：{"passed": true}

[lwy_v2_mac_integrity_check.json](metrics/lwy_v2_mac_integrity_check.json)：{"passed": true, "protected_files": 1572}

[lwy_v2_cleanup.json](metrics/lwy_v2_cleanup.json)：{"queue_pid": 1563841, "nav3_sessions": [], "owned_background_processes": []}

地图、checkpoint、replay及大型栅格留在远端 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`；Mac同步报告、代码、JSON、图、日志和小视频。源包两处校验均为137文件/642,046,794字节，旧物理指纹保持 `a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d`。

较早的`lwy_v2_checks.json`顶层`g1_passed=false`引用上一轮旧G1；其`v2.passed=true`为新版几何回归通过。最终`lwy_v2_checks_final.json`已将上一轮状态单列为`legacy_g1_passed`，顶层G1对应本轮，并核对当前环境哈希。G1资格原始证据仍以`lwy_v2_g1.json`为准。

剩余限制：原始几何为视频近似，未实测标定；中心线静态净空不能保证任意偏航/侧移轨迹；离散通路见证不等于动力学保证；保守栅格仍会拒绝有数厘米原始净空的位姿。未完成训练/评估必须以队列状态为准。

进度命令：`ssh hpf@100.108.114.64 "cd ~/wang-sm/ATEC_NAV_WSM/research/exp_nav3 && cat metrics/lwy_v2_queue.json"`；实时日志以`active_log`为准，并行评估时看`active_logs`。手动同步最新报告和小产物：`python3 scripts/lwy_v2_fetch.py`。

## 本轮代码清单

| 文件 | 作用 |
| --- | --- |
| nav3/lwy_dataset_v2.py | 末门偏移、全几何反射、10 cm净空筛选、固定重采样与新划分。 |
| nav3/lwy_control_v2.py | 显式规划参数与时域选项，旧默认保留。 |
| nav3/lwy_env_v2.py | 可配置航向增益，物理判定原样保留；子步诊断。 |
| nav3/lwy_eval_v2.py | 独立PRNG、单环境vmap的G1与训练闸门。 |
| nav3/lwy_probe_v2.py | 短段、完整历史、精确现场保存/回放和单变量实验。 |
| nav3/lwy_v2_geometry_diag.py / lwy_v2_trace_analysis.py / lwy_summary_v2.py | 失败几何、子步净空、受控实验汇总与图。 |
| nav3/lwy_obstacles_v2.py / lwy_safe_obstacles.py | 静态+随机障碍通路检查、拒绝采样目录与运行时选择。 |
| nav3/lwy_mixed.py / lwy_sampler_probe_v2.py | 真正按回合75/25混合与重置回归。 |
| nav3/lwy_memory_v2.py | 全部地图+真实Agent+优化器的显存实测。 |
| nav3/lwy_training_runtime.py | 将原训练运行时与确定性评估设置分开，恢复Dreamer level1及PPO原库默认。 |
| nav3/lwy_checkpoint_v2.py | 无损保存bfloat16策略状态，兼容恢复本轮早期的NumPy |V2状态文件。 |
| nav3/lwy_gradient_diagnosis.py / lwy_forward_diagnosis.py / lwy_numerics_report.py | 冻结数据的单变量数值诊断、推理核对和四项原因排查。 |
| nav3/lwy_train_dreamer.py / nav3/train_ppo.py | 官方Dreamer循环GPU驱动、PPO显式LWY入口及状态恢复。 |
| nav3/lwy_ppo_update.py | 保持PPO损失与更新，补充梯度范数、策略方差和回报尺度日志。 |
| nav3/lwy_grouped_eval.py | 未采用的分组速度与逐位一致性实验，正式评估不导入。 |
| nav3/dreamer_adapter.py | 可选JAX确定性配置覆盖，默认不变。 |
| nav3/lwy_policy_v2.py / lwy_compare_v2.py | 独立策略PRNG、固定推理形状、统一验证/最终比较。 |
| nav3/lwy_comparison_runtime.py | 正式评估默认保留level0；只有用户决定和候选证据齐备才可启用level1，结果使用独立_rt1文件名。 |
| nav3/lwy_queue_v2.py / lwy_collapse_v2.py | 限时预算队列、半小时快照验证、早停与证据诊断。 |
| nav3/lwy_video_v2.py / lwy_report_v2.py | 原版XML推理对比视频、增量报告和曲线。 |
| configs/lwy_v2_*.json | 独立参数及失败现场配置，旧配置不改。 |
| scripts/lwy_v2_*.sh | 可脱离客户端执行的数据生成、诊断、G1、协议与零样本任务。 |
| scripts/lwy_v2_fetch.py | 只同步本轮报告与小产物；可有界后台等待远端完成，完成后审计进程并退出。 |
| tests/check_lwy.py | 新增末门偏移、完整反射/双反射、阈值、种子隔离与旧文件保护检查。 |
<a id="lwy-runtime2"></a>
# 第六轮续 2：正式训练与最终评估

更新时间：2026-11-12T06:01:02.945941+00:00；队列状态：`complete`。

## 协议修正与历史结果作废

验收方已批准所有方法（Dreamer、PPO、规划器）的训练与评估统一使用 `--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true`，并设 `TF_CUDNN_DETERMINISTIC=1`。此决定替代 TASK_LWY2.md 第1.5条旧设置要求。评估继续使用每回合独立PRNG、单环境vmap、固定100条策略流。

原因：本机level0导致Dreamer可通行图恒为0.48828125、原始梯度约10¹⁵；冻结同一批次改为level1后恢复正常。level1已通过600条完整轨迹重复一致性及G1复核。所有level0神经网络评估均作废，包括旧模型零样本0/100；作废结果不进入任何对比表，在新协议下全部重测。规划器G1与autotune无关，原版10/10、盲测20/20继续有效。[lwy_v2_evaluation_runtime_decision.json](metrics/lwy_v2_evaluation_runtime_decision.json)。

第五轮0.5–1小时的续训崩溃不是autotune问题：当时训练、评估都是level1。此次保留踩出路面失败位置/动作、奖励分量及新旧replay采样比例，用于识别相同的行为退化。

## 初始化、规模与保护机制

正式训练分别从原始健康 `dreamer_s0`、`ppo_s0` 只读复制至 `checkpoints/dreamer_lwy_runtime2/`、`checkpoints/ppo_lwy_runtime2/`。已有冒烟检查点和旧队列均保持不变；本次重新从原始权重开始。保留优化器、Dreamer旧replay的100,000个序列起点；正式阶段的段间恢复保存环境、PRNG、策略和replay状态。冒烟/恢复证据沿用上一轮已完成的两段短运行，PPO新协议另经完整地图+Agent+优化器实测。

Dreamer：4608环境，size12m，batch16×64，train ratio32，官方训练循环。每条训练计划8段×30分钟，训练不开过滤。每段复制独立快照并验证20图×2回合；原始梯度>10¹²或非有限立即停止。连续两次比最佳低超过30个百分点，或连续两次通过率下降，均停止该方法并诊断，不自动重启。最终按验证全通过率、平均门数、较早快照依次选择，测试集不参与选择。

原ppo_s0实际环境数为640，上一轮报告称128为原配置不准确。PPO模型仅741,015参数；显存不是由参数量或环境数单独决定。新任务常驻全部200张LWY裁片（约4.90GiB）和200张旧图（约1.33GiB），还包括128步rollout的BEV、循环状态、序列重排/堆叠临时张量、反向计算及默认BFC分配器保留池。nvidia-smi统计的是整卡占用，包括约911MiB桌面基线，不能把17,717MiB都称作128个环境的状态显存。本次原640环境完整测量通过，按用户条件恢复640，学习率、rollout128、sequence32、epochs4、minibatch16和GRU256均保持原配置。

[lwy_runtime2_ppo_memory_analysis.json](metrics/lwy_runtime2_ppo_memory_analysis.json)。

| 环境数 | 完成 | 峰值MiB | 各阶段峰值MiB |
| --- | --- | --- | --- |
| 128 | True | 17717 | {"loading": 9507, "rollout128": 17713, "optimizer": 17717} |
| 640 | True | 23529 | {"loading": 9507, "rollout128": 23529, "optimizer": 23529} |

已记账GPU占用 10.824 小时 / 12小时；含此前准备、诊断、冒烟和本次显存检查。两个评估进程并行时按占用墙钟并集计账。独立tmux队列有互斥检查、每子作业超时及预算上限。

## 训练曲线与模型选择

| 方法 | 快照h | 验证成功 | 平均门 | 停止原因 |
| --- | --- | --- | --- | --- |
| dreamer_lwy_runtime2 | 0.00 | 0.0% | 0.000 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 0.50 | 2.5% | 2.075 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 1.00 | 12.5% | 3.250 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 1.50 | 27.5% | 4.975 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 2.00 | 40.0% | 5.500 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 2.50 | 40.0% | 6.350 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 3.00 | 62.5% | 6.825 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 3.50 | 50.0% | 6.875 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| dreamer_lwy_runtime2 | 4.00 | 37.5% | 6.600 | two successive decreases in validation success; continuation request requires diagnosis before further training |
| ppo_lwy_runtime2 | 0.00 | 0.0% | 0.000 |  |
| ppo_lwy_runtime2 | 0.50 | 0.0% | 0.050 |  |
| ppo_lwy_runtime2 | 1.00 | 0.0% | 0.150 |  |
| ppo_lwy_runtime2 | 1.50 | 0.0% | 0.425 |  |
| ppo_lwy_runtime2 | 2.00 | 0.0% | 0.450 |  |
| ppo_lwy_runtime2 | 2.50 | 0.0% | 0.000 |  |
| ppo_lwy_runtime2 | 3.00 | 0.0% | 0.625 |  |
| ppo_lwy_runtime2 | 3.50 | 0.0% | 0.725 |  |
| ppo_lwy_runtime2 | 4.00 | 0.0% | 0.550 |  |

![验证曲线](figures/lwy_runtime2_learning_curve.png)

| 最终方法 | 选中快照 | 验证成功 | 平均门 |
| --- | --- | --- | --- |
| dreamer_lwy | checkpoints/snapshots/dreamer_lwy_runtime2_3.00h/latest.pkl | 62.5% | 6.825 |
| ppo_lwy | checkpoints/snapshots/ppo_lwy_runtime2_3.50h/latest.pkl | 0.0% | 0.725 |

[lwy_runtime2_model_selection.json](metrics/lwy_runtime2_model_selection.json)。

## 崩溃诊断

[lwy_v2_collapse_dreamer_lwy_runtime2_4.00h.json](metrics/lwy_v2_collapse_dreamer_lwy_runtime2_4.00h.json)

## 最终对比

已完成 80/80 个条件；完整矩阵为五方法×四数据集×随机障碍/过滤四条件。原版、盲测、旧留出各100回合，mini30回合。所有下表数字由新协议JSON生成。

| 方法 | 数据集 | 障碍/过滤 | N | 全通过率 [Wilson95%] | 平均门 | 失败分类 | 接管率 | 速度m/s | IoU全/遮挡/障碍 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dreamer_lwy | blind | 0/0 | 100 | 68.0% [58.3%,76.3%] | 7.250 | {"success": 68, "support_or_drop": 25, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 7} | 0.00% | 0.644 | 0.833/0.874/无定义 |
| dreamer_lwy | blind | 0/1 | 100 | 66.0% [56.3%,74.5%] | 6.830 | {"success": 66, "support_or_drop": 32, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 2} | 14.67% | 0.609 | 0.834/0.870/无定义 |
| dreamer_lwy | blind | 1/0 | 100 | 3.0% [1.0%,8.5%] | 2.770 | {"success": 3, "support_or_drop": 36, "collision": 60, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 1} | 0.00% | 0.636 | 0.849/0.883/0.308 |
| dreamer_lwy | blind | 1/1 | 100 | 3.0% [1.0%,8.5%] | 1.850 | {"success": 3, "support_or_drop": 87, "collision": 10, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 16.34% | 0.585 | 0.854/0.884/0.328 |
| dreamer_lwy | heldout | 0/0 | 100 | 88.0% [80.2%,93.0%] | 7.550 | {"success": 88, "support_or_drop": 0, "collision": 12, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.505 | 0.912/0.925/无定义 |
| dreamer_lwy | heldout | 0/1 | 100 | 96.0% [90.2%,98.4%] | 7.930 | {"success": 96, "support_or_drop": 0, "collision": 4, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 22.52% | 0.473 | 0.914/0.929/无定义 |
| dreamer_lwy | heldout | 1/0 | 100 | 51.0% [41.3%,60.6%] | 5.950 | {"success": 51, "support_or_drop": 7, "collision": 42, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.510 | 0.910/0.917/0.445 |
| dreamer_lwy | heldout | 1/1 | 100 | 70.0% [60.4%,78.1%] | 6.570 | {"success": 70, "support_or_drop": 4, "collision": 26, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 23.96% | 0.469 | 0.910/0.917/0.453 |
| dreamer_lwy | mini | 0/0 | 30 | 90.0% [74.4%,96.5%] | 7.800 | {"success": 27, "support_or_drop": 0, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.485 | 0.906/0.915/无定义 |
| dreamer_lwy | mini | 0/1 | 30 | 33.3% [19.2%,51.2%] | 6.667 | {"success": 10, "support_or_drop": 0, "collision": 20, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.13% | 0.443 | 0.910/0.914/无定义 |
| dreamer_lwy | mini | 1/0 | 30 | 50.0% [33.2%,66.8%] | 5.767 | {"success": 15, "support_or_drop": 3, "collision": 12, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.480 | 0.909/0.912/0.500 |
| dreamer_lwy | mini | 1/1 | 30 | 26.7% [14.2%,44.4%] | 5.700 | {"success": 8, "support_or_drop": 0, "collision": 22, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.92% | 0.440 | 0.905/0.902/0.458 |
| dreamer_lwy | original | 0/0 | 100 | 59.0% [49.2%,68.1%] | 6.640 | {"success": 59, "support_or_drop": 25, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 16} | 0.00% | 0.655 | 0.803/0.834/无定义 |
| dreamer_lwy | original | 0/1 | 100 | 61.0% [51.2%,70.0%] | 6.420 | {"success": 61, "support_or_drop": 28, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 11} | 16.31% | 0.614 | 0.808/0.841/无定义 |
| dreamer_lwy | original | 1/0 | 100 | 6.0% [2.8%,12.5%] | 2.270 | {"success": 6, "support_or_drop": 39, "collision": 55, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.645 | 0.806/0.848/0.271 |
| dreamer_lwy | original | 1/1 | 100 | 1.0% [0.2%,5.4%] | 1.170 | {"success": 1, "support_or_drop": 94, "collision": 5, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 16.17% | 0.608 | 0.816/0.868/0.230 |
| dreamer_s0 | blind | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 99, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.292 | 0.281/0.246/无定义 |
| dreamer_s0 | blind | 0/1 | 100 | 0.0% [0.0%,3.7%] | 0.050 | {"success": 0, "support_or_drop": 97, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 47.05% | 0.301 | 0.290/0.251/无定义 |
| dreamer_s0 | blind | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 99, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.292 | 0.281/0.246/无定义 |
| dreamer_s0 | blind | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.040 | {"success": 0, "support_or_drop": 97, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 49.23% | 0.300 | 0.289/0.248/0.207 |
| dreamer_s0 | heldout | 0/0 | 100 | 89.0% [81.4%,93.7%] | 7.630 | {"success": 89, "support_or_drop": 1, "collision": 10, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.510 | 0.953/0.957/无定义 |
| dreamer_s0 | heldout | 0/1 | 100 | 100.0% [96.3%,100.0%] | 8.000 | {"success": 100, "support_or_drop": 0, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 23.02% | 0.493 | 0.951/0.955/无定义 |
| dreamer_s0 | heldout | 1/0 | 100 | 73.0% [63.6%,80.7%] | 6.940 | {"success": 73, "support_or_drop": 2, "collision": 25, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.509 | 0.949/0.949/0.547 |
| dreamer_s0 | heldout | 1/1 | 100 | 89.0% [81.4%,93.7%] | 7.440 | {"success": 89, "support_or_drop": 5, "collision": 6, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 23.95% | 0.490 | 0.946/0.945/0.535 |
| dreamer_s0 | mini | 0/0 | 30 | 0.0% [0.0%,11.4%] | 6.000 | {"success": 0, "support_or_drop": 0, "collision": 30, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.477 | 0.951/0.950/无定义 |
| dreamer_s0 | mini | 0/1 | 30 | 56.7% [39.2%,72.6%] | 7.133 | {"success": 17, "support_or_drop": 0, "collision": 13, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.42% | 0.465 | 0.948/0.949/无定义 |
| dreamer_s0 | mini | 1/0 | 30 | 0.0% [0.0%,11.4%] | 5.600 | {"success": 0, "support_or_drop": 0, "collision": 30, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.475 | 0.947/0.938/0.547 |
| dreamer_s0 | mini | 1/1 | 30 | 50.0% [33.2%,66.8%] | 6.600 | {"success": 15, "support_or_drop": 0, "collision": 15, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.22% | 0.459 | 0.943/0.938/0.559 |
| dreamer_s0 | original | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.285 | 0.357/0.271/无定义 |
| dreamer_s0 | original | 0/1 | 100 | 0.0% [0.0%,3.7%] | 0.030 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 53.14% | 0.368 | 0.357/0.329/无定义 |
| dreamer_s0 | original | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.285 | 0.357/0.271/无定义 |
| dreamer_s0 | original | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.020 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 53.33% | 0.368 | 0.361/0.330/0.041 |
| planner | blind | 0/0 | 100 | 93.0% [86.3%,96.6%] | 7.860 | {"success": 93, "support_or_drop": 7, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.355 | 不适用 |
| planner | blind | 0/1 | 100 | 93.0% [86.3%,96.6%] | 7.860 | {"success": 93, "support_or_drop": 7, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.69% | 0.355 | 不适用 |
| planner | blind | 1/0 | 100 | 0.0% [0.0%,3.7%] | 1.510 | {"success": 0, "support_or_drop": 97, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.346 | 不适用 |
| planner | blind | 1/1 | 100 | 0.0% [0.0%,3.7%] | 1.510 | {"success": 0, "support_or_drop": 97, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.61% | 0.346 | 不适用 |
| planner | heldout | 0/0 | 100 | 80.0% [71.1%,86.7%] | 7.680 | {"success": 80, "support_or_drop": 0, "collision": 20, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.304 | 不适用 |
| planner | heldout | 0/1 | 100 | 80.0% [71.1%,86.7%] | 7.680 | {"success": 80, "support_or_drop": 0, "collision": 20, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.31% | 0.304 | 不适用 |
| planner | heldout | 1/0 | 100 | 67.0% [57.3%,75.4%] | 6.860 | {"success": 67, "support_or_drop": 0, "collision": 33, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.301 | 不适用 |
| planner | heldout | 1/1 | 100 | 67.0% [57.3%,75.4%] | 6.860 | {"success": 67, "support_or_drop": 0, "collision": 33, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 1.50% | 0.301 | 不适用 |
| planner | mini | 0/0 | 30 | 63.3% [45.5%,78.1%] | 7.267 | {"success": 19, "support_or_drop": 0, "collision": 11, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.296 | 不适用 |
| planner | mini | 0/1 | 30 | 63.3% [45.5%,78.1%] | 7.267 | {"success": 19, "support_or_drop": 0, "collision": 11, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.80% | 0.296 | 不适用 |
| planner | mini | 1/0 | 30 | 60.0% [42.3%,75.4%] | 6.133 | {"success": 18, "support_or_drop": 0, "collision": 12, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.288 | 不适用 |
| planner | mini | 1/1 | 30 | 60.0% [42.3%,75.4%] | 6.133 | {"success": 18, "support_or_drop": 0, "collision": 12, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 2.45% | 0.288 | 不适用 |
| planner | original | 0/0 | 100 | 100.0% [96.3%,100.0%] | 8.000 | {"success": 100, "support_or_drop": 0, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.352 | 不适用 |
| planner | original | 0/1 | 100 | 100.0% [96.3%,100.0%] | 8.000 | {"success": 100, "support_or_drop": 0, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.73% | 0.352 | 不适用 |
| planner | original | 1/0 | 100 | 0.0% [0.0%,3.7%] | 1.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.380 | 不适用 |
| planner | original | 1/1 | 100 | 0.0% [0.0%,3.7%] | 1.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.55% | 0.380 | 不适用 |
| ppo_lwy | blind | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.660 | {"success": 0, "support_or_drop": 98, "collision": 2, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.541 | 不适用 |
| ppo_lwy | blind | 0/1 | 100 | 1.0% [0.2%,5.4%] | 1.310 | {"success": 1, "support_or_drop": 96, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.91% | 0.512 | 不适用 |
| ppo_lwy | blind | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.390 | {"success": 0, "support_or_drop": 90, "collision": 9, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 1} | 0.00% | 0.500 | 不适用 |
| ppo_lwy | blind | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.590 | {"success": 0, "support_or_drop": 94, "collision": 6, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 24.38% | 0.465 | 不适用 |
| ppo_lwy | heldout | 0/0 | 100 | 59.0% [49.2%,68.1%] | 6.050 | {"success": 59, "support_or_drop": 17, "collision": 24, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.529 | 不适用 |
| ppo_lwy | heldout | 0/1 | 100 | 96.0% [90.2%,98.4%] | 7.840 | {"success": 96, "support_or_drop": 4, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 5.31% | 0.501 | 不适用 |
| ppo_lwy | heldout | 1/0 | 100 | 49.0% [39.4%,58.7%] | 5.350 | {"success": 49, "support_or_drop": 17, "collision": 34, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.539 | 不适用 |
| ppo_lwy | heldout | 1/1 | 100 | 80.0% [71.1%,86.7%] | 7.170 | {"success": 80, "support_or_drop": 13, "collision": 7, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 6.01% | 0.493 | 不适用 |
| ppo_lwy | mini | 0/0 | 30 | 53.3% [36.1%,69.8%] | 6.767 | {"success": 16, "support_or_drop": 4, "collision": 10, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.501 | 不适用 |
| ppo_lwy | mini | 0/1 | 30 | 83.3% [66.4%,92.7%] | 7.467 | {"success": 25, "support_or_drop": 2, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 5.94% | 0.490 | 不适用 |
| ppo_lwy | mini | 1/0 | 30 | 33.3% [19.2%,51.2%] | 5.633 | {"success": 10, "support_or_drop": 7, "collision": 13, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.489 | 不适用 |
| ppo_lwy | mini | 1/1 | 30 | 70.0% [52.1%,83.3%] | 6.967 | {"success": 21, "support_or_drop": 3, "collision": 6, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 6.53% | 0.473 | 不适用 |
| ppo_lwy | original | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.560 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.535 | 不适用 |
| ppo_lwy | original | 0/1 | 100 | 1.0% [0.2%,5.4%] | 1.170 | {"success": 1, "support_or_drop": 99, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 28.16% | 0.524 | 不适用 |
| ppo_lwy | original | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.430 | {"success": 0, "support_or_drop": 96, "collision": 4, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.528 | 不适用 |
| ppo_lwy | original | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.660 | {"success": 0, "support_or_drop": 97, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 30.29% | 0.512 | 不适用 |
| ppo_s0 | blind | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 99, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.288 | 不适用 |
| ppo_s0 | blind | 0/1 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 98, "collision": 2, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 4.58% | 0.287 | 不适用 |
| ppo_s0 | blind | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 99, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.288 | 不适用 |
| ppo_s0 | blind | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 98, "collision": 2, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 4.58% | 0.287 | 不适用 |
| ppo_s0 | heldout | 0/0 | 100 | 55.0% [45.2%,64.4%] | 5.930 | {"success": 55, "support_or_drop": 30, "collision": 15, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.502 | 不适用 |
| ppo_s0 | heldout | 0/1 | 100 | 75.0% [65.7%,82.5%] | 6.930 | {"success": 75, "support_or_drop": 24, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 1.90% | 0.499 | 不适用 |
| ppo_s0 | heldout | 1/0 | 100 | 44.0% [34.7%,53.8%] | 5.390 | {"success": 44, "support_or_drop": 26, "collision": 30, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.502 | 不适用 |
| ppo_s0 | heldout | 1/1 | 100 | 63.0% [53.2%,71.8%] | 6.280 | {"success": 63, "support_or_drop": 32, "collision": 5, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 2.57% | 0.497 | 不适用 |
| ppo_s0 | mini | 0/0 | 30 | 56.7% [39.2%,72.6%] | 6.567 | {"success": 17, "support_or_drop": 10, "collision": 3, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.498 | 不适用 |
| ppo_s0 | mini | 0/1 | 30 | 83.3% [66.4%,92.7%] | 7.267 | {"success": 25, "support_or_drop": 4, "collision": 1, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 2.00% | 0.497 | 不适用 |
| ppo_s0 | mini | 1/0 | 30 | 30.0% [16.7%,47.9%] | 5.267 | {"success": 9, "support_or_drop": 13, "collision": 8, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.502 | 不适用 |
| ppo_s0 | mini | 1/1 | 30 | 60.0% [42.3%,75.4%] | 6.433 | {"success": 18, "support_or_drop": 1, "collision": 11, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 2.77% | 0.491 | 不适用 |
| ppo_s0 | original | 0/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.278 | 不适用 |
| ppo_s0 | original | 0/1 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 8.27% | 0.276 | 不适用 |
| ppo_s0 | original | 1/0 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 0.00% | 0.278 | 不适用 |
| ppo_s0 | original | 1/1 | 100 | 0.0% [0.0%,3.7%] | 0.000 | {"success": 0, "support_or_drop": 100, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 0} | 8.27% | 0.276 | 不适用 |

逐回合失败s、对应1km区段、动作与奖励分量在 `metrics/lwy_runtime2_final_*_rt1.json`；IoU每20策略步采样，空并集记为无定义。

## 视频与检查

[lwy_runtime2_dreamer_vs_ppo_rt1.mp4](videos/lwy_runtime2_dreamer_vs_ppo_rt1.mp4)：同种子、不加过滤，原版XML回放快环境真实推理位姿；不是腿部动力学验证。

[lwy_runtime2_check_diag_final.json](metrics/lwy_runtime2_check_diag_final.json)：{"passed": true, "undefined_iou_checked": true, "protected_original_files": 679, "protected_changed": [], "environment_fingerprint": "a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d", "isolated_variant_nonlog_occupancy_changes": 0, "mini_in_train": false, "synthetic_seeds_in_train": false, "original_checkpoint_hashes": {"dreamer_s0": "49832630385c8c1bc3fbd4e302da1e65bd1616afd5d1470edcb0c683b7d72353", "ppo_s0": "e505feff7ecb32b9a5a48efd685f0879906bb8c2d085431a7f5221232a051cc7"}}

[lwy_runtime2_checks_final.json](metrics/lwy_runtime2_checks_final.json)：{"passed": true, "protected_files": 4311, "protected_changed": [], "protected_maps_added": [], "split_sizes": {"train": 200, "val": 20, "blind": 20, "test": 1}, "mini_in_train": false, "test_in_train": false, "validation_blind_in_train": false, "old_environment_fingerprint": "a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d", "g1_passed": true, "v2": {"passed": true, "version": "lwy-v2.1", "split_sizes": {"train": 200, "val": 20, "blind": 20, "test": 1}, "mirror_involution": true, "mirror_projected_geometry": true, "clearance_threshold_enforced": true, "endpoint_offset_m": 1.5, "old_lwy_files_unchanged": 1203}, "legacy_g1_passed": false, "g1_protocol_current": true}

## 未解决问题与产物

地图仍是视频近似，未经实测；保守栅格、离散通路和快环境动力学的限制沿用上文。仅一个训练种子；没有以测试成绩调参。若学习策略低于规划器，按实际结果汇报，不用过滤成绩代替无过滤能力。

队列状态：[lwy_runtime2_queue.json](metrics/lwy_runtime2_queue.json)；训练日志 `metrics/train_*_lwy_runtime2.jsonl`；快照、replay与大型地图只留远端 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`。报告、指标、图、日志、小视频同步回Mac。

本次改动：`lwy_training_runtime.py`统一批准设置；`lwy_queue_v2.py --released`隔离正式运行与预算；`lwy_train_dreamer.py`支持独立目录；`train_ppo.py`逐更新检查原始梯度；`lwy_memory_v2.py`隔离测量输出；`lwy_collapse_v2.py`读取指定运行；`lwy_video_v2.py`读取指定模型及参考轨迹；`lwy_report_runtime2.py`从有效结果生成本节。

<!-- LWY_R7_START -->
# 第七轮

状态：`failures_complete`。仅在 Mac nav3 与指定远端 exp_nav3 内工作。

## 随机障碍修复与闸门

第六轮正式训练/评估使用 SafeLwyEnv 的离散可行性目录，不是直接使用 LwyEnv 的原始提案。旧目录提供离散几何通路，却不足以保证带噪声规划器绕行通过。原版 ob1 的旧逐回合记录是全部已过 1 门再失败，任务描述中的“第 1 门之前”与原始数据不一致。

本轮入口 `LwyEnvR7` 默认 `obstacle_rule=shoulder`：障碍保留原尺寸、朝向、沿程位置和左右侧，仅向路肩外移至名义侵入路面 4 cm。不扩展支撑区；包络、栅格保守量、支撑/碰撞终止条件和运动学函数全部保持原样。对每套摆放，复核全部 0.1 m 离散见证位姿与新障碍矩形的分离轴，并保留原静态栅格证书；另报告障碍附近中心线路径的真实栅格余量。离散证书仍有采样误差限制，规划器闸门另外实测。

新评估入口 `python -m nav3.lwy_compare_r7` 默认修复规则，`--obstacle-rule legacy` 可复现第六轮规则；历史代码和目录不改。旧宽路 heldout/mini 保持原已检查的摆放规则。

随机障碍闸门：原版 10/10；盲测 95/100；通过=True。

实测栅格分辨率 0.050 m，保守量 PAD=0.0528 m。支撑层存储 edge + 0.025 + PAD，而终止阈值是 0.025；两者抵消，阈值不能当作额外可驶路肩。

原版目录 8 套×3 障碍的矩形完整边长范围为 0.246–0.428 m；沿程位置、原/新横向位置与尺寸逐项列在 `metrics/lwy_r7_obstacle_geometry.json`。

| 摆放规则 | 原版障碍切片几何通道最小m | 可行机体中心区间宽度中位m | 最小m |
| --- | --- | --- | --- |
| legacy | 0.712 | 0.110 | 0.045 |
| shoulder | 1.010 | 0.373 | 0.300 |

切片只说明横向余量，不单独证明路径连通；连通性与噪声跟踪分别由完整离散证书和规划器闸门验证。

## 退化诊断

旧日志中 actor 熵字段为 `train/ent/action`，旧崩溃提取器漏掉了它。3–4 h：熵约 0.721→0.527；BEV 损失 238.41→209.15、priv_trav 损失 14.43→12.66、价值损失 1.019→0.917；梯度约 775→790。旧 replay 样本比例已经为 0。世界模型损失下降和有限梯度不能保证闭环策略稳定；旧数据耗尽与失败相关，但不能单凭相关性断言因果。

3 h 官方完整状态仍在。replay 清单中 4608 个无对应磁盘文件的记录长度全部为 0，恢复器按原逻辑重建空块，其余非空历史分块完整。新运行复制精确 3 h replay/优化器/随机数/采集器状态，不使用 4 h 的后续样本。为适应共享 GPU，仅保留固定的 2048 个采集器前缀，不同步重置；三条对照保持相同缩减。本轮 replay_old 指 3 h 快照已有数据，历史日志 replay_old 指最早 dreamer_s0 数据，二者口径不同。

短对照：原学习率 4e-5 + online=True；仅学习率降为 1e-5；仅 online=False。每条计划 12 min，验证 20 图×4 回合。有限预算和单次干预不足时，不宣称唯一根因。

| 快照h | 回报归一化尺度 |
| --- | --- |
| 3.0 | 63.634 |
| 3.5 | 61.539 |
| 4.0 | 58.445 |

三个快照的尺度下降约 8%，未显示大幅跳变；只有三个采样点，不能排除中间的瞬时波动。不能将归一化后的 ret 值误当成尺度。

旧 ob1 原版规划器失败都在第三区段 root_and_sign_bends，s=161.0–240.8 m，中位 190.9 m；末态横向偏移绝对值中位 0.109 m。几何证书虽可行，带定位误差的绕行所需动态余量不足，不能将 0/100 简化为每套几何都绝对堵死。

| 历史指标 | 约3.00h | 最后日志约4.03h |
| --- | --- | --- |
| train/ent/action | 0.721267 | 0.527266 |
| train/loss/bev | 238.406 | 209.149 |
| train/loss/con | 0.0214561 | 0.022905 |
| train/loss/dyn | 18.359 | 17.9427 |
| train/loss/policy | -0.000925846 | -0.00251879 |
| train/loss/priv_trav | 14.4287 | 12.6616 |
| train/loss/rep | 18.359 | 17.9427 |
| train/loss/repval | 1.23708 | 1.16457 |
| train/loss/rew | 0.658984 | 0.653524 |
| train/loss/state | 0.00744748 | 0.00937522 |
| train/loss/value | 1.01921 | 0.916556 |
| train/opt/grad_norm | 774.891 | 790.162 |

| 历史验证快照 | 支撑失败总数 | s=970–990m | s≥990m | s=150–170m |
| --- | --- | --- | --- | --- |
| 3.0h | 13 | 6 | 0 | 0 |
| 4.0h | 25 | 11 | 7 | 4 |

上表是历史每图2回合协议，只用于分析同一历史验证集的退化；本轮选模统一使用每图4回合，不将两种样本数混作提升。

### 既有失败位置与动作

| 数据集 | 失败类型 | 区段 | 次数 | 横向偏移绝对值中位m | 请求速度均值m/s |
| --- | --- | --- | --- | --- | --- |
| original | support_or_drop | 2 narrow_woodland_path | 2 | 0.107 | 0.800 |
| original | support_or_drop | 4 dense_canopy | 9 | 0.094 | 0.779 |
| original | stuck_or_timeout | 10 final_steps_and_moss | 16 | 0.083 | 0.028 |
| original | support_or_drop | 3 root_and_sign_bends | 3 | 0.094 | 0.746 |
| original | support_or_drop | 10 final_steps_and_moss | 5 | 0.159 | 0.679 |
| original | support_or_drop | 1 entry_stairs | 1 | 0.097 | 0.570 |
| original | support_or_drop | 5 fern_cutting | 5 | 0.122 | 0.766 |
| blind | support_or_drop | 10 narrow_woodland_path | 3 | 0.262 | 0.127 |
| blind | support_or_drop | 3 dense_canopy | 1 | 0.114 | 0.800 |
| blind | support_or_drop | 5 retaining_ledge | 1 | 0.126 | 0.794 |
| blind | support_or_drop | 10 root_and_sign_bends | 7 | 0.215 | 0.416 |
| blind | support_or_drop | 1 final_steps_and_moss | 1 | 0.153 | 0.723 |
| blind | stuck_or_timeout | 10 narrow_woodland_path | 3 | 0.096 | 0.037 |
| blind | support_or_drop | 3 root_and_sign_bends | 2 | 0.115 | 0.789 |
| blind | support_or_drop | 5 narrow_woodland_path | 1 | 0.127 | 0.792 |
| blind | support_or_drop | 5 fern_cutting | 1 | 0.122 | 0.753 |
| blind | support_or_drop | 9 green_railing_descent | 2 | 0.151 | 0.770 |
| blind | support_or_drop | 10 exposed_scrub_ridge | 3 | 0.236 | 0.495 |
| blind | support_or_drop | 2 green_railing_descent | 1 | 0.144 | 0.766 |
| blind | stuck_or_timeout | 10 entry_stairs | 3 | 0.044 | 0.022 |
| blind | support_or_drop | 8 root_and_sign_bends | 1 | 0.092 | 0.714 |
| blind | stuck_or_timeout | 10 final_steps_and_moss | 1 | 0.044 | 0.039 |
| blind | support_or_drop | 10 green_railing_descent | 1 | 0.057 | 0.796 |

旧轨迹每5秒采样，不能据此认定5Hz动作振荡；本轮补录保留失败前25秒逐策略步历史，原始目标点、速度档和横向偏移见对应分析 JSON。

| 对照 | 验证全通过率 | 平均门数 |
| --- | --- | --- |
| baseline | 41.2% | 5.963 |
| low_lr | 48.8% | 5.950 |
| uniform | 48.8% | 5.912 |

按验证结果选择续训配置 `low_lr`。

| 对照 | 实际更新数 | GPU进程秒（含恢复） | 相对未续训新增成功/新增失败 | 相对原学习率新增成功/新增失败 |
| --- | --- | --- | --- | --- |
| baseline | 2176 | 748.8 | 3/24 | 0/0 |
| low_lr | 3712 | 752.1 | 6/21 | 14/8 |
| uniform | 4160 | 759.3 | 6/21 | 16/10 |

各臂时间包含初始化与 replay 恢复，实际更新数见表；更新量差异是本次墙钟对照的限制。逐回合配对结果与探索性 McNemar 检验保存在 `metrics/lwy_r7_diagnosis_analysis.json`，单次验证和多个候选不能证明唯一因果。

| 对照末态 | actor熵 | 梯度范数 | 回报尺度 | 3h已有数据累计采样比例 |
| --- | --- | --- | --- | --- |
| baseline | 1.0942386388778687 | 489.796630859375 | 64.0011215209961 | 0.9936082856537402 |
| low_lr | 0.9424184560775757 | 484.64056396484375 | 62.48956298828125 | 0.9537358642972537 |
| uniform | 0.7283632159233093 | 662.5126953125 | 61.42708206176758 | 0.9770933445458914 |

本轮可确认的是继续优化后的闭环支撑稳定性退化。数值爆炸无证据；最早 s0 数据耗尽是历史事实，但不是由时间相关性就能确定的根因。回报尺度只有有限快照证据，奖励比例未做单变量干预，因此这两项仍是未排除因素，不为写出“唯一根因”而扩大结论。

### 失败前 25 秒的完整动作检查

| 数据集 | 类型 | 数量 | 横向偏移绝对值中位m | 末目标横向绝对值中位m | 请求/实际速度中位m/s | 25秒进展中位m | 转向标准差中位° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | stuck_or_timeout | 16 | 0.083 | 0.056 | 0.012/0.033 | 0.596 | 7.9 |
| original | support_or_drop | 25 | 0.118 | 0.363 | 0.784/0.711 | 18.291 | 13.0 |
| blind | stuck_or_timeout | 7 | 0.060 | 0.071 | 0.035/0.042 | 0.585 | 8.4 |
| blind | support_or_drop | 25 | 0.153 | 0.257 | 0.689/0.446 | 13.543 | 12.7 |

逐回合的区段、坐标、目标点、速度、树根/陡坡诊断、转向换符号比例与奖励分量见 `metrics/lwy_r7_failure_analysis.json`。地形标记是末态局部诊断；随机策略转向变化不能单独证明确定性极限环。不改变台阶和树根限速，避免用提高速度掩盖卡住问题。

## 改动、训练与选模

仅随机摆放规则和选中的优化/采样参数改变。Dreamer 与 PPO 仍共用原奖励；观测与动作规格不改，训练不开过滤，75% LWY / 25% 旧图的按回合混合保持不变。原始梯度超过 10¹² 或非有限立即停止。每 30 min 独立快照，验证 80 回合；连续两次低于最佳超过 30 个百分点或连续两次下降均停止，不自动反复重启。选模依次比较验证全通过率、平均门数、较早快照，测试集不参与。

显存实测：`{"baseline_mib": 5105, "peak_mib": 17260, "available_fraction": 0.6246466930469191}`。他人的 GPU 进程未更改。

## 最终比较

已完成 2/64 个正式条件。原版/盲测/旧留出各100回合，mini30回合。所有方法使用 autotune=1、deterministic_ops、独立回合 PRNG、单环境 vmap、固定100策略流。

| 方法 | 数据集 | ob/sf | 全通过率 [Wilson95%] | 平均门 | 失败分类 | 接管率 | 速度m/s | IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dreamer_lwy | blind | 0/0 | 68.0% [58.3%,76.3%] | 7.250 | {"success": 68, "support_or_drop": 25, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 7} | 0.00% | 0.644 | [0.8325422577349342, 0.8741002389773496, None] |
| dreamer_lwy | original | 0/0 | 59.0% [49.2%,68.1%] | 6.640 | {"success": 59, "support_or_drop": 25, "collision": 0, "wrong_branch_or_reverse": 0, "stuck_or_timeout": 16} | 0.00% | 0.655 | [0.8025012550955826, 0.833508655092908, None] |

## 可选支撑余量过滤

仅评估：从公共 BEV 支撑/路线通道和速度状态预测 0.8 s 足迹，在物理包络外采样检查 8 cm 余量，不读取特权地图。受 0.125 m BEV 栅格和离散足迹采样限制，不是连续几何安全保证。触发时由原规划器接管；默认 margin=0，原过滤器不变。

| 数据集 | 障碍 | 全通过率 | 接管率 |
| --- | --- | --- | --- |

## 验证、视频与未解决事项

视频：`videos/lwy_r7_dreamer.npz/.json` 为远端 CPU 实际推理，`videos/lwy_r7_dreamer.mp4` 在 Mac 渲染。CPU 与 GPU 不承诺逐位一致，视频不能替代正式 GPU 评估；MuJoCo 仅回放快环境位姿，不代表轮足底层验证。

`metrics/lwy_r7_check_diag.json`：通过。
`metrics/lwy_r7_check_lwy.json`：通过。
`metrics/lwy_r7_unit_checks.json`：通过。

执行路径变更：Mac 的整个 ATEC_NAV 目录在任务期间移至 `/Users/amiao/ATEC_NAV/`，后续继续使用其中同一份 nav3；没有在 dreamerv3 中重建或修改文件。

新增回归测试首次使用几何专用 venv，原始错误：`ModuleNotFoundError: No module named 'jax'`。随后改用已有 `$NAV3_PYTHON` 并设置 `JAX_PLATFORMS=cpu`；未向共享环境安装包。原失败日志保留。


GPU 进程墙钟记账：0.996 h。包括显存探测、诊断、训练、验证、最终评估及初始障碍闸门；闸门按 CPU 目录生成完成至最后闸门指标写出的时间差保守计入（223.2 s）。CPU 克隆/几何分析/测试/录像和 Mac 渲染不计 GPU；进程初始化与保存均计入，不以纯 kernel 时间缩减预算。

限制：仅一个训练种子；地图为视频近似，未实测；没有修改真实碰撞/支撑阈值；离散通路并非连续动力学保证；短对照不一定足以识别所有相互作用。未完成步骤仍明确标为待完成。

代码清单：`lwy_obstacles_r7.py` 摆放与证书；`lwy_env_r7.py` 默认/旧规则切换；`lwy_mixed_r7.py` 新训练入口；`lwy_compare_r7.py` 同协议评估；`lwy_support_filter_r7.py` 可选支撑检查；`lwy_train_r7.py` 官方循环续训；`lwy_train_ppo_r7.py` / `lwy_ppo_update_r7.py` PPO检查；`lwy_r7_queue.py` 闸门/预算/早停；`lwy_r7_analysis.py` / `lwy_r7_obstacle_analysis.py` 只读证据；`lwy_r7_clone.py` 精确克隆；`lwy_r7_report.py` 从指标生成本节；`lwy_video_record_r7.py` CPU固定100策略流录像。
