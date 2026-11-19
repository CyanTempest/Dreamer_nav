# HANDOFF — 2026-11-19 19:30

接手前先读 `CLAUDE.md`（规范）。本文件只记当前状态，接手后更新它。

标注：**[已验收]** = 主会话独立核实过；**[执行者报告]** = 来自子代理、未复核；**[推断]** = 没有直接证据。

## matrix-v4 独立分支交接（2026-11-19）

本 worktree 从 `284ecad` 分出。调研、接口证据、采集与评估方案在 `docs/matrix_v4/`，安装包和后续 VM 操作在 `envs/matrix_v4/README.md`。已在服务器 CPU 复核 MATRiX 三分卷 SHA256、Python 锁依赖解析，以及私有包回读和 MJCF 文件引用；私有输入包留在本 worktree 的 `data/matrix_v4_transfer/`。本分支准备阶段的 Colab 用量为零，镜像未构建。其他执行者的 A100 日志另标出处。正式采集前需要验证两路 Airy 在 A 段上的回波、内置登阶状态、冷启动/复位及标定后的闭环速度；模型部分的下一步是离线 BEV 数据适配与限时训练。main 的进度见下文历史记录。

## 1. 现状

- **地图**：用户否定了 0.1 m 栅格的 2.5D 地图观感，要求“在原始 3D 扫描基础上尽可能完整复原”。`ad54145` 合入 `nav4/matrix/`：从稠密 LAS（`map_remov.las`）重建 A 段 0–722 m（路线 ±6 m）的 MuJoCo 场景 `data/trail_a_mujoco/full_v1/`（177 级台阶独立踏面+垂直踢面、扶手胶囊、墙/岩/树根实体、502 棵树干、软植被只视觉、遮挡补面 group=4）。用户看过走赛道/环绕视频（`full_v1/tour_*`，Mac `~/ATEC_NAV/map_views_full_v1/`）后认可：“地图可以”。
- **v8 的已知错误**：8.8 m 处 27.7 cm 踢面是平台低估造成的，实际该段最大 16.1 cm；硬障碍覆盖与 full_v1 差异大。训练地图应改为从 full_v1 导出（v9，未做）。
- **训练**：v8 地图那一轮最好 1.91 次协助/100 m（4 回合评估，规划器+完美定位 1.94，+航位推算 50.6）。之后 4090 反复掉卡，转 Colab。
- **4090 GPU 不可用**：Xid 79 共 4 次（11-17 03:56、14:20、17:27，11-19 15:44），最近三次都是单个训练几分钟内掉卡。需要检查供电/接线；在那之前不在它上面跑 GPU 负载。CPU 照常可用（不要同时加载多份全段场景，11-19 曾把 62 GB 内存吃满导致失联约 15 分钟）。
- WIP（未验收）：`wip/guard`、`wip/bev2`。

## 2. 阻塞项与资源

1. **训练在 Colab**：服务器 `~/.local/bin/colab`（google-colab-cli）已登录，余额约 2511 计算单元，可选 T4/L4/G4/H100/A100。用户 11-19 同意上传训练地图到 Colab 临时 VM（不用 Google Drive）。日志与权重拉回 4090 保存。
2. **Codex**：服务器账号额度到 11-22 19:07；期间用 Mac 本机 codex（gpt-6.2-sol）经 ssh 操作服务器（凭据不放共享服务器）。
3. **推送**：Mac 中转 `~/.cache/dreamer_nav_relay.git`；Mac 代理不通时借服务器代理（`ssh -f -N -L 17897:127.0.0.1:7897 ...`，并 `unset no_proxy NO_PROXY`）。
4. **网络**：服务器 Clash 订阅凭据会轮换；Codex 报 `workspace routing discovery failed` 时先查节点。

## 3. 已完成 [已验收]

| 内容 | 提交 | 验收证据 |
|---|---|---|
| 审计、三处修复、对照实验 | `5131a09` | 默认路径逐回合复现；增益 2 与 4 无差别；确定性动作 100/144，采样 89/144 |
| 清理 358.7 GB | — | 清单 `audit_20261116/cleanup_manifest.txt` |
| `CLAUDE.md`、`HANDOFF.md` | `75f5e78` 起 | — |
| `nav4` 核心（环境、传感器、跟踪器、规划器、策略包装、评估） | `850aa4a`、`f22abe9` | 干净检出 32 项测试通过；9 个留出窗口上规划器 140/144（旧 141）；特权泄漏 0 |
| 建图代码 v6（`nav4/mapping/`） | `b8b132b` | 31 项测试通过；主会话独立重算卡点 31 个站点、8 段（v5 是 161 个、34 段） |
| 无定位训练管线 | 见本次提交 | 82 项测试全过；默认行为逐回合不变（旧 Dreamer 72 行、规划器 72 行与旧结果完全一致）；`env.py` 的 diff 已逐行审读，`train_dreamer.py` 只审了结构 |
| 建图 v7 + `robot.obstacle_sampling` | `b01e0eb` | 85 项 env/mapping/train 测试通过；nearest 在冻结 v6 地图上 8 回合逐行一致；协助 168 次从 rows 复核 |
| 地图 v8（登阶能力阈值） | `3870564` | 111 项测试通过；v8 冻结地图上规划器定位协助 1.94/100 m、航位推算 50.6/100 m、严格 0/8（~5.3 m，跟踪航向滞后），均从 rows 重算 |
| 评估器：终点前协助算送过终点 | `e002266` | 10 项测试通过；定位规划器 v7 结果逐行不变；航位推算数字从 rows 重算 |
| MATRiX 仿真器 v1.1.2 | — | 三个分卷在服务器 `../matrix_sim/downloads/matrix_v1.1.2/`，服务器上 SHA256 校验通过；未解压、未运行 |

## 4. 进行中 / 未完成

### 4.1 A 段地图

**v7 [已验收]**：`data/trail_a/` = v7，只读副本 `data/trail_a_v7_frozen/`；v6 副本 `data/trail_a_v6_frozen/`（对照评估都用冻结副本）。

- 障碍层 = 格中心到最近身体高度带碰撞点的 XY 距离 − 2 cm 测距噪声；`distance_method=grid` 保留 v6 做法。
- 卡点：v6 31 站/8 段 → v7 11 站/2 段（用 v6 路线复算是 11 站/4 段）。95 m、616–619 m、664 m 已无卡站。
- 规划器（定位、linear、seed 914901、8 回合）：严格 0/8，都在 ~4.99 m 无支撑；协助 8/8 到达，2.92 次/100 m（v6 4.50），≤3 次协助完成 0/8。
- 剩余两处 [执行者报告]：5.4–6.8 m 有 7 站找不到整机有支撑的横偏；19.6–20.0 m 支撑可行横偏与避障横偏无交集。点云双侧碰撞点净距 1.43 m / 0.94 m，但这不是可支撑宽度。证据 `audit_20261116/nav4_v7/`、`data/trail_a/v7_diag/`。

### 4.2 无定位 Dreamer 训练管线 [已验收，CPU 范围]

- **观测**：`route_inputs=false`，去掉走廊和路点；`odometry_inputs=true`，把以下里程计量放进 `state` 的 0–6 位：
  - 累计距离 / 1000 m
  - 相对起点航向的 cos 和 sin
  - IMU 俯仰、横滚
- **误差模型**：距离比例误差 2%；航向随机游走 0.002 rad/√m，即 728 m 后约 3°；IMU 噪声 0.01 rad。
- **特权解码目标**：`priv_route`（真实路线走廊）和 `priv_progress`（进度、航向误差、横偏），只进解码器。
- **训练脚本**：`nav4/train_dreamer.py`，用官方训练循环，配置在 `nav4/config/train.json`：
  - 256 个环境，size12m，batch 16×64，train ratio 32，replay 20 万。
  - 每 10 分钟存档；每 30 分钟做 4 次整段评估，用确定性动作和协助模式。
  - GPU 空闲不足 10 GB 时以退出码 3 拒绝启动。
- **CPU 冒烟** [执行者报告]：
  - 8 个环境跑 35 次更新约 300 秒，约 8.7 秒一次更新。
  - 续训时更新计数接着走。
  - 新权重的特权泄漏为 0。
- **启动命令**：GPU 空出来后用，第一次要盯着看。

  ```bash
  cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3 && source audit_20261116/nav4_dev/env.sh
  tmux new -d -s nav4_train_gpu "$NAV3_PYTHON -m nav4.train_dreamer --logdir checkpoints/nav4_dreamer_s0 --seconds 14400 > logs/nav4_dreamer_s0.log 2>&1"
  # 续训（预算按累计训练时间计）：
  # $NAV3_PYTHON -m nav4.train_dreamer --logdir checkpoints/nav4_dreamer_s0 --seconds 28800 --resume
  # 评估新权重：python -m nav4.evaluate --method dreamer --checkpoint checkpoints/nav4_dreamer_s0/latest.pkl ...（会自动读取保存的环境配置）
  ```

### 4.3 关键发现：地图 + 航位推算行不通

评估器修正后（终点前协助计为送过终点，之前在 721.5 m 原地死循环，数字被放大）：

| 定位方式（规划器，协助模式，8 回合） | 每 100 m 协助 |
|---|---|
| 近似完美定位，v7 | 2.92 |
| 航位推算，v7 linear，seed 914901 | 61.8（旧口径 722） |
| 航位推算，v6 nearest，seed 20261116，预算 25000 | 67.0（旧口径 181） |

结论不变：没有精确定位时先验地图 + 航位推算不可用。证据 `audit_20261116/nav4_evalfix/`。

### 4.4 MATRiX 与登阶模式 [执行者报告，未运行验证]

证据在 `../matrix_sim/evidence/`。

- 独立运控 Robot MC v0.6.6 已下载校验，在 `../matrix_sim/downloads/robot_mc_v0.6.6/`。M1 对应 `ROBOT_TYPE=ZGWS`。
- 登阶模式：模型 `policy_stair`，状态 `RL_State_Stair`。
  - 进入：先处于行走状态，再通过二进制 UDP SDK 发 `control_mode=18`、`motion_mode=20`。
  - 退出：发其他 `motion_mode`。
  - 手柄没有入口。
- 指令结构体 40 字节，布局是从二进制调试信息还原的。
- `run_mc.sh` 第 19 行把 `ROBOT_TYPE` 写死成 `XG`，要改。
- 运控进程不需要 sudo。
- 待做：
  - 解压并起仿真器（`inside_mc=false`，需要 GPU）。
  - 实测 M1 速度响应，校准 `nav4` 的机体参数。
  - 试着导入扫描地形。

### 4.5 route-free 安全检查与 BEV 误报 [未验收，`wip/guard`]

- `wip/guard`（`3e13cc6`）：不读走廊和里程计的几何否决 + 候选回退。原 planner/hybrid 逐位不变。
- 用“恒定直行”基线测（冻结 v6，8 回合）：不加检查 20.5 次/100 m；加检查第 2 轮 38.7 次/100 m，碰撞 757→52、无支撑 426→168，但 87% 的步在刹停，平均速度 0.04 m/s。不合入。
- 根因诊断 [执行者报告]（`audit_20261116/nav4_obsfp/`）：规划器两个整段里，真值可行的当前足迹有 48% 的步被观测 BEV 判危险（逐格 2.6%）；0.8 s 预测路径 63%。障碍误报格里 67% 是地图类型 0，类型 2（软/稀疏/豁免）约 27%；去掉高度噪声只减少 14%。成因（记忆配准漂移、BEV 聚合、地面提取、遮挡）未拆清。
- 含义：任何拿这张 BEV 做硬否决的检查都会锁死；Dreamer 面对同样的观测。

### 4.6 BEV v2 [未验收，`wip/bev2`]

- 记忆每格存自洽的（地面候选、顶面、时间）整组覆盖；障碍参考与碰撞带同口径。v1 逐位不变，吞吐 0.98×。
- v8 地图同轨迹：安全足迹误报步率 33.7% → 29.1%（v7 时 v1 是 48%）；危险足迹漏报格 21.0% → 25.9%；未知增加；规划器协助 1.945 → 2.153/100 m。收益小、漏报上升，不合入。
- 推断：剩下的误报主要是“几何可见 ≠ 可碰撞”（真值按地图分类把软/稀疏植被、点数不足的回波判为可穿越），这是激光的真实歧义。Dreamer 可从奖励学习；硬否决式安全检查会被它锁死。是否给 BEV 加回波密度类通道需要单独论证（仿真里的密度要贴近 Airy）。

## 5. 下一步

1. Colab A100 上用 v8 跑通训练流程（`e03cfc4` 起，带 best.pkl），日志拉回 4090。
2. 从 full_v1 导出 v9 训练地图（多层支撑、亚格阶沿、机身高度带硬障碍、软植被、补面置信度；方案见 `full_v1/USAGE.md`），验收后在 Colab 重训。
3. 对比表（v9，同种子，协助模式）：规划器+完美定位、规划器+航位推算、无定位 Dreamer、Dreamer+软代价安全检查。
4. MATRiX 闭环：需要能用的 GPU（4090 修好后），步骤见 `full_v1/USAGE.md`。
5. 鲁棒性评估；部署（Orin、实机 BEV 节点与仿真管线一致）；裁剪旧代码。

## 6. 用户答复（2026-11-16 22:20）

1. GPU：等 `csy_homie` 训练结束后立刻接上我们的训练（已授权）。由 Codex（gpt-6.1-sol）在 tmux `nav4_gpuwatch_codex` 监视；启动参数读 `../nav4_train_spec.json`（现为 `a63e988` + 冻结 v6 地图 `data/trail_a_v6_frozen`，在独立 worktree `../exp_nav3_train_a63e988/` 里跑，不受项目根改动影响）。v7 验收后由主会话改规格文件。
2. 登阶模式：用户确认 M1 loco 能力足够通过 0.28 m 踢面和 47° 短坡。
3. 深度相机：Gemini 336L，装在 M1 头部，水平朝前。
4. 仍待用户判断：v7 之后若还有走不通的窄处，附点云净宽请用户判断现场能否通过。

## 6.1 执行方式变更（2026-11-16）

执行者改为 4090 上的 Codex（`codex exec`），主会话（Opus，Mac）规划和验收。任务单/日志/最后输出在 `../<任务>_prompt.md`、`../<任务>.log`、`../<任务>_last.md`。当前在跑：`nav4_v7`（地图 v7 + `robot.obstacle_sampling`）、`nav4_guard`（不依赖路点的几何安全检查 + `--method forward/guarded`）、`nav4_gpuwatch_codex`。
服务器 Clash 的“魔戒.net”订阅凭据 11-16 轮换过，旧凭据全部节点超时；已从用户 Mac 的新订阅替换（备份为 `*.bak.20261116-*`）。Codex 报 `workspace routing discovery failed` 时先查这个。

## 7. 容易踩的坑

- 旧代码有哈希门禁，文件清单见 `CLAUDE.md` 第 4.4 节；新代码不受它约束。
- XLA autotune 必须是 1。GPU 被占时用 `JAX_PLATFORMS=cpu` 跑测试和小评估。
- 运行环境：`audit_20261116/nav4_dev/env.sh` 是 nav4 的，`audit_20261116/audit_env.sh` 是旧代码的；两者都要在项目根 `source`。
- Dreamer 默认随机采样动作；评估要写明用的是 `--action mode` 还是 `sample`。
- 不要运行旧报告生成器 `nav3/lwy_report*.py`，它会截掉 `LWY_REPORT.md` 的后半部分。
- R8-P3 的微调是失败的，权重已删；R9 用的是 R8-P2 的权重 `checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl`。
- `git add` 只要有一个路径不存在就会整体失败。提交后用 `git show --stat` 确认内容（`850aa4a` 就漏过一次，由 `f22abe9` 补上）。
- 用 ssh 执行多行 Python 时，先把脚本写成文件再 `scp` 过去执行，不要把含引号的 heredoc 直接嵌在 ssh 命令里。
- 同一账号下有别人的 tmux 会话和进程，清理时只动自己的。
