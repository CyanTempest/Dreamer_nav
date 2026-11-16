# HANDOFF — 2026-11-16 21:40

接手前先读 `CLAUDE.md`（规范）。本文件只记当前状态，接手后更新它。

标注：**[已验收]** = 主会话独立核实过；**[执行者报告]** = 来自子代理、未复核；**[推断]** = 没有直接证据。

## 1. 现状

- 新仿真包 `nav4/` 的核心、建图代码 v6、无定位训练管线都已验收并合入 `main`。
- A 段地图仍有 5 处窄通道走不通。亚格精度修复（v7）做到一半时执行者中断，被跟踪的文件没有被改动。
- Dreamer 还没在 GPU 上训练过一步，因为 GPU 被别人占满。

## 2. 阻塞项

1. **GPU**：`csy_homie` 的训练（`HOMIE_ELF3`）占用 22.8 / 24.5 GB，从 17:41 持续到现在。训练脚本要求至少 10 GB 空闲才启动。需要用户协调。
2. **子代理不稳定**：18:36 出现 403 额度不足，21:07 和 21:12 出现网关 502，各中断过一次执行任务。遇到时按本文件第 4 节续做。
3. **推送**：服务器连不上 github.com，要经用户本机中转，见 `CLAUDE.md` 第 4.7 节。本机中转仓库已改用 HTTPS 加 gh 凭据，因为本机走 SSH 会在代理后卡住。

## 3. 已完成 [已验收]

| 内容 | 提交 | 验收证据 |
|---|---|---|
| 审计、三处修复、对照实验 | `5131a09` | 默认路径逐回合复现；增益 2 与 4 无差别；确定性动作 100/144，采样 89/144 |
| 清理 358.7 GB | — | 清单 `audit_20261116/cleanup_manifest.txt` |
| `CLAUDE.md`、`HANDOFF.md` | `75f5e78` 起 | — |
| `nav4` 核心（环境、传感器、跟踪器、规划器、策略包装、评估） | `850aa4a`、`f22abe9` | 干净检出 32 项测试通过；9 个留出窗口上规划器 140/144（旧 141）；特权泄漏 0 |
| 建图代码 v6（`nav4/mapping/`） | `b8b132b` | 31 项测试通过；主会话独立重算卡点 31 个站点、8 段（v5 是 161 个、34 段） |
| 无定位训练管线 | 见本次提交 | 82 项测试全过；默认行为逐回合不变（旧 Dreamer 72 行、规划器 72 行与旧结果完全一致）；`env.py` 的 diff 已逐行审读，`train_dreamer.py` 只审了结构 |
| MATRiX 仿真器 v1.1.2 | — | 三个分卷在服务器 `../matrix_sim/downloads/matrix_v1.1.2/`，服务器上 SHA256 校验通过；未解压、未运行 |

## 4. 进行中 / 未完成

### 4.1 A 段地图

**v6 状态**：输出在 `data/trail_a/`，已备份到 `data/trail_a/v6_before_subcell/`。

- 修复前的 v5 在 `data/trail_a/v5_before_fix/`。
- 保存的路线长 722.9 m，3615 个站点。
- 规划器仿真结果：严格模式 0/8，都在 5 m 处踩空；协助模式 8/8 到达，每 100 m 约 4.5 次协助。
- 剩下的 5 处窄通道：

  | 路线位置 | 两侧是什么 |
  |---|---|
  | 5.4–6.8 m | 台阶转弯处，一侧扶手立柱，一侧陡坎 |
  | 19–20 m | 左侧硬物，右侧土坎 |
  | 95 m | 0.65–0.78 m 高的密实物体，可能是石头或树桩 |
  | 616–619 m | 左侧扶手，后有 0.5 m 落差；右侧密实点 |
  | 664 m | 左侧 0.4 m 台坎，右侧悬挑物 |

  按 0.1 m 栅格算净宽 0.55–0.75 m，机体宽 0.58 m。

**v7 [未做完，要重新派]**

- 思路：一个格里有一个障碍点，整格就算占用，窄通道每侧平均损失约 5 cm。
- 任务：
  1. 障碍层改为"格中心到最近碰撞点的距离，减去实测测距噪声"。
  2. 报告里的足迹检查对障碍层用双线性采样。
  3. 加两根间距 0.62 m 的立柱的单元测试。
  4. 仍走不通的地方，从点云直接量出净宽。
- 仿真侧要配合：`nav4/env.py` 加配置键 `robot.obstacle_sampling`，取值 `"nearest"`（默认，必须逐位不变）或 `"linear"`；`linear` 时环境里所有障碍层的真值检查都用双线性采样，并配一个单元测试；`nav4/config/train.json` 里设为 `linear`。这一项也还没做。
- 可写路径：建图部分只动 `nav4/mapping/`、`nav4/tests/test_mapping_*.py`、`data/trail_a/`；仿真部分只动 `nav4/env.py`、`nav4/config/`、新测试。两部分交给同一个执行者，或者先后做，避免同时改 `env.py`。

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

### 4.3 关键发现：地图 + 航位推算行不通 [执行者报告]

规划器在 A 段整段上的结果（协助模式，8 次，用同一张地图）：

| 定位方式 | 每 100 m 协助次数 |
|---|---|
| 近似完美定位（原有的小随机游走） | 4.1 |
| 只靠航位推算，用上面的误差模型 | 181 |

结果文件：`audit_20261116/train_dev/p2_planner_forward_assist_*.json`。这支持用户的判断：没有精确定位时，拿先验地图配航位推算不可用，要靠在线感知导航。该误差模型下几十米后横向误差就超过路宽（**[推断]**）。

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

## 5. 下一步

1. 地图 v7 加上 `robot.obstacle_sampling`（4.1），然后重跑规划器整段评估。
2. GPU 空出来后开始无定位 Dreamer 训练（4.2），先跑约 4 小时，看每 30 分钟的整段评估曲线。
3. 对比表，统一在 A 段整段、同种子、协助模式下评估：
   - 规划器 + 完美定位（上限参考）
   - 规划器 + 航位推算
   - 无定位 Dreamer
   - 无定位 Dreamer + 规划器式几何安全检查（需要一个不依赖路点的版本）
4. 鲁棒性评估：植被扰动、随机障碍、传感器噪声、关闭深度相机。
5. MATRiX 闭环和机体参数校准（4.4）。
6. 部署：导出策略到 Orin；写实机节点（激光鸟瞰图、SDK 指令、切换登阶模式）。
7. 裁剪 `nav3/` 等旧代码。

## 6. 等用户回答

1. GPU 时段怎么协调，或者有没有别的机器。
2. 地图里量到一级 0.28 m 的踢面和 1 m 尺度 47° 的坡，登阶模式确认能过吗？
3. 比赛时 M1 是否装 Gemini 深度相机，装在哪。
4. 如果 v7 之后仍有走不通的窄处，需要用户判断现场 M1 能否通过。届时会附上从点云量出的净宽。

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
