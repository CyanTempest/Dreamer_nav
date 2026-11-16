# CLAUDE.md — ATEC2026 A 赛段导航（Dreamer_nav）

本文件是本仓库的长期规范。当前进度、阻塞项和下一步见 `HANDOFF.md`，每次交接时更新它。规范与用户在对话里的最新指示冲突时，以用户为准，并回头改本文件。

## 1. 任务（用户 2026-11-16 确定）

- **机器人**：智身 M1 轮足（厂商型号 zsm-1w），碰撞包络 1.03 × 0.58 m。前后各一台 RoboSense Airy 96 线半球雷达，另有 Gemini 深度相机。
- **目标**：只走官方扫描赛道的山路部分（A 段），顺采集轨迹的方向，从头走到尾。没有打卡点坐标，"走完全程"就是任务。
- **方法**：用 DreamerV3 世界模型训练导航策略，训练地图来自官方 3D 扫描。同条件下与规则规划器比较，谁效果好用谁。
- **交付**：既要仿真成功率，也要实机可部署。代码 2026-12-03 封存。
- **无定位**：实机没有地图坐标系下的精确定位，不用 RTK。策略输入里不能有依赖地图定位的量（路线走廊、路点、沿线进度）。真实路线只用于奖励、评估和世界模型的特权解码目标。
- **感知**：以双 Airy 激光为主；深度相机可选，训练时随机关闭；不用 RGB。理由是 12 月现场外观会变，几何比外观稳。
- **底层运动**：全程使用 M1 官方 loco 策略的登阶模式（MATRiX 仿真器 + MATRiX_Robot_MC 运控）。
- **地形**：用户声明 A 段地形对 M1 全部可通行。这只针对地形本身，路旁的硬障碍（树干、石头、栏杆、墙）仍按碰撞处理。

## 2. 环境

- **服务器**：`ssh hpf@100.108.114.64`（主机名 elf，Ubuntu 22.04，RTX 4090 24 GB，32 核，62 GB 内存）。同一账号多人共用。
- **项目根**：`/home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3`。代码直接在服务器上改。旧文档里说的"Mac 工作区为准"已作废，不要运行 `scripts/sync.sh`。
- **Python**：在项目根执行 `source scripts/env.sh`，解释器是 `$NAV3_PYTHON`（`../envs/dreamer/bin/python`，3.11 + JAX）。另有 `.venv_lwy`（numpy/scipy/shapely）和 `.venv_r8`（JAX + warp）。临时工具用 `~/.local/bin/uvx`，不往已有环境里装包。
- **JAX 必须带的设置**：`XLA_FLAGS="--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true"`。autotune 为 0 时本机上 Dreamer 数值会坏（第 6 轮的事故）。GPU 被占用时用 `JAX_PLATFORMS=cpu` 跑测试和小评估。
- **官方数据**：三个压缩包在 `/home/hpf/atec_opendataset/`；已解压的规划点云在 `../official_atec/e5a3b987/extracted/01_LiDAR_point_cloud/ds_map/plannar_map.ply`。轨迹 CSV、两份 PLY 和 2D 导航图在同一局部坐标系，没有经纬度基准。
- **MATRiX**：`../matrix_sim/`（下载包、解压内容、调查证据）。
- **GitHub**：`https://github.com/CyanTempest/Dreamer_nav`，公开仓库。服务器连不上 github.com，推送要经用户本机中转，见第 4.7 节。

## 3. 仓库布局

| 路径 | 内容 | 状态 |
|---|---|---|
| `nav4/` | 新仿真包：环境、规划器、策略包装、评估、`mapping/` 建图 | 主线，新代码都写在这里 |
| `data/` | 官方扫描的衍生地图、图、报告 | 不入库 |
| `nav3/`、`r8_p2_runner/`、`r9_official_runner/`、`scripts/`、`tests/`、`vendor/` | 第 3–9 轮旧代码 | 只读参考，等 nav4 落地后裁剪 |
| `dreamerv3/` | 上游 e01491f 的打过补丁的副本 | 不改；补丁内容见 `nav3/dreamer_adapter.py` 文件头 |
| `audit_20261116/` | 审计证据、对照实验脚本和结果、清理清单 | 留档 |
| `checkpoints/`、`maps*/`、`metrics/`、`fullgeo/` | 旧权重、旧地图、旧指标 | 不入库 |

## 4. 工作规范

### 4.1 分工
- 主会话（Fable）负责澄清需求、规划、写任务单、验收。Opus 子代理（Agent 工具，`model: "opus"`）负责执行：探索、编码、跑实验。
- 任务单必须写明：目标、可写路径的白名单、禁止事项、验收测试、要汇报的数字。子代理看不到对话，任务单是它的全部背景。
- 两个子代理并行时，各自的可写路径不能重叠。

### 4.2 验收
- 子代理的报告只是线索。合入 `main` 之前，主会话必须自己做这几件事：读 diff；重跑测试；从原始结果文件独立重算关键数字；确认白名单之外没有文件被改。
- 没通过验收的代码不进 `main`。需要异地留存时提交到 `wip/*` 分支，并在 `HANDOFF.md` 标明未验收。
- 改动默认行为之前，先证明旧结果可以逐回合复现；改完说明哪些数字因此不可比。

### 4.3 证据与汇报
- 报告里的每个数字都要能追到保存的结果文件。区分三类：实测的、从文件里读到的、推断的。
- 失败、退步、没做成的事要直接写出来。不为凑结论调阈值，每个阈值要有物理依据。
- 样本少时写清楚样本量和相关性（例如 9 张图上的回合彼此相关），不把提示性差异写成结论。

### 4.4 代码
- 新代码写在 `nav4/`，不 import `nav3`。参数只放在 `nav4/config/*.json`，别处不写死。
- 同一个物理规律只实现一次。跟踪律在 `nav4/tracker.py`，环境、规划器、过滤器共用。
- 每个行为改动配一个 CPU 可跑的单元测试，用合成地图。
- 旧代码有哈希门禁：下列文件改一个字节，旧的训练和评估入口都会断言失败。除非要重签门禁，否则不要动它们。
  - `nav3/env.py`、`nav3/bank.py`、`nav3/control.py`、`configs/env.json`、`configs/calibration.yaml`、`metrics/map_manifest.json`
  - `nav3/lwy_env.py`、`lwy_env_v2.py`、`lwy_bank.py`、`lwy_import.py`、`lwy_control.py`、`lwy_control_v2.py`、`lwy_eval_v2.py`、`lwy_geometry.py`、`configs/lwy_v2_tracker.json`、`configs/lwy_v2_planner.json`
- 策略输入与特权信息严格分开：真值只能出现在 `extra` 和 `priv_*` 键里。每次改观测都要跑特权泄漏检查。

### 4.5 评估
- 以 A 段整段为单位。成功的定义是到达终点且同一步没有失败。
- 同时报告：0 次协助完成率、≤3 次协助完成率（比赛每段允许 3 次人工协助）、每 100 m 协助次数、失败类型和位置。
- 各方法用相同地图、相同种子，在重置时配对。Dreamer 默认会随机采样动作，评估时写明用的是采样还是确定性动作。
- 鲁棒性单独评估：在改动过的地图上测（植被扰动、随机障碍、传感器噪声、里程计漂移）。
- 规划器始终作为对照一起跑。带定位的规划器是上限参考，实机做不到，表里要注明。

### 4.6 数据红线
- 官方数据集禁止再分发。原始数据和一切衍生物（地图、点云裁剪、能看出赛道形状的图）只放在 `data/` 或其他被忽略的目录，不提交。
- 仓库是公开的。密码、令牌、下载链接不写进任何被跟踪的文件。提交前检查暂存区里的大文件和密钥。

### 4.7 git
- 每次验收通过后提交。提交信息说明改了什么、为什么、验证了什么。
- 推送经用户本机中转（本机已登录 gh，服务器连不上 github.com，仓库禁用了 deploy key）：
  ```bash
  cd ~/.cache/dreamer_nav_relay.git
  git fetch server main:refs/heads/main && git push github main:main
  ```
  本机连不上 GitHub 时，提交先留在服务器，在 `HANDOFF.md` 记一笔。
- 令牌只留在用户本机，不放到共用的服务器账号上。

### 4.8 共享服务器
- 启动 GPU 作业前先看 `nvidia-smi`。别人的进程不停、不降优先级。GPU 被占满时不硬挤，改 CPU 或告诉用户协调。
- 长任务放 tmux，会话名以 `nav4_` 开头，日志写文件。结束时只清理自己的会话。
- sudo 只在用户明确同意的事项上用，用前先看清要装什么，确认不动显卡驱动和别人在用的库。sudo 密码不写进仓库。

### 4.9 删除
- 删除前逐项确认没有被在用代码引用，列出清单和大小，再删，并留下清单文件。先例见 `audit_20261116/cleanup_manifest.txt`。
- 各轮有报告结果的最终权重和指标保留。

### 4.10 第三方二进制
- MATRiX 和 Robot_MC 是未经审计的二进制。每个包放独立目录，校验发布的 SHA256 之后再解压，启动脚本先读后跑，不用提权运行。
- 运控的 SDK 没有公开头文件，字段布局是从二进制调试信息还原的。使用前先在仿真里实测确认，不能直接上实机。

### 4.11 必须先问用户
- 需求或指标的含义不明确时。
- 涉及实机、sudo、公开仓库里的敏感内容、删除有报告结果的数据时。
- 需要占用共享 GPU 的长时间训练，而 GPU 上有别人的作业时。
- 其余有常规默认做法的事直接做，在汇报里说明选了什么。

## 5. 常用命令

```bash
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh                     # 必须在项目根执行，它用 $PWD
source audit_20261116/audit_env.sh        # 同上，并把缓存重定向到审计目录，不改已有缓存文件

# R9 单元测试（7 项，CPU）
cd r9_official_runner && PYTHONPATH=. ../.venv_lwy/bin/python -m unittest discover -s tests

# 旧基线复现：规划器 18/18、Dreamer 12/18、接管 15/18
bash r9_official_runner/run_holdout.sh

# nav4 的测试、评估命令见 nav4/README.md
```
