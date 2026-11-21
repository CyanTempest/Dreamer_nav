# ATEC M1 原赛道 100 米自主导航 v015

v015 使用 ATEC 香港赛道的 v009 场景，保留入口台阶、林间弯道和排水沟。MuJoCo 运行 M1 的轮足物理，DreamerV3 根据深度、本体状态和局部目标给出速度与转向，MATRiX Robot MC v0.6.6 在隔离的本地仿真环境中执行动作。局部目标由已知路线中心线和仿真位姿生成。

## 正式验收

固定种子为 6001–6010。每种动作模式每阶段评测 10 次，十阶段全部达标，200 次中 **194 次无禁止碰撞到达**。

| 阶段 | 均值动作 | 采样动作 | 每种模式门槛 |
| --- | ---: | ---: | ---: |
| 原赛道 100 米，无深度噪声 | 10/10 | 10/10 | 8/10 |
| 原赛道 100 米，有深度噪声 | 10/10 | 8/10 | 8/10 |
| 50 米 | 10/10 | 9/10 | 8/10 |
| 台阶段 20 米、15 米、10 米，各阶段 | 10/10 | 9/10 | 8/10 |
| 6 米、3 米，各阶段 | 10/10 | 10/10 | 8/10 |
| 55–58 米弯道、65–68 米平地，各阶段 | 10/10 | 10/10 | 9/10 |

[验收摘要和 200 次逐项结果](evidence/acceptance.json)记录了原始验收文件及录像的 SHA-256。正式影片来自带噪均值模式的种子 6001，1280×720、5 fps、1,996 帧、399.2 秒。[逐帧里程索引](evidence/accepted_video.frames.csv)对应原版影片。第二处台阶的跟随镜头有局部前景遮挡。原版录像画面含比赛 3DGS 场景，公开分支提供录像的规格与校验记录。

终点判定沿用验收口径，沿路线里程在目标前 0.15 米以内、平面距离小于 0.25 米。正式录像回合的平面终点误差约 0.146 米。结果适用于仿真。

## 训练与运行参数

| 项目 | v015 实际设置 |
| --- | --- |
| 模型和训练设备 | DreamerV3 `size1m`，JAX CUDA float32，随机种子 31 |
| 模型结构 | RSSM 确定性状态 512 维，32 组离散状态，每组 4 类，想象长度 15 步 |
| 优化 | 学习率 `4e-5`，恒定调度，配置预热 1,000 步，自适应梯度裁剪 0.3 |
| 离线补训 | 从 v014 的 90,840 次更新起补训 4,800 次，累计 95,640 次。432 条训练序列、96 条留出序列。补训在线交互 0 步 |
| 批次 | 4 条序列 × 64 帧，模仿学习预热 16 帧。每 4 次更新取 1 个专项批次 |
| 观测与动作 | 64×64×3 米制深度、64 维本体状态和 3 维局部目标。前进 ±0.5 m/s、转向 ±0.3 rad/s |
| 运行频率 | 导航动作 5 Hz，MuJoCo 物理和禁止碰撞检查 500 Hz |
| 奖励 | 进度权重 10，成功 +30，碰撞与摔倒各 −20，横向偏差权重 1.5，净距权重 2.0，净距阈值 0.04 m |

[有效模型配置](weights/v015/config.yaml) · [环境配置](weights/v015/integration.json) · [补训参数和实际命令参数](configs/v015_training_parameters.json) · [最终权重](weights/v015/final/agent.pkl)

## 复现最终策略

在 Linux x86_64 上使用 Python 3.11、可用的 NVIDIA CUDA/EGL 环境、`g++`、`git`、`ripgrep`、`iproute2` 和支持非特权用户命名空间的 `unshare`。MATRiX v0.6.6 官方运行包使用 Ubuntu 22.04 依赖。以下命令在本目录执行。

### 1. 安装 Python 环境和上游代码

```bash
python3.11 -m venv .venvs/scene
python3.11 -m venv .venvs/dreamer-gpu
.venvs/scene/bin/python -m pip install -r configs/requirements-scene.txt
.venvs/dreamer-gpu/bin/python -m pip install -r configs/requirements-dreamer-gpu.lock.txt

mkdir -p third_party/runtimes
git clone https://github.com/danijar/dreamerv3.git third_party/dreamerv3
git -C third_party/dreamerv3 checkout e01491fad6434b2245a3b8ca201dd7faedcc458c
```

### 2. 准备运行包与本地资产

- 从 [MATRiX Robot MC v0.6.6](https://github.com/GENISOM-AI/MATRiX_Robot_MC/releases/tag/v0.6.6) 下载 Linux x86_64 运行包并校验 `SHA256SUMS`，解包到 `third_party/runtimes/MATRiX_Robot_MC-v0.6.6/`。在运行包目录下运行下列命令，将包内依赖展开到本地。仿真由本项目的 `run_matrix_isolated.sh` 启动。
- 将自己可使用的 v009 场景和 M1 模型放到或软链接到下面列出的路径。资产 SHA-256 见 [input_sha256.json](evidence/input_sha256.json)。比赛原始扫描及派生场景按 [ATEC 数据集使用条件](https://huggingface.co/datasets/atec999/atec-opendataset#授权说明--permissions)获取。原始素材、第三方模型与训练回放在本地提供。
- 对于需要准备场景的参赛队员，`build_competition_route.py` 使用获准使用的原始赛道与密集 LiDAR 裁剪生成路线和高度场。`complete_atec_scene.py --ground-relative` 生成外观及障碍，`compact_scene_collision.py` 压缩碰撞盒，`refine_surface_collision.py` 进行 2 cm 局部细化。v009 的最后一次细化使用 v008 场景和 49.55–50.55 米区间。验收所用的场景版本以文件校验值为准。

```text
assets/terrains/competition_100m/v009/
  manifest.json  centerline.csv  terrain.npz  obstacles.npz  visual.obj  splats.npz
assets/robots/m1/mujoco/v001/
  manifest.json  robot.xml  meshes/
third_party/runtimes/MATRiX_Robot_MC-v0.6.6/
  build/export/  deps/  local_deps/
```

```bash
(
  cd third_party/runtimes/MATRiX_Robot_MC-v0.6.6
  mkdir -p local_deps
  for package in deps/*.deb; do dpkg-deb -x "$package" local_deps; done
)
bash scripts/build_matrix_bridge.sh
bash scripts/reproduce_v015.sh check
```

### 3. 运行短程、全程和录像

```bash
# 65–68 米短程物理验证
bash scripts/reproduce_v015.sh smoke outputs/smoke

# 100 米带噪均值动作，种子 6001
bash scripts/reproduce_v015.sh course outputs/course100

# 该回合成功后，用原生 3DGS 与保存的物理状态导出录像
bash scripts/reproduce_v015.sh video outputs/course100

# 重跑十阶段固定队列。全部阶段通过后，评测入口会导出本轮成功录像
bash scripts/reproduce_v015.sh suite outputs/formal-suite
```

`video` 和 `suite` 的录像步骤还需要 CUDA 编译工具链、CUDA 版 PyTorch 和 `gsplat`。原环境使用 PyTorch 2.7.0 + cu128、gsplat 1.5.3。将 `CUDA_HOME` 指向本机工具链。`suite` 的录像子进程读取 `weights/v015/integration.json` 的 `recording.cuda_home`，默认 `/usr/local/cuda`。新回合的物理状态、指标和 MP4 保存在所选输出目录。原始的最终验收脚本及其完整证据核验还需要原赛道数据、场景父链、原始参照录像和全套物理记录。

## 重跑离线补训

从 v014 重跑补训还需自行提供 `replay_inputs/v014/` 中的起始检查点和对应配置、`replay_inputs/v015/` 下与原划分一致的回放清单和序列，以及其声明的父场景。训练及验证回放实测约 1 GiB。输入 TOML 中的序列长度由补训命令的 `--batch-length 64` 覆盖，训练脚本设置随机种子 31。使用 [补训命令参数](configs/v015_training_parameters.json)中的 `command_argv`，把输出放到新目录。最终权重可直接用于上面的评测命令。

发布前在单独检出目录使用本机已安装的依赖和获准使用的资产执行了 `check --video` 与 `smoke`。短程回合成功，记录进度 2.889/3.0 米，距目标 0.163 米，未记录禁止碰撞。
