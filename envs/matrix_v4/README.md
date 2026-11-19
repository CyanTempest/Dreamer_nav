# matrix-v4 环境与私有输入

安装目标为已有的 Linux x86_64 GPU VM。Colab 根目录默认为 `/content/matrix-v4`。Docker 使用同一套安装脚本和版本锁。准备完成后，代码在 `nav/`、`mmbench2/`，MATRiX 在 `MATRiX_v1.1.2/`，私有地图与 V3 权重在 `inputs/{data,checkpoints}/`。Python 锁覆盖 Linux 上运行时的传递依赖。系统包取对应 Ubuntu 发行版的签名仓库，宿主驱动按实际 VM 匹配。

## 三个进程

| 进程／解释器 | 版本与职责 |
|---|---|
| `venvs/bridge/bin/python` | Python 3.11.18，eclipse-zenoh 1.10.2、NumPy 1.26.4、MuJoCo 3.3.0、无 GUI 的 OpenCV。负责收包、数据/校验和动作传输。发布版自带物理库，无需给 UE 链接 Python MuJoCo。 |
| `venvs/v3/bin/python` | Python 3.11.18，JAX/jaxlib/CUDA 插件 0.4.33、Ninjax 3.5.1。64 个包以服务器 `../envs/dreamer` 的 `uv pip freeze` 为约束解析，保留了从源码安装的 elements 等包。项目从 `nav/` 导入内置 DreamerV3。 |
| `venvs/v4/bin/python` | Python 3.10.21，torch 2.8.0、torchvision 0.23.0、CUDA 12.8 用户态库，按 MMbench2 的训练依赖裁剪，保留 tensordict、wandb 的导入依赖。只拉取固定提交 `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`；模型模块位于 `mmbench2/src`。 |

V3 与 V4 采用独立进程和 venv。同名 NVIDIA 包在两套锁里有互斥的版本（例如 cuDNN 9.1 与 9.10），一个解释器无法同时按两份锁安装。bridge 和教师用本机 IPC 交换带 episode/step ID 的观测与动作；V4 离线读关闭的分片。采集时先用 CPU 教师，启动环境设 `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false`，构造 `DreamerPolicy` 时覆盖 `jax_overrides=dict(platform="cpu", autotune=1, deterministic=True)`，按权重里的 obs_space 加载。若 CPU 达不到 5 Hz，测独立 JAX GPU worker 的延迟和与 UE 共存时的显存；沿用 `XLA_FLAGS="--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true"`。训练阶段退出采集实例，给 PyTorch 独占 GPU。

MMbench2 的 tokenizer 和 dynamics 入口显式指定在线 W&B。安装只取源码和依赖。将本项目的训练入口改为本地记录，移除两处在线初始化、图像/权重上传，并关闭 BEV 的 LPIPS 后，才给训练器读取比赛数据。完整训练改动见 [V4 调研](../../docs/matrix_v4/V4_SURVEY.md)。

## 4090 上的私有包

初始包已在本 worktree 的 `data/matrix_v4_transfer/`。归档只含私有运行子集，包括 full_v1 的组合版 XML/OBJ、两份候选配置、assembly、配准元数据、净宽表、路线，以及 V3 的 `latest.pkl` 与配置。独立 MuJoCo 场景、建图中间物、巡游视频和 replay 可以留在 4090。路线由建图采用的同一个冻结 v8 中心线重采样，输出扫描坐标系的 `station_m, xy_m`。制作新快照时从服务器的 matrix-v4 worktree 执行以下命令。

```bash
../envs/dreamer/bin/python envs/matrix_v4/private_assets.py pack ../exp_nav3 data/matrix_v4_transfer
(cd data/matrix_v4_transfer && sha256sum -c matrix-v4-private.tar.gz.sha256)
```

## 在已分配的 Colab GPU VM 上安装

前提是主会话已经把本分支推到项目 GitHub，并明确提供已在运行的会话名。脚本沿用该名称接收资产。以下为后续使用命令。在 4090 的 worktree 中，以小块传输避免 Colab Contents API 把 210 MiB 的包做成一次 Base64 请求。远端默认收件目录为 VM 的 `/content`；更新另一份快照时使用新的私有收件目录，避免混用旧块。

```bash
set -euo pipefail
: "${SESSION:?请指定已有会话名}"
cli="$HOME/.local/bin/colab"
dir=data/matrix_v4_transfer
bundle=matrix-v4-private.tar.gz
parts=$(mktemp -d "$dir/parts.XXXXXXXX")
split -b 8M -d -a 3 "$dir/$bundle" "$parts/$bundle.part-"
for part in "$parts"/*; do
  "$cli" upload -s "$SESSION" "$part" "content/$(basename "$part")"
done
"$cli" upload -s "$SESSION" "$dir/$bundle.sha256" "content/$bundle.sha256"
```

在同一台 VM 的 Bash 终端运行以下安装命令（Colab 单元格首行可用 `%%bash`）。`MATRIX_SCENE=window_00` 选择首个 100 m 窗口；省略时安装完整 A 段。可用 `MATRIX_INBOX` 和 `MATRIX_V4_ROOT` 覆盖收件目录及工作根。也可直接传同名完整归档和校验清单。

```bash
set -e
curl -fsSL https://raw.githubusercontent.com/CyanTempest/Dreamer_nav/matrix-v4/envs/matrix_v4/colab_setup.sh -o /content/colab_setup.sh
bash /content/colab_setup.sh
```

在 Colab 默认 root shell 下安装系统库和固定版本 uv 后，脚本将后续工作交给无特权用户 `matrixv4`。下载三个 v1.1.2 分卷（合计 4,991,256,614 B），固定发布清单 SHA256，逐卷校验，按 001→003 流式解压。失败可重跑。私有数据到齐较晚时，用 `bash /content/colab_setup.sh assets` 单独接收；修改场景时停掉对应实例，再用 `MATRIX_SCENE=window_00 bash /content/colab_setup.sh assets`。脚本保留现有 nav checkout 的修订，记录拉到的 SHA；mmbench2 校验固定 SHA。

私有归档通过 SHA256 后才解包到按哈希命名的目录。脚本保留发布版 ZGWS 的 STL，从候选配置合入路线场景，明确设置 `inside_mc=true`、`mujoco_running=false`。活动传感器只有双 GPU Airy、IMU 和 Odom。两路点云均使用各自挂载位姿。输出里 `inputs/checkpoints/nav4_dreamer_v8_s0/latest.pkl` 是传给教师的路径。

图形准备阶段先测试宿主的 NVIDIA Vulkan 设备及 OptiX 库装载。若 VM 只带计算库，从 NVIDIA 官方 HTTPS 端点下载**与宿主版本相同**的 Linux 驱动归档，校验厂商 SHA256，以普通用户解包所需 EGL/OptiX 用户态库，生成仅用于本实例的 EGL ICD。后续仿真进程以 `matrixv4` 身份 `source /content/matrix-v4/graphics.env`。设备探测日志为 `vulkan-summary.log`。物理/传感器开机后另验双路实收点云、场景命中及登阶状态。临时盘需容纳两套 CUDA wheel、下载缓存、分卷、约 6.2 GiB 发布目录及轨迹；按 60 GiB 可用空间起估，另给录制分片预留空间。拉回 4090 的校验和续传设计见 [采集设计](../../docs/matrix_v4/DESIGN.md)。

## 其他 NVIDIA 宿主上的 Docker

以 `envs/matrix_v4/` 为唯一 build context，从而只发送环境文件。镜像固定 NVIDIA CUDA 12.8.1 Ubuntu 22.04 amd64 digest。构建只安装系统和 Python 依赖。启动准备容器时才获取代码、发布包和只读挂入的私有输入。宿主需已配置 NVIDIA Container Toolkit，并传入 compute、utility、graphics、display 能力。以下命令留给有相应宿主的后续会话。

```bash
docker build --platform linux/amd64 -t atec-matrix-v4 envs/matrix_v4
docker run --rm --gpus all \
  -v matrix-v4-runtime:/opt/matrix-v4 \
  -v "$PWD/data/matrix_v4_transfer:/inputs:ro" \
  atec-matrix-v4
```

让镜像内 `matrixv4` 的 UID 或 GID 对只读挂载具备目录遍历和文件读取权，可用宿主 ACL 单独授权。首次使用空的 named volume 时 Docker 会复制镜像中该目录的 venv。容器退出后产物保存在该卷。软件探测及安装不会创建 Colab 会话；本次只完成服务器 CPU 静态核验，未构建镜像。

## 本次在 4090 CPU 上核对的结果

发布方 HTTPS 清单的 SHA256 与本地副本一致，三个本地分卷逐个通过校验。私有包为 219,638,588 B。用真实发布包的配置和 STL 在临时目录执行接收，再按 full、window_00、full 切换。递归检查入口 XML 的 include 和文件引用，全段的 13,783 个资源引用与首窗的 992 个均存在；路线有 14,441 个有限 XY 点，终站 722 m；接收后的 checkpoint 与源文件逐字节相同。三份 Python 锁均通过 Linux 依赖解析，两个 Bash 脚本通过 ShellCheck 与语法检查。Dockerfile 做了解析及 amd64 基础镜像 digest 核对。目标 GPU 上的完整安装与正式采集留待闭环验收。
