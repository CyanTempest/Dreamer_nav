# M1 Stair 与 nav4 闭环

策略使用提交 `58a2872` 的 `nav4` 和 `dreamerv3`，模型结构及观测空间从 `best.pkl` 读取。每次正式评估从 0.75 m 出生。完整 A 段使用 frozen forward 地图的最后一道门作为终点。

## 运行目录

Colab 上的 `/content/nav` 放冻结源码，`/content/best.pkl` 放权重，`/content/collect` 放同一 worktree 的 `matrix_v4/collect/{sensors,control}.py`。`/content/val` 提供 full_v2 验证使用的 `fit_step.py`、站姿目录和 CustomWorld 包。MATRiX 安装在 `/content/matrix-v4`。将本目录复制到 `/content/nav_bridge`。

`/content/reference/route.npz` 使用 frozen map 中的 `s`、`route`、`width` 和 `gates`。`ground.npy` 是采集验证使用的沿线站号、横偏和地面高程表。这两份资料供裁判和复位使用。服务器实验目录 `research/matrix_nav/ops/` 保存本次实际使用的安装、传输、回收和预算脚本。

## 标定和评估

1. 在服务器 CPU 上以冻结源码及只读 `data` 为输入运行 `fit.py 42`，把生成的 JSON 上传为 `/content/cal_pose.json`。在 VM 中用 bridge 环境运行 `runtime.py`，其依赖路径包含 `/content/collect` 和发布包 `Tools`。输入固定的前进和双向旋转命令，记录 Odom 并将实测速率写入 `/content/eval/calibration/calibration.json`。
2. VM 的 nav4 环境使用冻结版 `nav4/colab/requirements.txt` 和 `eclipse-zenoh==1.10.2`。运行 `bash /content/nav_bridge/start_policy.sh`。该入口设置 JAX CUDA 平台及 22% 显存比例。
3. `drive.py` 在 5 Hz 物理时钟调用确定性动作头。20 Hz 心跳使用当前 Odom 重新求世界目标点在机体系的坐标，调用 nav4 跟踪律，将前进和偏航指令按标定缩放，由 Puerts 调用原生 Stair 控制器。向外发布的 JPEG 仅由 `record_camera.py` 接收。
4. 失败时 VM 发布 `request.json`。服务器服务运行 `fit.py`，按 1 m 前移目标检查站姿并将答案写入 `pose-NNNN.json`。随后冷启动 MATRiX、重新起身、清空点云记忆和策略状态。从起点累计的里程和初始航向延续。每段使用独立目录 `leg-NNNN`。
5. VM 中运行 `finalize.py` 汇总各段已编码的展示视频，保存 Stair 确认记录和 ffprobe 结果。服务器运行 `analyze.py RESULTS SEED914901_NPZ ASSEMBLY_JSON`，生成轨迹、协助表和五站 BEV 对比。

## 观测与记录

`perception.py` 依 XCDR1 字段偏移、行跨度和点时间解码前后 Airy。每圈分成十组，以 Odom 接收时间锚点插值位置和姿态，使用发布包的传感器安装旋转约定。世界格保存本次扫描的最低、最高回波及采样时刻。地面开运算只接受有回波的样本，地面基准优先取机体近邻测得高度的中位数，邻域完全空缺时用机体高度减 0.50 m。机器人盲区内不插入自身回波。已知格重算 nav4 支撑掩码，走廊通道恒零，8 s 后转未知。BEV 延迟一帧，里程计状态和上一动作保持当前步口径。

速度状态用约 0.10 s 的 Odom 位姿差分换算为水平机体系前进、侧移和偏航速率。标定中实测前进 0.458 m/s 时，ZGWS 发布的线速度第三轴为 0.458，第一轴为 −0.009。IMU 姿态按对应时间戳选择。

每个 `steps-NNNN.npz` 保存送入策略的完整六通道 uint8 BEV、13 维状态、原始 twist、IMU 四元数、原始动作、Odom 位姿及速度、最近归一化指令、决策仿真时刻与接收墙钟，以及用于复核的横偏、站号、离地高度和传感器年龄。跨段统计从各段内部计算行走距离，将搬运和冷启等待单列。
