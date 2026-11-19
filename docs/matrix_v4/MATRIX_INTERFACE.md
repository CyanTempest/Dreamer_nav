# MATRiX v1.1.2 与 A 段采集器的接口

本文依据服务器的 Linux 发布包做静态核对。「发布包」表示读到的配置、说明或 Python 工具，「历史证据」表示旧调查记录，「建议／待测」表示尚需在目标 VM 验证。本文没有启动仿真或采样。发布包标识见 `RELEASE_BUILD.json:2–10`。

路径简写均相对服务器的 `research/` 目录。未加前缀的是 `matrix_sim/extracted/matrix_v1.1.2/MATRiX_v1.1.2/` 内路径。下文 `cfg/` 指该包内 `UeSim/Content/model/config/`，`F/` 指 `exp_nav3/data/trail_a_mujoco/full_v1/`，`FC/` 和 `FZ/` 分别指 `F/matrix/UeSim/Content/model/{config,zgws}/`，`E/` 指 `matrix_sim/evidence/`。行号按文件从 1 开始。

## 启动和图形后端

- **发布包**提供离屏参数。Pixel Streaming 示例把 `-RenderOffScreen` 传给 `./UeSim.sh`，脚本在包根目录用 `"$@"` 透传参数。仅需 Zenoh 时可先试 `./UeSim.sh -RenderOffScreen -ResX=640 -ResY=480`。此精简组合和分辨率是建议值。文档中的 Pixel Streaming 地址只用于串流。证据见 `docs/pixelstreaming_tutorial.md:76–97`、`docs/USAGE_CN.md:14–24`、`UeSim.sh:4–18`。
- **发布包**要求 Linux x86_64 和工作的显卡驱动，建议至少 6 核、16 GB 系统内存。Vulkan 启动错误写入 `UeSim/Saved/Logs/UeSim.log`。Airy GPU 路径另需 NVIDIA CUDA/OptiX、随包的 `libAiryOptix.so` 与 PTX。初始化失败可能回落到 CPU，运行中失效可能只停发一个传感器。资料没有给出此包的最低 Vulkan API／NVIDIA 驱动版本或 A100 配置实测。证据见 `docs/USAGE_CN.md:5–11,159–167`、`docs/SENSORS_CN.md:36–46`。
- **目标 VM 待测**。在有授权的后续会话里检查非 root UeSim 进程能否使用 Vulkan loader 和 NVIDIA ICD、枚举硬件物理设备、建立图形队列，随后读 UE 选中的适配器、两台 Airy 的实际后端和持续到达的帧。CUDA 可见不能代替 Vulkan／OptiX 检查。发布说明要求进入场景并创建机器人，默认 `mujoco_running=false`，提供离屏画面尚不能证明从冷启动可无人进场。单独测「进程启动 → 机器人生成 → 首对点云 → 运动响应」。证据见 `cfg/config.json:6–19`、`docs/USAGE_CN.md:74–95`、`docs/SENSORS_CN.md:40–48,62–64`。

### 另一个执行者的 A100 冒烟记录（只读）

在 `research/matrix_v4_smoke/` 的同一台 Colab A100-SXM4-80GB 上，宿主驱动为 580.82.07。默认离屏启动在 Vulkan SM6 profile 检查报 `VK_ERROR_INCOMPATIBLE_DRIVER`。用匹配版本的 NVIDIA 用户态库、EGL ICD（`libEGL_nvidia.so.0`）和无特权用户后，`vulkaninfo` 枚举到 A100，UE 的 Vulkan RHI 选中 A100，CUDA context 创建成功，`/Game/Maps/MainWorld` 载入完成。此轮导入的是发布自带的 ZGWS，日志计 17 个机器人网格、0 个场景网格。分组主时钟记为 500 Hz，采用 `drop_whole_group` 且 `require_realtime=false`。这个设置仍需用连续时间戳计算实际速度与丢组率。出处为该目录 `logs/vm_probe.log:4–11,69–77`、`logs/default_scene_console.log:738–751`、`logs/vulkan_probe_output.log:35–67`、`scripts/egl_scene.sh:6–10,16–18`、`logs/egl_scene_console.log:812–824,1322–1340,1365–1367,1407–1414`。

同一冒烟工作的初次运控试探里，端口清单未显示 UDP/TCP 7447 监听；示例发送了起身、行走、旋转，监控器却未能连接本地 Zenoh。此样本未取得可验证的动作响应或点云，见 `logs/initial_inputs_output.log:3–20,28–36`（相对上述冒烟目录）。

发布配置的 `mujoco_running=false` 应保留。另一执行者保存的诊断字符串将 true 与外部 MuJoCo 接管、禁用内置硬件/运控联系起来。进场和重置函数名出现在其调查记录，尚无适用于 Python 采集器的消息契约。出处为 `logs/runtime_options.log:46,157`、`logs/spawn_functions.log:12–26`；这些是静态线索。

## Zenoh 订阅与坐标

**发布包**在一个实例内用 Zenoh TCP `0.0.0.0:7447`，本机订阅器作 client 连 `tcp/127.0.0.1:7447`。停机后把 `cfg/sensors/config_dual_airy_gpu.json` 的两个条目及 `cfg/sensors/config_default.json` 的 `IMU`、`Odom` 条目合入主配置 `robot.sensors`。后者初始为空；保留 `sensor_sync_mode=true`。只载入 dual 预设时两路惯导数据都缺。先订阅 `**` 发现 key、速率和载荷，再精确订阅。证据见 `cfg/config.json:6–19`、`docs/SENSORS_CN.md:5–25,50–64`、`Tools/zenoh_topic_monitor.py:58–93`。

| 数据 | 请求频率及候选 key | 消息和使用 | 发布包依据 |
|---|---|---|---|
| 前 Airy | 10 Hz，`rt/front_lidar` | 有组织的 ROS 2 XCDR1 `sensor_msgs/msg/PointCloud2`。附着 base_link，位置 (0.46615,0,0.1) m，RPY (0,−90,0)° | `cfg/sensors/config_dual_airy_gpu.json:2–29`、`docs/ZenohRecording_CN.md:38–44`、`docs/LidarCameraProjection_CN.md:3–6,262–278` |
| 后 Airy | 10 Hz，`rt/rear_lidar` | 同格式。位置 (−0.454745,−0.040323,0.156355) m，RPY (0,−90,180)°，前后均指定 GPU、滚扫和忽略挂载对象 | `cfg/sensors/config_dual_airy_gpu.json:18–28,30–57`、`docs/ZenohRecording_CN.md:38–44` |
| IMU | 500 Hz，`rt/imu` | `sensor_msgs/msg/Imu`。Header、XYZW 姿态、局部 FLU 角速度 rad/s 和线加速度 m/s²，三个 3×3 协方差阵。全零表示未知。投影工具解码时跳过线加速度，需要时自行取出 | `cfg/sensors/config_default.json:2–20`、`docs/RoamerX_Lite_Integration.md:96–105`、`docs/LidarCameraProjection_CN.md:302–313`、`Tools/visualize_lidar_camera_projection.py:413–431` |
| Odom | 100 Hz，活动预设 `/odom` 预期为 `rt/odom`；另一份指南列 `rt/odom/mujoco_odom` | `nav_msgs/msg/Odometry`，含 stamp、父子 frame、pose/twist 和各 6×6 协方差阵。pose 是世界下的 base_link（m、XYZW），twist 是 child/body FLU（m/s、rad/s）。记录实际 key、父 frame 及 child frame，避免把机体系速度再当世界系 | `cfg/sensors/config_default.json:21–40`、`docs/LidarFisheyeProjection_CN.md:52–59`、`docs/RoamerX_Lite_Integration.md:96–105`、`docs/LidarCameraProjection_CN.md:315–326`、`Tools/visualize_lidar_camera_projection.py:372–409` |
| 奖励用真值 | 建议从上述 Odom 的绝对 pose 取样，仍为 100 Hz 请求值 | 文档将 pose 定义为 `T_world_base_link`。需核验三控制点和轴向后才作为 A 段真值。配置还有 `state_port="mujoco/state"`；本包上述文档和工具未给它单列一个可直接解码的机器人真值协议或频率。不要假定它是 ROS Odom 或 500 Hz，也不要用无 Header 的动态框代替 | `docs/LidarCameraProjection_CN.md:315–326`、`docs/LidarFisheyeProjection_CN.md:52–63`、`cfg/config.json:15–18`、`F/USAGE.md:22–27` |

**Airy 解码**。PointField 包含 float32 的 x/y/z/intensity、uint16 ring、float64 timestamp，常见步长 26 B，逐点时间为绝对秒。读取实际字段定义、端序、行跨度和两级时间戳。NaN 表示无回波。intensity 含仿真语义标签，策略输入舍弃它。统一 Header 为 `map`，坐标仍在雷达局部 FLU，发布器已完成厘米转米和 Y 翻转。用各自外参 `Rz(+yaw) Ry(−pitch) Rx(−roll)` 转到基座，切忌再次镜像。两雷达按 Header 组对；滚扫需按逐点时间和可部署里程计补偿，先在静止或关闭畸变时核对。工具提供 XCDR1 对齐、端序和 XYZ 提取，原函数返回时已舍弃时间列，做去畸变须保留它。证据见 `docs/LidarCameraProjection_CN.md:181–204,262–278,296–300`、`Tools/visualize_lidar_camera_projection.py:166–188,497–525,540–550`、`docs/SENSORS_CN.md:38–48`。

**建议**。保留原始时间戳及本机单调到达时刻，按仿真时间每 200 ms 生成一个决策样本，统计有效帧率和配对率。绝对 pose、跨回合航向和路线只送奖励／终止／评估进程。V3 里程计特征以相对起点及机体可测量量构造。Header 若与壁钟不同步，单调间隔适合计速，不能直接相减算端到端时延。投影工具明确记录 Odom 与其他源时钟可能不同，监控器仅对可解释为 epoch 的 stamp 估时延。证据见 `docs/LidarFisheyeProjection_CN.md:52–61,73–81`、`Tools/zenoh_topic_monitor.py:104–123`。

## 内置运控、速度和登阶

- **发布包**以 `inside_mc=true` 启动内置控制器，支持 M1 对应的 `zgws`。UDP `0.0.0.0:7447` 接受每包一条 UTF-8 JSON 手柄报文，与同号 TCP Zenoh 可并行。0.5 s 未刷新时速度输入过期，Auto 可能回到手柄，没有执行 ACK。使用工具的完整中性报文模板，其余轴、按钮归零，正前进写 `LY=+u_x`，正左移写 `LX=−u_y`，正左转写 `RX=−u_w`，三个 stick 量均限幅到 [−1,1]。该协议没有直接接收 m/s、rad/s 的字段。工具对平移 0.12、旋转 0.18 内的测试动作判为死区。证据见 `docs/MOTION_CONTROL_CN.md:17–35,37–45,69–82,91–106,145–153`、`Tools/matrix_mc_udp_test.py:24–54,63–83,114–132,239–251`。
- **建议**。策略/数据集的物理速度需要设备标定函数映射到摇杆。用 Odom 测前、后、横移、旋转各方向的死区、响应、饱和与耦合，记录 SI 目标、实际 sticks、实测 twist。传输线程按示例 20 Hz 重发最近一个未过期的 5 Hz 决策，采样或策略停滞时归零。起身为 `LB+Y`，释放→按下→释放后观察 `STAND_HOLD`；不要逐决策调用示例 `move()`，它退出时主动切平衡站立。证据见 `Tools/matrix_mc_udp_test.py:24–26,74–104,122–132`。
- **登阶是阻塞项**。内置包清单含 `zg_wheels/odom_stair` 和 `policy_stair`；官方说明要求按当前机型启用的动作判断。发布的内置 UDP 命令表和报文字段未列从 Walk 进入 Stair 的按键或 `control_mode/motion_mode`。历史交接记载单独发布的 Robot_MC v0.6.6 从行走态经 40 B 二进制 SDK 指令 `control_mode=18, motion_mode=20` 进入 `RL_State_Stair`，切其他 motion_mode 退出；保存的外置控制器符号也有 Stair onEnter。该记录不能确定内置 7447 的 Stair 协议。需要内置入口、模式转换和持续状态的日志/遥测证明，包括每次 reset 后重新进入。没有这个证据时只记录接口调试数据，不能宣称完成用户指定的全程登阶采集。若内置版本未暴露入口，需要供应商提供方法或由主会话决定是否改方案。证据见 `Manifest_NonUFSFiles_Linux.txt:192,212`、`docs/MOTION_CONTROL_CN.md:63–82,91–106`、`Tools/matrix_mc_udp_test.py:36–55,184–222`、`exp_nav3/HANDOFF.md:86–99`、`E/stair_onenter.txt:1–15`、`E/robot_mc_v0.6.6.list:403`。

## 重置、时钟和多实例

| 问题 | 静态证据 | 建议和验收 |
|---|---|---|
| 热重置／传送 | 发布工具 `send_spawn_udp.py` 面向 8081/UDP，只列道具、行人，坐标为相对 player 的生成参数。运控命令只有起身、停止、动作等。活动配置有初始 position，文档未给已有机器人自由关节、控制器隐状态和时钟的一键复位或 step RPC。见 `Tools/send_spawn_udp.py:27–44,48–85,100–109`、`Tools/matrix_mc_udp_test.py:199–222`、`cfg/config.json:10–19`、`F/USAGE.md:7–9,24–27`。 | 后续先测可重复冷重启路径。停止本实例、合入抽样起点的配置和机器人 MJCF 初始 pose（位置、朝向、站立高度），重启并重走生成／起身／登阶流程。需验证 JSON 与 MJCF 初值优先级，连续两轮检测实际 pose、速度收敛、新 stamp 和新 episode ID，清空所有延迟消息、BEV 历史与策略状态。若无无人进场入口，自动重置及连续批采尚未打通。 |
| 固定步长 | 全段物理 MJCF 明写 `timestep="0.002"`，即 500 个积分步/仿真秒。传感器同步模式使用公共仿真时序。记录器的 0.25–4 倍是离线消息回放，云台协议中的固定步长 watchdog 限云台；这些没有定义 MATRiX 闭环锁步 API、外部 step(n)、暂停/续跑或超实时开关。见 `FZ/zgws_atec_full.xml:1–4`、`FZ/atec_scene_full.xml:1–4`、`docs/RoamerX_Lite_Integration.md:76–80`、`docs/ZenohRecording_CN.md:53–60`、`docs/CameraGimbalUdpJsonControlProtocol.md:1058–1066`。 | 5 Hz 决策理想为 100 个物理步，属于异步控制调度目标，不能断言每次调用固定走了 100 步。核对时间源后测 RTF=仿真时间增量/单调壁钟增量，分别统计满图单实例和多实例的配对帧、错过决策与轨迹吞吐。预算按实测有效 transition/s，不预设 RTF>1。 |
| 多实例 | 配置可改 Zenoh 监听 TCP endpoint。内置 UDP 默认绑定 wildcard:7447，工具的 `--port` 只改客户端目的端，配置及文档未给内置服务改端口方法。多个程序复用同一工作目录还会共用 config 和 Saved/Logs。发布指南给系统 RAM 建议和传感器负载提示，没有单实例显存或并发上限。见 `cfg/config.json:2–19`、`Tools/matrix_mc_udp_test.py:59–70,184–196`、`docs/MOTION_CONTROL_CN.md:100–106`、`docs/USAGE_CN.md:5–24`、`docs/SENSORS_CN.md:36–46`。 | 从单实例起。若能证明服务端 UDP 可配，为每 worker 配唯一 TCP、UDP、Saved、工作配置；否则首选每实例一个可用的网络、IPC 命名空间（检查内置共享内存）及私有可写配置/日志，内部 loopback 维持 7447/TCP+UDP，桥接进程在同一命名空间。禁用多播发现并分别连接对应 router，避免 key 串线。Colab 可能不给 namespace/container 权限，退回单实例。每次只增加一个实例，用独立指令验证另外实例不动，再比较合计有效采集率、丢帧、峰值显存/RAM 和重启峰值，超过单实例吞吐才保留。 |

## A 段自定义场景

1. 按 `F/USAGE.md:22–27` 把 `FZ/` 下 `zgws_atec_*.xml`、`atec_scene_*.xml`、`atec_full/` 合入私有发布副本相同的 zgws 路径并保留包自带 `assets/`。组合入口同时 include M1 本体与场景，模型路径相对 `UeSim/`。证据见 `FZ/zgws_atec_full.xml:1–4`、`FZ/atec_scene_full.xml:10–20`、`docs/USAGE_CN.md:26–52`。
2. 先用 `Content/model/zgws/zgws_atec_window_00.xml`（0–100 m），通过后替换为 `Content/model/zgws/zgws_atec_full.xml`。保留候选的 `scene_model=""`、`main_body="base_link"`、起点及 `sensor_sync_mode=true`。**两个候选配置实际均为 `inside_mc=false`、`sensors={}`**，须显式改为 true 并合入前述四个传感器。该项目使用组合 MJCF；历史调查未给独立 scene_model 的精确定义。证据见 `FC/config_atec_window_00_candidate.json:2–19`、`FC/config_atec_full_candidate.json:2–19`、`F/assembly.json:4–23`、`E/map_import.md:10–15`。
3. 校验生成后机体、前进轴、竖直控制点和 Odom 的原点／手性，并检查激光是否从两侧看到在物理地图可定位的踏面、扶手和墙。MJCF 可编译和导入、UE 显示、GPU Airy 射线命中是三项验收。文档将雷达表面描述为碰撞表面；full_v1 的渲染网格、补面、flex 碰撞及软植被有不同分组。若只得到机器人/默认地面的回波，先解决 UE／射线场景导入；BEV 此时无法验证导航。证据见 `docs/SemanticCamera_CN.md:40–46`、`F/USAGE.md:7–10,22–27`、`E/map_import.md:10–15,21–23`。
