# MATRiX 双 Airy 采集

服务器 CPU 准备位置库，Colab 临时 VM 执行离屏仿真。每个 slot 使用独立网络、IPC、挂载命名空间、可写共享内存和 UE 工作目录。内部 Zenoh 与控制端各自绑定本机 7447（TCP/UDP），采集 worker 直接进入同一命名空间。slot 对应 window_00 至 window_13。一次回合冷启动一次 UE，结束时清理进程组。同一档的各 slot 独立复位。并发冷启动用文件锁错开最初的主视口和着色器初始化，进入物理初始化后释放。

## 准备与启动

1. 在具有私有 full_v2、full_v1 和冻结路线的源检出目录，用 mjmap CPU 环境运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python /path/to/collect/spawns.py 0 /path/to/inputs/window_00.json --count 48`，对整数 0–13 分别执行。拟合复用 collision_spawn.prepare 和 stable。按窗口均匀抽站号，横偏范围 ±0.2 m、航向扰动 ±10°。在踏面内，沿路线方向求踏面两条边的交点，取中点并拟合轮端支撑。每个候选经过 3 s 无夹具承重验证。回合种子从通过的离散库抽样，可在运行前扩大库。数据中的 fit_seed 和 attempt 可重建候选。
2. 将 full_v2/work/ground_rays.csv 的 17 条横向采样按站号排列，保存形状为 (站数,17,3) 的 ground.npy，最后一维取原第 0、1、5 列（站号、横偏、扫描系碰撞面高度）。运行时按最近站号和横偏查支撑高度，减 assembly.json 的 shared_shift_m[2]。缺观测地面独立标记。
3. 在临时 VM 运行 envs/matrix_v4/colab_setup.sh system，然后携带 full_v2 私有包执行 MATRIX_SCENE=window_00 的 collect 阶段，安装 bridge 依赖、图形库及已校验发布版。将许可的 CustomWorldBundle.pak 放入 /content/val。将本目录置于 /content/col/collect，以上位置库与 ground.npy 置于 /content/col/inputs。
4. 在 VM 运行 `python3 /content/col/collect/benchmark.py 1 --seconds 600`。运行目录包含 tier.json、resources.jsonl、逐回合启动和控制日志及 episode.json，完成后才公布 done.json。分母从第一实例启动前计时，包含启动、起身验收、复位及本地分片写入。退出边界外仅做收尾。
5. 在服务器执行 `python sync.py /path/to/data --transport 'bash /path/to/mux.sh' --watch`。transport 是从服务器向当前 VM 发起命令的 SSH 包装器。只枚举完成标志，断开后从已收字节恢复，校验通过后原子改名。服务器运行单测用 `python -m unittest -v test_collect`，可选 MV4_CDR_DIR 指向原始验证回合。test_spawns 需要 CPU mjmap 和几何模块。

## 时序与验收

所有卡沿用 config.json。Odom 的物理戳驱动 0.2 s 决策栅格，跳过错过的栅格；UE 定时器名义 50 ms 重发，过期 500 ms 归零。在同一次决策快照取已收到的最新两帧。Airy 的 Header 对应扫描起点，末个 firing 的绝对秒取自点字段。用 VM 的 UTC 与单调时钟差投影扫描末时刻。采集器保存 Odom 的双钟锚点并在有效区间内把扫描末时刻插值到 Odom 域 (scan_sim_end)。接收时延、时钟误差和 100 Hz Odom 的离散化会影响此估计。两路采样墙钟帧龄均在 [0,0.20] 秒且站立才记合格。这一上限对应一个决策周期，使用墙钟可计入发布与排队延迟。复位等待不生成决策步。每回合重新建立时基。

前视点的圆弧曲率按 2 sin(航向误差)/目标距离计算。用前进 SI 速度乘曲率得到角速度，超出原地调头门槛时单独控制偏航。IMU 只取不晚于 Odom 的样本，0.10 s 内找不到配对时发零速，持续 1 s 才判断流。

姿态门槛取自上一轮翻倒与正常步态观测。起身等待 4 秒，验收需机身高于碰撞面 0.30 m 且俯仰和横滚均小于 0.70 rad。上一轮成功登阶初始俯仰约 21°，失败入口机身曾降到约 0.13–0.19 m。行驶容许降到 0.26 m 和 1.10 rad，后者沿用上一轮终止阈值。横偏 1.2 m、15 仿真秒内未推进 0.10 m 沿用上一轮循迹诊断尺度。时间按物理秒衡量使卡顿与运动停滞可以区分。奖励采用 nav4 的 2×进度增量（限幅 ±0.5 m）、每步 −0.001 和驾驶失败 −15。运维截断独立记原因。

## 分片字段

每回合是可用 numpy.load(allow_pickle=False) 读取的压缩 NPZ。meta 为 JSON 字符串，含种子、窗口、出生姿态、驾驶器、配置、GPU、并发和终止原因。起身失败回合可有零行。每条决策的 frame_index 两列按前、后顺序指向该侧帧号，−1 表示未收到。唯一帧只写一次，例如 front_00000_mm、front_00000_intensity、front_00000_cdr_zstd（用 codec.unpack_wire 恢复原报文字节）、front_00000_shape_count 和 front_00000_times。shape_count 为原始 height、width、原始槽数和有限点数，times 为 Header、末点 UTC 以及单调接收时刻。96×900 图的列保留设备原生 firing 顺序，较短帧在尾部补零，无回波或填充的距离为 0。65535 为量程溢出或该端点值，完整 CDR 可恢复所有原始位，包括 float32 XYZ、intensity、逐点 float64 时间和 NaN。intensity 是发布器的标签值。

step 数组含 age_s、imu_quat (xyzw)、imu_omega、actual_twist、action、priv_pose (xyz+xyzw)、priv_progress、priv_cross_track、priv_clearance、reward、qualified 和 terminal。action 为 SetWalkCommand 接收的归一化三轴目标，actual_twist 为 Odom 的 SI 速度。reward 记录从上一状态到本状态的增量，action 是在本状态发出的命令，训练转移应匹配下一步的 reward。末步终止或截断时标记 terminal。V3 可实现 control.Driver.command，在内部用已选双帧与 IMU 构建其所需 BEV，生命周期按回合管理。分片保留原始传感器表示。

## 服务器产能测试入口

在研究服务器将 MATRIX_COLLECTION_DIR 指向实验目录，内含 inputs 和相邻的 exp_nav3_matrix_v4、matrix_v4_val、envs。host/suite.sh 检查余额并依次申请 T4、L4、G4，逐档采样、校验归档并关闭会话。已有结果可用 host/analyze.py 和 host/audit.py 在相同变量下复算。通用入口只用于新预算，运行前准备对应的已授权 SSH 密钥和私有包。
