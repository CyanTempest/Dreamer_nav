# nav3：局部导航实验

代码主目录是本 Mac 工作区；运行目录是远端 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`。GPU 作业均使用 `nav3_` 前缀 tmux。当前实测状态见自动生成的 `REPORT.md` 和 `metrics/`；未通过闸门不能据此宣称训练完成。

## 复现

```bash
# Mac 同步源码（不传 checkpoint/replay，不删除远端结果）
bash scripts/sync.sh
ssh hpf@100.108.114.64
cd ~/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/env.sh
"$NAV3_PYTHON" -m nav3.maps --train 200 --heldout 20 --workers 6
"$NAV3_PYTHON" tests/check_contracts.py
"$NAV3_PYTHON" tests/check_dreamer.py
"$NAV3_PYTHON" -m nav3.profile --envs 1024
"$NAV3_PYTHON" -m nav3.evaluate --g1
"$NAV3_PYTHON" -m nav3.raycheck --reference --workers 3
"$NAV3_PYTHON" -m nav3.raycheck --compare
# 仅 G1 已通过才允许以下命令
"$NAV3_PYTHON" -m nav3.train_dreamer --name dreamer_s0 --seed 0 --steps 100000 --seconds 7200
"$NAV3_PYTHON" -m nav3.train_ppo --name ppo_s0 --seed 0 --steps 100000 --seconds 7200
"$NAV3_PYTHON" -m nav3.gates
# 正式训练还要求 G2 已通过；单次上限 <3 h，可 --resume
"$NAV3_PYTHON" -m nav3.train_dreamer --name dreamer_s0 --seed 0 --steps 100000000 --seconds 3300 --resume
"$NAV3_PYTHON" -m nav3.audit
"$NAV3_PYTHON" -m nav3.report
```

以上长命令实际运行须放入 tmux，例如 `tmux new -d -s nav3_profile 'bash -c "source scripts/env.sh; $NAV3_PYTHON -m nav3.profile > logs/profile.log 2>&1"'`。先检查 `nvidia-smi`；不得停止其他进程。`scripts/fetch.sh` 将指标、图和日志拉回 Mac。

## 接口与坐标

`FastEnv(MapBank(split), N)` 提供 JIT 编译的批量 `reset(key)`、`step(state, action)`；返回模拟器状态、公共观测及单独的诊断信息。训练代码只能通过公共观测调用策略。状态字典有真值，不能直接给策略。

BEV 为 uint8 `[B,64,64,6]`，轴 0 向前、轴 1 向左；x 范围 [-2,6] m，y 范围 [-4,4] m。通道顺序：相对高程（±2 m）、障碍高度（0–2 m）、几何推断支撑、未知、主路线走廊、观测年龄（0–8 s）。`env.decode` 还原尺度。路线走廊由估计位姿和路线图生成，岔路不画错误支路；不是隐藏支撑区标签。

向量为三个机体系路点的 xy/4、阶段/8、带噪实际 vx/vy/wz、上一个三维动作。`priv_trav` 为 uint8 `[B,64,64,1]`，用完整可支撑区、静态和回合障碍进行机体矩形腐蚀。只有 Dreamer 解码器使用它；`tests/check_dreamer.py` 用同随机键替换全部特权标签，验证策略动作及隐状态逐值不变。

默认动作 `a∈[-1,1]^3`：目标半径 `r=0.5+1.25(a0+1)` m，方位 `θ=a1·π/3`，速度 `v=0.4(a2+1)` m/s；目标在当前机体系。20 Hz 跟踪器指向该固定世界目标、允许平移，转向为 `w=clip(2v·y/r²,±0.75)`，再施加速度/加速度限制和一阶动力学。速度档为零时不会主动原地旋转。滑移随实际速度缩放。`configs/env.json` 集中管理机体、时延、滑移、速度和噪声参数。直接速度消融使用 `(0.8a0,0.6a1,0.75a2)`。

## 地图、噪声与边界

`vendor/map/scene.py` 是已验收生成器的原样副本，`nav3/maps.py` 只调用生成逻辑，不调用 PLY/MJCF 导出。训练种子 10000–10199，留出 20000–20019，mini_trail 为测试。栅格 conservative coverage 保留细杆；台阶更新 ground，障碍更新 top 和距离场。稀疏 32×32 tiles 常驻 GPU，路线段投影使用 float32。

随机图有两项可达性修正：将与窄路/台阶收窄区重叠的倒木移到附近宽路段；将倒木旁唯一绕行侧的冲突石块移到相邻外侧路肩，保留数量、尺寸和离地高度。修改记录在每张地图的 metadata。原始只读目录不变，mini_trail 不做这两项修正。`nav3.clearance_audit` 核对倒木邻域的路线朝向矩形足迹连通性；它是诊断，不能替代真实策略评估。

快观测用双雷达的标定半球、极坐标高程最大坡度遮挡、近身盲区、距离丢点、高程噪声、里程计随机游走、运动补偿累积、衰减和延迟。累积在世界朝向的 16×16 m 滚动网格上，使用整数格平移，避免重复亚像素重投影失真。它是 2.5D 近似，不能表达导引带下方透空、真实反射率或腿部遮挡。`raycheck` 的 60 位姿完整射线参考专门用于量化这些误差；静态对比不等于实机标定。

地图外支撑判定和碰撞使用 1.03×0.58 m 矩形采样足迹；每 50 ms 检查。训练无安全过滤；评估另测过滤。终止后先输出 terminal 观测，下一步输出 first/reset，遵循官方 replay 的 `o_t,r_t,a_t` 对齐。

## 训练与存档

用户最新范围：**Dreamer 和 PPO 各一个种子（seed 0），各累计 5 小时**（含已完成训练时间）。恢复队列使用 `--train-seconds 18000 --budget-seconds 86400`，训练目标仍各累计 5 小时；扩大调度余量以覆盖事故、续训和完整评估。历史失败墙钟不扣后续方法预算，同一配置最多失败 3 次，之后继续下一配置。额外种子、随机预填对照和消融暂缓。每段最多 10000 秒，自动从 checkpoint 续训到累计目标；不是每段重新训练 5 小时。训练后仅评估规划器、dreamer_s0 和 ppo_s0。

2026-11-10 显存调优：调度器从 `configs/runtime.json` 读取 Dreamer 采样环境数，当前为 **4608**（此前 32）。此前联合短探针占用 **21944/24564 MiB，89.3%**，记录见 `metrics/memory_probe_4608.json`；OOM 修复后完整训练的实际峰值为 **72.23%**，以 `metrics/training_resources_summary.json` 为准。保持 size12m、训练 batch 16×64、train ratio 32。JAX 仍按需分配，分配上限为 95%，没有用占位张量填充显存。

环境数改变时，续训保留模型、优化器、replay 和更新计数，只重新开始采样回合并清空策略 carry；事件记录为 `resize_envs`。新训练至少预填每环境 66 步，避免数千环境下序列不足。Replay 容量仍为 100 万，分块为 64，原有 256 步分块仍可加载。2026-11-11 OOM 恢复最终改用无损 LZ4 内存块保存大数组：dtype/值、序列和官方采样器不变，抽样时逐值还原，循环状态更新后重新无损压缩。直接调用主机已有 `liblz4.so.1`，不向共享 Python 环境安装包。曾验证的文件映射方案因磁盘缓存抖动已停用。

Dreamer 用官方 size12m、batch 16×64、replay 容量 100 万、train ratio 32；官方 Replay、优化器和想象学习算法均未替换。批量策略适配器调用同一官方 policy，实际执行动作写回 carry。规划器预填单独计数，在线更新数为 `floor(online_steps×ratio/1024)`。检查点保存模型及优化器、策略 carry、train carry、随机状态和计数；环境状态逐数组保存到独立文件，Replay 原子存盘，并保存有效 chunk、FIFO 索引和 worker 流以续接上下文。恢复兼容旧版检查点，旧对象加载完成即释放。RSS 每 10 秒记录，超过 50 GB 时先保存检查点，再以退出码 75 交给调度器恢复。默认回合随机障碍开启。

PPO 为 CNN+256 GRU、32 步 BPTT、128 步 rollout、4 epochs、clip 0.2、GAE λ=0.95。完整 rollout/优化器探针后选用 640 环境（约 71.9% 显存、1.066 万步/秒），保留原 seed 0 冒烟权重、优化器和累计时间；调整环境数时只重置采样回合。actor 和 critic 都只见相同公共观测。训练脚本记录每次更新、环境步数及墙钟，正式比较按相同墙钟预算，不把仿真吞吐当训练吞吐。

环境限制、闸门失败、未训练的配置会在报告中明确标成未完成；缺失值不会填成零或伪造模型预测。

恢复启动命令（先确认 GPU 没有其他训练）：

```bash
source scripts/env.sh
"$NAV3_PYTHON" -m nav3.runner --train-seconds 18000 --budget-seconds 86400
```

长任务在远端 `nav3_queue` tmux 中运行。`scripts/watch_queue.py` 每分钟记录本实验进程 RSS、交换使用、GPU 显存及主机可用内存；恢复验证与事故证据见 `metrics/oom_recovery_*.json`。`scripts/fetch.sh` 同时拉取 `REPORT.md`。

并发恢复后，循环状态替换可能让 libc 保留已释放页面；RSS 超过 32 GB 时，每分钟最多一次执行垃圾回收和 `malloc_trim(0)`，只归还空闲页。回收前超过 50 GB 仍会触发存档与退出，不能因回收后下降而漏报。最终验收脚本为 `scripts/verify_queue_completion.py`：核对两个 5 小时预算、24 个配对评估条件、至少 1560 回合及环境指纹。此次队列已完成，详细结果及原项目仍待完成的设计项见 `REPORT.md`，远端检查点与 Replay 保留。

## 第四轮 mini-trail 诊断

现场几何、29 次精确复现、原始几何净空、200 图统计、反事实和防护候选见 [DIAG_REPORT.md](DIAG_REPORT.md)。诊断数据为 `metrics/diag_*`，图为 `figures/diag_*`，逐步观测与 20 Hz 控制记录保存在诊断 NPZ。原 checkpoint 和旧评估 JSON 未修改，mini-trail 未用于训练。

`python -m nav3.diag_candidate --all` 可运行显式的观测障碍防护候选（跳过已有同名结果）；它不替换默认过滤器。候选提高 mini-trail 开过滤通过率，但接管率约 80%，无过滤策略未修复，留出 IoU 有统计显著下降。完整复现命令与限制见诊断报告。

旧报告的无随机障碍邻域 IoU 已改为“不适用”。重新派生且保留旧 JSON：

```bash
"$NAV3_PYTHON" -m nav3.report --tables-only --summary-output metrics/diag_evaluation_summary.json --report-output metrics/diag_evaluation_table.md
"$NAV3_PYTHON" -m nav3.diag_statistics
"$NAV3_PYTHON" -m nav3.diag_report
"$NAV3_PYTHON" -m nav3.diag_plots
```
