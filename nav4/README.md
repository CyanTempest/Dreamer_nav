# nav4：A 段山路全程导航仿真包

自包含的 JAX 批量环境、采样规划器、Dreamer 策略包装和全程评估，替代 `nav3/` + `r9_official_runner/` 的多层旧代码（旧代码未改动）。不 import `nav3`；`dreamerv3/` 原样使用。

## 文件

| 文件 | 内容 |
|---|---|
| `config/env.json`, `config/planner.json` | 唯一参数来源（动作缩放、BEV、时序、跟踪器、延迟、包络、噪声、奖励、限速……） |
| `tracker.py` | 唯一的 20 Hz 目标跟踪律；环境子步、规划器前向预测、支撑风险过滤共用 |
| `bank.py` | `TrailBank`：读 manifest 中任意张地图（float16/float32 均可，ground 为 float32），`sample/atlas/route_at/project` |
| `sensor.py` | 双 Airy 激光雷达 + 深度相机的可见性模型、噪声、局部几何支撑判定 `support_mask` |
| `env.py` | `TrailEnv` |
| `planner.py` | `Planner`（采样规划器、`assess`、`support_risk`）与 `Hybrid` 接管包装 |
| `policy.py` | `DreamerPolicy`（含 `deterministic`）、`load(checkpoint)`、`leakage_check` |
| `evaluate.py` | 全程评估 CLI（含 assist 模式） |
| `tests/` | CPU 单元测试与合成地图 `tests/synthetic.py` |

## 接口（与旧 Dreamer 检查点兼容）

- `env = TrailEnv(TrailBank(manifest), num_envs, {'section.key': value})`
- `reset(key, mids=None, start_s=None, offset=None)`、`step(state, action)` → `(state, obs, extra)`；某槽位 `is_last` 后，下一次 `step` 把它重置（该步动作被忽略）。训练重置站点见 `task.*`；`state['restart']=(站点, 横偏)` 有限时按该位姿重置并保留障碍（评估的 assist 用）。
- 动作 3 维 ∈[-1,1]：目标半径 `action.radius`、方位 ±`action.bearing_deg`、速度 `action.speed`；5 Hz 决策 = `timing.substeps` 个 50 ms 跟踪子步。
- `obs`：`bev` uint8 [N,64,64,6]（地面高、障碍高、支撑、未知、路线走廊、记忆年龄），`state` float32 [N,13]（估计位姿系下 3 个路点/`waypoint_scale`、已过门数/8、测量速度、上一动作），`priv_trav`（特权解码目标，不进编码器），`reward/is_first/is_last/is_terminal`。
- `extra`（永不进 obs）：`occluded`、`clean_trav`，以及真值 `route_mask`[N,64,64]（真位姿下的走廊，含被遮挡格）、`progress`（真站点/路线长）、`heading_error`、`lateral`。
- `route_inputs=false`：走廊通道、6 个路点和进度项置零，形状不变（无地图系定位策略用）；组装在 `TrailEnv.assemble` 一处。
- 失败码：1 跌落/无支撑，2 碰撞，3 偏离路线/倒退，5 卡住/超时。成功 = 到达最后一个 gate 站点且同一步无失败。

## 主要配置键

`timing`（dt, substeps）· `action` · `tracker`（motor_tau, yaw_gain=2, norm_floor, 限速/限加速度）· `dynamics`（延迟、增益随机范围，打滑）· `speed_limit`（陡坡/台阶 0.28 m/s：前后 ±steep_probe 高差 > steep_rise）· `robot`（包络 1.03×0.58，足迹采样，支撑阈值）· `terrain`（max_step 0.20 m、max_grade_deg 35、collision_height）· `bev` · `memory`（8 s 过期、观测延迟）· `sensor`（雷达/相机安装、波束、`atlas_bands`=3 即身体高度层、噪声范围）· `localization` · `task`（起点分布、超时速度、偏航阈值）· `reward` · `obstacles`（count=0 关闭）· `assist`。
规划器：`planner.json`（候选数、前视、预测时域/延迟、代价权重（单位：秒）、刹车阈值、`filter.*` 接管阈值与保持步数）。

奖励（每 5 Hz 步）：`progress·clip(Δs) + gate·过门·[无失败] − time − smooth·mean(Δa²) − clearance·max(0, d₀−min(支撑距, 障碍距)) − lateral·(横偏/max(半宽,floor))² − failure·[失败]`。

## 运行

```bash
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh          # 解释器 $NAV3_PYTHON、显存上限、确定性 XLA 标志
JAX_PLATFORMS=cpu $NAV3_PYTHON -m unittest discover -s nav4/tests -p 'test_*.py'   # 约 3 分钟；或逐个: -m unittest nav4.tests.test_env_support
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method planner \
    --repeats 8 --seed 914901 --output audit_20261116/nav4_dev/c2_planner_s914901.json --plot
$NAV3_PYTHON -m nav4.evaluate --manifest data/trail_a --maps forward --method hybrid --action mode \
    --checkpoint checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl --repeats 8 --assist --output ...
```
评估输出：每条 traverse 一行（成功、失败码与站点、最终站点、步数、平均速度、接管比例、assist 次数），汇总严格完成率（0 assist）、≤3 次 assist 完成数、每 100 m assist 数。`--env key=JSON` 可覆盖环境配置做诊断。

## 旧缺陷 → nav4 行为

| 旧缺陷 | nav4 | 测试 |
|---|---|---|
| 支撑/安全检查把"比机身下方低 0.55–0.6 m"当掉落（`env.py:175`、`r8_p1_online.py:135`、`control.py:33`），15°以上下坡 2 m 外全被判危险 | `support_mask`：3×3 中值后，相邻格台阶 > 0.20 m 或坡度（minmod）> 35° 或障碍 ≥ 碰撞高才无支撑；`assess`、`support_risk` 用该通道 | `test_env_support` |
| `jnp.roll` 求梯度，BEV 边缘两行/列在坡上被误判 | 边缘填充（edge pad），无环绕 | `test_env_support.SupportMask.test_planes_supported_up_to_the_bev_edge` |
| R8/R9 高度噪声为 0；障碍、支撑由无噪声地面算 | 配置噪声（高度 std 0.015–0.035，dropout 基值/距离项中值即拟合值），每回合随机；障碍/支撑由同一份带噪、可见性掩码后的测量导出 | `test_env_noise` |
| 状态向量路点用真站点（`env.py:224` 泄漏） | 路点与走廊均由带漂移的估计位姿求 | `test_env_localization` |
| 固定 40 m 窗、从 2 m 处出发、同步失败仍算成功（过门奖励照给） | 训练随机站点（10% 在起点），评估从 `route_start`；成功 = 到达终站且同步无失败，失败步不给过门奖励；8 个 gate 保留为进度 | `test_env_task.Task` |
| 陡坡限速硬编码 | `speed_limit.*` 配置 | `test_env_task.SpeedLimit` |
| 随机障碍按 1.7 m 宽度规则全部失效 / 不保证通道 | `obstacles.count`（默认 0）；按真值横向可通行区间放置，保留包络宽 + 0.10 m 通道；传感器可见 | `test_env_obstacles` |
| 三份跟踪律（偏航增益 2/4、范数下限 0.1/0.15、缺 `min(speed,norm)`、固定/随机延迟），R9 忽略 `yaw_tracking_gain` | 单一 `tracker.substep`，默认 yaw_gain 2，规划器/过滤器用名义延迟与增益 | `test_tracker_consistency` |
| `route_at` 假设 0.2 m 点距，`project` 用地图站点层（全局最近点，回头弯/上下层路线会跳段） | 二分查找真实弧长；`project` 在上一站点 ±`task.project_window`（5 m）内搜索。窗口 1 m 时 segment_1010 的急弯处机器人切弯，投影卡在窗口端点被判偏离（码 3），见 README 下方说明 | `test_bank_loader` |

已知限制：
- 从高处看向 > 0.2 m 的下跌坎，坎下地面处于雷达阴影中，表现为"未知"而非"无支撑"（`test_occluded_drop_reads_unknown`）。
- 规划器代价按旧版只含障碍/边界/未知，不含支撑通道（支撑只在 hybrid 的接管判定里用）。
- 训练随机起点不检查包络可行性；地图中心线不可行的站点会在第 1 步失败（`data/trail_a/forward` 约 22% 站点中心线不可行）。
- `project_window` 必须大于机器人一步内可能"切掉"的弧长，又要小于上下层回头弯之间的弧长间隔。
