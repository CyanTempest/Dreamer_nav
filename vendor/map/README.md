# MuJoCo mini-trail 地图交付

本轮仅重建地图、搭建 MuJoCo 场景、渲染和导出传感器几何数据。**没有训练，也没有运行或评估策略。** 原目录 `research/exp_minimap/` 未修改。新增内容均在本目录；只在允许的 `research/envs/dreamer/` venv 中安装了 MuJoCo 和依赖。

先看 [图片总览](renders/00_contact_sheet.png)、[45 秒巡线视频](renders/mini_trail_follow_45s.mp4)，或在浏览器打开 [本地图片/视频画廊](renders/index.html)。所有图都是实际 MuJoCo 渲染或数值检查图，未使用生成式图片替代场景。

## 地图与本轮改动

- mini-trail 水平弧长 206 m，含起伏的三维路线长 **208.271 m**；8 个有序打卡门，2 个岔路。
- 普通路宽 2.60 m，坡段约 2.15 m，台阶段 2.20 m，窄路 **1.08 m**。窄路核心长 8 m，入口、出口各过渡 5 m。
- 用积分航向生成规则中心线，急弯最小曲率半径 **4.499 m**；连续上下坡各有 16 m 的 15° 恒坡核心，坡度在前后各 5 m 内平滑启停。
- 两组台阶各 4 级，标称每级 0.15 m、踏面 0.65 m，累计升/降 0.60 m。入口、出口宽度也走 5 m 曲线过渡。高度场保留阶跃，额外用石质 box 构成平踏面；石面高于栅格表面 15 mm，以避免重叠闪烁。栅格在立面处会形成约一格宽的斜面，台阶侧面的离散折线属于显式台阶几何。
- 单侧落差的附加深度最大 2.1 m，向路外平滑下降并融入远处地面；绿栏杆立柱直径 36 mm、横杆直径 32 mm。黑色导引带中心离地 0.40 m，带宽 45 mm，厚 4 mm，横跨第一个岔路。树、石块、倒木、落叶肩和门标牌都进入 MJCF。
- 机体用盒子、四根腿和四个轮子示意 0.93×0.48×0.585 m 包络；青色标前 Airy，橙色标后 Airy。不是厂家 CAD 或动力学模型。

## 宽度与地形连续性

统一采用五次 smoothstep `S(t)=6t^5-15t^4+10t^3`，区间外夹紧到 0/1。每次宽度变化至少用 5 m，端点的一阶、二阶导数为零。坡度的启停、路肩高度、落差场也使用平滑权重。

路面距离来自**显式左右偏移边界及圆端帽**，不是给“最近中心线采样点”分配一个离散路宽。岔路通过连续距离场的平滑并集（0.35 m 融合范围）并入，整个场景只有一张共享高度场，避免旧版分段覆盖高度产生尖刺。路面高程向背景山坡在 6 m 内融合；台阶影响向肩部在 2.8 m 内衰减。落差采用全局连续的坐标函数，消除了急弯内侧最近路径切换造成的接缝。

主交付高度场分辨率 0.10 m，矩阵 [1960, 595]，NPZ 约定轴 0=x、轴 1=y；MuJoCo `.bin` 用行=y、列=x，并在 XML 中恢复实际最低高程及高程跨度。材质贴图与同一地形坐标对齐，颜色表示泥土、草地、石面和落叶，不表示高度。

窄路采用 1.08 m 的依据：M1 宽 0.48 m，直行时总余量 0.60 m，即左右各 0.30 m。沿用上一轮 0.58 m 的保守宽度包络后，还剩左右各 0.25 m。这个余量用于地图/定位/离散误差，并不代表实机通行保证。

旧 `nav_env.py` 对支持掩码做 EDT，点落入非支持单元时距离恒为 0，再减半格 0.0625 m，因而得到 **−0.0625 m**。该值不能作为真实越界距离；0.125 m 最近邻取整本身的二维位置误差上界为 0.0884 m。新代码 `Scene.clearance()` 对连续道路边界给出带符号距离。中心线上的 1020 个机体包络姿态，最小连续净空 **0.2500 m**，取整网格后的最小净空 0.1755 m，均无边界越界。边界附近 801 个平移探针中，旧式取整出现 39 个误报、0 个漏报。这只是几何判定审计，不能认定上一轮所有策略越界都是误报，也没有在本轮重跑策略。

## 自检结果

统计为 x/y 四连通相邻高差；轴向坡度是 `atan(dz/0.1)`，梯度坡度由二维中心差分得到。全图包含故意设计的陡峭落差侧坡，不能用全图最大坡度代表可行驶路面的坡度。

| 区域 | 最大相邻高差 / m | 最大轴向坡度 | 最大二维梯度坡度 |
|---|---:|---:|---:|
| 全高度场 | 0.163435 | 58.54° | 60.47° |
| 非台阶，含落差坡面 | 0.163435 | 58.54° | 60.47° |
| 路面，排除台阶 | 0.031566 | 17.52° | 17.66° |
| 窄路两端过渡及 1 m 路肩 | 0.109501 | 47.60° | 50.90° |
| 台阶区域 | 0.152184 | 56.69° | 44.63° |

- 网格从 0.10 m 加密到 0.05 m：非台阶最大相邻高差从 **0.163435 m** 降到 **0.081804 m**，比例 **0.501**；真正的高度跳缝不会随网格尺寸这样减半。
- 两种网格共同节点的最大高度差 0.00164938 m；非台阶孤立峰/坑相对四邻居的最大突出量 0.001108 m。
- 台阶是有意保留的跳变，细网格台阶区域最大相邻高差 0.150036 m，单独列出，不混入连续地形判据。
- MuJoCo 高度场数据与 NPZ 的最大恢复误差 7.15e-07 m；300 个网格节点的垂直射线误差不超过 6.81e-07 m。
- 5 份 MJCF 均成功加载；视频逐帧解码验证 900 帧、1280×720、20 fps、45 秒。完整指标见 [map_checks.json](metrics/map_checks.json)、[media_checks.json](metrics/media_checks.json)。

随机生成器会打乱 6 个候选路段，再加入连续位置扰动；长度和急弯位置也改变。同一个 `seed` 可复现。下面数字表示特殊路段位于第几个已通过的门之后（0 表示门 1 之前），不是策略通过记录：

| 地图 | 窄路 | 上台阶 | 下台阶 | 有带岔路 | 普通岔路 |
|---|---:|---:|---:|---:|---:|
| mini_trail | 4 | 1 | 7 | 2 | 6 |
| random_001 | 6 | 1 | 3 | 7 | 4 |
| random_007 | 1 | 7 | 3 | 6 | 7 |
| random_019 | 1 | 7 | 7 | 4 | 6 |
| random_042 | 1 | 6 | 8 | 2 | 4 |

## Airy 与 RGB

依据 [RoboSense 官方 Airy 96 线手册](https://robosense-robotics.github.io/product-manual/en/Airy/) 的规格及附录 D：96 个标称垂直角（约 −0.07° 至 89.4°）、水平 360°/0.4°、10 Hz、0.1 m 盲区、60 m 最大量程。官方 10% 反射率下各通道 20/30 m 能力和发射时间偏移也保存在 [airy_specs.json](sources/airy_specs.json)。厂家当前产品页还包含 192 线版本，本项目没有混用其出点率。

每台完整帧投射 **96×900=86,400** 条射线；预览帧为 **24×180=4,320** 条。默认第 0 帧无周期缺口；每第 10 帧屏蔽 32°，与手册 856,320 点/秒的平均出点量相符。射线在实际 MuJoCo hfield、树干、树冠、栏杆、黑带、石块和机体上求交；考虑机体遮挡，排除雷达发射壳体自遮挡。使用 `mj_multiRay`，并与 256 次独立 `mj_ray` 交叉核对，最大距离差 **0 m**。

固定在 s=65.0 m，完整双雷达帧得到 55,515 个前向点、45,024 个后向点；预览分别 2,725/2,306 点。前向完整帧在黑带上有 58 个几何命中。青色代表前 Airy，橙色代表后 Airy。前后半球安装方向相反，因此前 RGB 内后雷达投影点数为零是正常视场关系；提供同位置的后 RGB 和融合俯视点云图，完整展示双雷达。

安装位置是工程假设：前后 x=±0.465 m、z=0.54 m，半球主轴朝机体前后；相机 z=0.555 m、垂直 FOV=68°。没有厂家实机外参/DIFOP 标定。本轮是**静态、理想最近交点几何帧**，没有反射率回波丢失、噪声、天气或扫描运动畸变；60 m 是几何上限，不能据此声称黑带或所有材质在实机上都能检出。完整与预览点云在同一静止姿态产生。

PLY 使用 map 坐标米制；对应 NPZ 保留每条射线方向、距离、命中 geom、通道、方位角、有效掩码和原点。相机 NPZ 保留内外参及深度，可独立复投影。

## 重新生成与查看

已验证环境：Python 3.11、MuJoCo 4.0.0、RTX 4090、EGL。无需更改系统或 conda 环境。下面从工作区根目录执行：

```bash
cd ~/wang-sm/ATEC_NAV_WSM
# 已安装；如重建 venv，可使用：
UV_CACHE_DIR="$PWD/research/exp_mujoco_map/cache/uv" \
TMPDIR="$PWD/research/exp_mujoco_map/tmp" \
uv pip install --python research/envs/dreamer/bin/python \
  -r research/exp_mujoco_map/requirements.txt

# 全部重建（地图、随机示例、几何自检、PLY、旧转换器、图片、视频、雷达）
bash research/exp_mujoco_map/rebuild.sh
```

分别运行时：

```bash
cd ~/wang-sm/ATEC_NAV_WSM/research/exp_mujoco_map
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
../envs/dreamer/bin/python scene.py
../envs/dreamer/bin/python scene.py --random --seed 123
../envs/dreamer/bin/python render.py --stills
../envs/dreamer/bin/python render.py --video --duration 45 --fps 20
../envs/dreamer/bin/python sensors.py
../envs/dreamer/bin/python check_map.py --refine
```

视频沿三维中心线匀速摆放 mocap 机体，完整路线压缩在 45 秒内约 4.61 m/s，**只是可视化播放速度**。不调用 `mj_step`，没有控制器、动力学可行性或轮足接触追踪。`render.py` 默认输出到本目录 `renders/`，使用不同 `--map` 时会更新相同名称的主预览产物。

有显示器的机器上，进入本目录并激活 venv 后即可：

```bash
source ../envs/dreamer/bin/activate
python view.py                    # 初始停在宽度过渡附近
python view.py --move --speed 1.0  # 运动学慢速查看
python view.py --map maps/random_001 --s 145
```

交互查看器使用 `mujoco.viewer.launch_passive`，自动选择 GLFW；鼠标拖动旋转、滚轮缩放，空格切换移动。可选 `--s` 指定初始弧长。当前无显示器会话验证了所有 XML 加载与 EGL 渲染，没有声称已手工操作桌面 GUI。EGL 已成功使用；若另一台机器没有 EGL，可尝试 `MUJOCO_GL=osmesa`，需要该机器已有相应 Mesa 支持，本轮没有安装系统包。

## 导出与兼容性

主地图在 [maps/mini_trail/](maps/mini_trail/)，随机实例在 `maps/random_001/`、`random_007/`、`random_019/`、`random_042/`。每份都包含：

- `scene.xml`：可独立加载的 MJCF；相对引用 `assets/terrain.bin`、`terrain_albedo.png`、`pine.obj` 和门牌纹理。复制时保留整个目录。
- `heightfield.npz`：高程、支持候选、落叶掩码、原点、分辨率、路线、宽度、连续边界距离等。
- `route.yaml`：路线、8 个有序门、岔路、特殊路段、宽度和尺寸说明；保留旧转换器要求的米制字段。
- `props.json`：静态几何体参数；`terrain.ply`：全覆盖地形采样点云。

主地图另有 `scene.ply`（地形及全部静态物体，3,091,835 点）、`pointclouds/airy_{front,rear,pair}_{full,preview}.ply`、对应射线 NPZ、相机标定/深度 NPZ。

直接调用了**上一轮未改动的** `../exp_minimap/convert_scan.py`，读入新 `scene.ply` 和 `route.yaml`，结果在 [converted_legacy/](maps/mini_trail/converted_legacy/)。路线位于点云边界内、采样覆盖率 **100%**，输出 23,903 个障碍候选格。转换器仍只提供几何支持候选，无法从 PLY 推断落叶下是否承重；完整真值掩码单独留在高度场 NPZ，没有伪造语义支持。

## 图片与视频清单

视频：[mini_trail_follow_45s.mp4](renders/mini_trail_follow_45s.mp4)，45 秒，1280×720，20 fps，H.264，900 帧。

| 图片路径（均在 renders/） | 内容 |
|---|---|
| [00_contact_sheet.png](renders/00_contact_sheet.png) | 总览联系表（建议先看） |
| [01_overview_top.png](renders/01_overview_top.png) | MuJoCo 全局俯视，门 1–8 标号 |
| [02_overview_bird.png](renders/02_overview_bird.png) | MuJoCo 斜俯视鸟瞰 |
| [03_width_transition.png](renders/03_width_transition.png) | 5 m 路宽过渡 |
| [04_narrow_drop_rail.png](renders/04_narrow_drop_rail.png) | 1.08 m 窄路、单侧落差、绿色细杆栏杆 |
| [05_sharp_bend.png](renders/05_sharp_bend.png) | 约 4.5 m 半径急弯 |
| [06_stairs_up.png](renders/06_stairs_up.png) | 上台阶，4×15 cm |
| [07_stairs_down.png](renders/07_stairs_down.png) | 下台阶，4×15 cm |
| [08_fork_tape.png](renders/08_fork_tape.png) | 岔路一与悬空黑带 |
| [09_fork_plain.png](renders/09_fork_plain.png) | 岔路二 |
| [10_uphill.png](renders/10_uphill.png) | 连续 15° 上坡 |
| [11_downhill.png](renders/11_downhill.png) | 连续 15° 下坡 |
| [12_leaf_shoulder.png](renders/12_leaf_shoulder.png) | 落叶路肩 |
| [13_log_rocks.png](renders/13_log_rocks.png) | 倒木、石块 |
| [14_m1_lidar_mounts.png](renders/14_m1_lidar_mounts.png) | M1 示意机体与前后 LiDAR 安装位置 |
| [15_tape_close.png](renders/15_tape_close.png) | 黑带近景：中心离地 40 cm，带宽 4.5 cm |
| [16_front_overlay_full.png](renders/16_front_overlay_full.png) | 前置 RGB + 完整点云投影 |
| [16_front_overlay_preview.png](renders/16_front_overlay_preview.png) | 前置 RGB + 降采样点云投影 |
| [16_front_rgb.png](renders/16_front_rgb.png) | 同一位置前置相机 RGB |
| [18_rear_overlay_full.png](renders/18_rear_overlay_full.png) | 后向 RGB + 完整点云投影 |
| [18_rear_overlay_preview.png](renders/18_rear_overlay_preview.png) | 后向 RGB + 降采样点云投影 |
| [18_rear_rgb.png](renders/18_rear_rgb.png) | 同一位置后向 RGB（辅助） |
| [19_airy_full_vs_preview.png](renders/19_airy_full_vs_preview.png) | 前后 Airy 融合点云：完整帧与降采样帧并排对比 |
| [20_width_and_elevation_profiles.png](renders/20_width_and_elevation_profiles.png) | 弧长—路宽曲线、高程和曲率 |
| [21_random_feature_locations.png](renders/21_random_feature_locations.png) | 随机地图特殊路段位置对比 |
| [22_video_frames.png](renders/22_video_frames.png) | 视频六帧抽查 |
| [random_001_overview.png](renders/random_001_overview.png) | 随机地图 MuJoCo 俯视预览 |
| [random_007_overview.png](renders/random_007_overview.png) | 随机地图 MuJoCo 俯视预览 |
| [random_019_overview.png](renders/random_019_overview.png) | 随机地图 MuJoCo 俯视预览 |
| [random_042_overview.png](renders/random_042_overview.png) | 随机地图 MuJoCo 俯视预览 |

文件大小和 SHA-256 见 [artifact_manifest.json](artifact_manifest.json)。离屏图使用英文短标签，中文解释和路径均列在此文档。

## 已处理的问题与边界

旧版硬切宽度、岔路覆盖高程和最近路径切换接缝都已处理；完整高度场通过加密检查。大范围阴影贴图曾产生三角斑点，近景现采用局部阴影范围并在创建渲染上下文前设置，避免把渲染伪影误当成地形缺陷。台阶立面与低多边形树冠仍可见离散面片，这是几何细节与 MuJoCo 光栅渲染的限制，不等于宽度连接断层。
