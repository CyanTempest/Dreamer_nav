"""Assemble the Chinese review report from measured artifacts only."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p0_common import ROOT,GEO,dump,json_ready
from .r8_p0_measure import decode,iou

def read(name):return json.loads((ROOT/'metrics'/f'r8_p0_{name}.json').read_text())
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'

def main():
 e=read('export');s=read('stations');a=read('analysis');v=read('ray_validation');t=read('throughput');b=read('bake_probe');ex=read('sensor_extrinsics');integ=read('integrity');ds=read('downsample')
 assert all(x['status']=='complete' for x in [e,s,v,t,b])
 base=np.load(GEO/'r8_baseline.npz');rays=np.load(GEO/'r8_sensor_bevs.npz');selected=[r for r in s['rows'] if r['pitch_deg']==-15];zones=read('baseline')['stations'];trav=[]
 for i,station in enumerate(base['stations']):
  x=decode(rays[f'bev_{int(station)}_-15']);rp=minimum_filter(((x[...,2]>.5)&(x[...,3]<.5)).astype(np.uint8),size=(9,5),mode='constant',cval=0)>0;gt=base['clean_priv_trav'][i,...,0]>0
  row=dict(station_m=float(station),ray_proxy_vs_priv_trav_iou=iou(rp,gt),ray_proxy_cells=int(rp.sum()),priv_trav_cells=int(gt.sum()))
  for label in ['clean','noisy']:
   q=base[label+'_decoded'][i];p=minimum_filter(((q[...,2]>.5)&(q[...,3]<.5)).astype(np.uint8),size=(9,5),mode='constant',cval=0)>0
   row[label+'_proxy_vs_ray_iou']=iou(p,rp);row[label+'_proxy_vs_priv_iou']=iou(p,gt)
  trav.append(row)
 dump(ROOT/'metrics/r8_p0_traversability.json',dict(definition='9x5 min-filter of known channel-2 support, zero padded; conservative footprint proxy, not original privileged collision/support truth',rows=trav))
 gpu=[json.loads(p.read_text()) for p in (ROOT/'metrics').glob('r8_p0_gpu_*.json')];peak=max(x['peak_mib'] for x in gpu)
 full=next(x for x in t['rows'] if x['name']=='full');nposes=b['proposed_full_bake']['pose_count'];raybake=full['median_seconds']/2048*nposes
 class_names={'unclassified':'未分类','road':'道路','stairs':'台阶/踢面','rail':'护栏','bank_soil':'坡面土/林地','rock':'石头','root':'树根','tree_trunk':'树干','tree_leaves':'树叶','shrub_mixed':'灌木（枝叶混合）','grass':'草','fern':'蕨','wall':'挡墙','sign':'标牌','tree_branch':'木质树枝'}
 text=f'''# R8-P0：LWY 完整几何与实机传感器射线仿真可行性

日期：2026-11-13。任务性质：几何、静态传感器与吞吐实验；没有启动 RL 训练、策略评估或队列。

## 结论与验收

完整几何导出和 GPU 射线投射可行，推荐下一步采用 **B：离线多位姿烘焙、保留视角相关可见性，训练时查询并重投影**。A 可作为离线参考或少量环境的校验器，当前实现不适合 2048 环境每个训练步都做全分辨率射线。

- 已导出 {e['objects']:,} 个可见对象、{e['triangles']:,} 个三角形，全部应用世界变换；无网格简化或植物删减。展开数组 {e['world_array_bytes']/1e9:.2f} GB，耗时 {e['seconds']:.2f} s。
- 10 个站点覆盖 10 个分区；每站均保存 **2×96×900 Airy** 点云与 **1280×800 Gemini、0°/−15°/−30°** 深度图，超过“至少 3 站”要求。
- 4090 实测 2048 位姿全分辨率纯射线每步中位数 **{full['median_seconds']:.3f} s**，即 {full['median_env_steps_per_second']:.1f} env-steps/s。这是整批真实执行，不是单环境乘法推算。
- 各受监控阶段本任务显存最高 **{peak} MiB**，低于 8 GB。新进程顺序执行 GPU 实验；未终止任何其他进程。显存每 0.25 s 采样，极短瞬时峰值不由该采样严格证明；分块缓冲及实测峰值均有充分余量。
- **−15° 为当前推荐的 provisional 俯仰角**，依据见下表；这不是实机安装标定结论。

主要产物：[导出指标](../metrics/r8_p0_export.json)、[外参](../metrics/r8_p0_sensor_extrinsics.json)、[逐站指标](../metrics/r8_p0_stations.json)、[差异汇总](../metrics/r8_p0_analysis.json)、[射线校验](../metrics/r8_p0_ray_validation.json)、[吞吐](../metrics/r8_p0_throughput.json)、[烘焙实测](../metrics/r8_p0_bake_probe.json)。

## 1. 环境、完整几何与语义

复用上次 `.venv_r8/` 的 Python 3.11.18、bpy 4.5.3 LTS、warp-lang 1.19.0，补装 imageio 到同一虚拟环境；依赖清单为 [r8_requirements.txt](../fullgeo/r8_requirements.txt)。上次 `r8_baseline.npz` 的 ZIP 完整性通过，复用其中静态基线。上次未留下完整网格；本次新建导出程序解决实例枚举生命周期问题并流式写盘。没有改写原有 nav3、配置、地图、checkpoint、旧指标。

Blender depsgraph 中可见对象全部转换为三角形并应用 `matrix_world`；当前文件没有对象修改器需要额外处理。8,642 个对象共享 524 个 evaluated mesh，唯一顶点总数 1,392,688。展开顶点数 {e['vertices']:,}，三角形数 {e['triangles']:,}。实例化是精确共享表达，不是 LOD。

世界坐标输出为 `fullgeo/r8_world_vertices.npy`（float32）、`r8_world_faces.npy`（uint32 全局顶点索引）、`r8_world_semantic.npy`（逐三角形 uint8）。`r8_instances.json` 保留每对象世界矩阵、包围盒、顶点和面偏移；`r8_template_*.npz` 保留模板顶点、三角形、逐面语义/材质、逐角点 UV，供 GPU 无损共享使用。模板数组约 {e['template_array_bytes']/1024**2:.2f} MiB，不含 BVH 和 alpha 贴图；展开数组约 {e['world_array_bytes']/1024**3:.2f} GiB。导出 CPU 最大 RSS {e['maxrss_mib']:.1f} MiB，完整展开数据通过流式文件写入，无须全部驻留内存。

{table(['语义','三角形数'],[(class_names[k],f'{n:,}') for k,n in e['class_triangles'].items()])}

语义映射规则：`surface_*` 先按共享 bundle 中 `road_*` 碰撞三角形质心匹配为道路（容差 2 mm），其余林地/土面为坡面土；混凝土面再按 bundle 台阶 station 范围标为台阶/踢面。植物按材质名分：`*_leaves` 为树叶，`*_branches` 为木质树枝，树干材质为树干，`fern_*`、`grass_*`、`shrub_*` 分别归蕨、草、灌木。非植物的 metal/rock/bark/concrete_wall/sign 分别归护栏/石头/树根/墙/标牌。原 renderer 的 pass_index 1/2/3/4 只提供道路/表面/构件/植物的大类，本次细分未把该粗标签冒充叶片真值。集合均为 Scene Collection，不能用于更细区分。灌木同一材质包含木枝和叶片，**不强行拆成“全软叶片”**。未分类三角形数为 0。

7 组植物 alpha 贴图使用原始分辨率、重复 UV、双线性采样和线性空间 0.5 截断；tree_small_02_leaves 的 sRGB alpha 先转换到线性空间，其余为 Non-Color，映射节点均为单位变换。透明命中继续追踪，最多 128 次；30 个正式扫描以及去植被反事实扫描的上限触发数均为 0。见 [alpha 审计](../metrics/r8_p0_alpha_audit.json)。这是光学表面几何近似，不能代表 850 nm/雷达波长下的真实叶片透射率。

### 坐标一致性

三种共享表面逐顶点最近邻误差均为 0；原三角形数量逐类一致。Blender 存储的 `geometry_sha256` 与当前 scene_bundle.json 的 SHA256 相同。{len(e['primitive_checks'])} 个可见 primitive 包围盒中心与原 box/ellipsoid 位置或 capsule 端点中点比对，最大误差 {max(x['center_error_m'] for x in e['primitive_checks'])*1000:.4f} mm。未作为可见物体导出的 {len(e['missing_primitive_names'])} 个 primitive 已核对全部为树干碰撞代理，原 renderer 明确跳过 `tree_trunk_collision*`，由真实树模型承担视觉几何，不能重复把胶囊添加为光学障碍。

独立检查所有 8,642 个实例，每实例抽 3 顶点、3 三角形：展开坐标与模板×矩阵误差为 {integ['expanded_geometry']['max_vertex_difference_m']} m，索引和语义错配为 0。对上次保护清单中的 {integ['protected_files_checked']} 个原文件重新哈希，差异为 0；详见 [完整性指标](../metrics/r8_p0_integrity.json)。这一抽查不等价于逐三角形扫描或对所有工作目录文件的全量审计。

## 2. 传感器位姿与射线模型

统一坐标：地面参考原点，x 向前、y 向左、z 向上。LWY 位姿使用当前基线地图采样的地面 z 和路线 yaw，机身不随坡面额外俯仰/横滚。所有正式对照处于同一站位，地图为 `maps_lwy/test_862428`，路线全长 1000 m。LWY 源包本身注明其布局/尺寸是视频参考近似，并非测绘重建；本报告没有将其等同实机赛道扫描。

### 双 Airy

直接复用 `vendor/map/sensors.py:rays`（与 nav3/raycheck.py 相同）以及 `nav3/calibration.py` 的实测链。每颗 96×900，方位间距 0.4°，全双雷达 172,800 条射线，量程 0.1–60 m，frame_index=0，无第 9 帧缺口。96 个名义垂直角完整写入外参 JSON；未取得每台设备的出厂逐束校准或时间戳。

原两颗地面高度（base_height=.54）为 0.540000 和 0.551898 m。按用户要求，保持相对链不变，整体下移约 0.050949 m，使均值为 0.495 m。高度来自用户站立估计，旋转与相对平移来自已有实测链。

{table(['传感器','最终 x/y/z（m）','native +z 半球轴在机身坐标'],[(name,', '.join(fmt(x,6) for x in np.array(ex['airy']['ground_transforms'])[i,:3,3]),', '.join(fmt(x,6) for x in np.array(ex['airy']['ground_transforms'])[i,:3,2])) for i,name in enumerate(['前 Airy','后 Airy'])])}

完整 4×4 矩阵、原矩阵、共同 z 偏移均在 [外参 JSON](../metrics/r8_p0_sensor_extrinsics.json)。

### Gemini 336L（provisional）

已检索 exp_nav3、~/visualnav、~/wyg_hiking 的相关配置。visualnav 的 camera_front.yaml 是 160×120 通用 USB 相机配置；wyg_hiking 的通用深度配置是 480×270、0.2–15 m，均未明确标注 Gemini。elf3.xml 包含 head_collision（相对 torso 的 z=.30，x≈−.01…+.02），但 torso 初始高度 1.1 m，与本任务底盘形态/高度没有可确认的坐标链，因此没有直接套用。

依据 [Orbbec 官方 Gemini 336L 页面](https://www.orbbec.com/products/stereo-vision-camera/gemini-336l/) 和 [Gemini 330 系列数据手册 V1.6](https://www.orbbec.com/wp-content/uploads/2025/06/Gemini-330-series-Datasheet-V1.6.pdf) 第 10、37 页：选择深度 1280×800，名义 30 fps，深度 FOV H90°/V65°，针孔内参 fx=640、fy={ex['gemini']['fy']:.6f}、cx=639.5、cy=399.5，畸变置零。实际仿真是静态一帧，并未模拟 30 Hz 时间过程。

光心位置 **(0.30, 0, 0.64) m**：z 来自用户估计；x=.30 为机身前部中心估计，y=0、roll=0、yaw=0 均 provisional。俯仰角用向下为负的 0°、−15°、−30°。以左 IR 光心为深度坐标，optical x 向右、y 向下、z 向前；每档完整变换矩阵已保存。

量程在**光轴 Z** 上裁剪为 0.5–20 m；95 mm 双目基线仅记录规格，未实现双目匹配。0.5 m 对应本次 provisional 视差搜索范围 128；手册也列出范围 256 时 Min-Z=0.25 m，因此不能把产品总量程 0.17 m 直接当作 1280×800 的统一近限。官方理想量程 0.25–6 m；20 m 上限是几何实验裁剪，远端输出不代表实际可靠深度。

### Warp 实现与校验

共享模板建立 524 个 Mesh BVH；实例包围盒建立顶层 Bvh，射线逆变换到各候选实例局部坐标，缩放距离后查询最近的有效 alpha 表面。使用完整场景，不裁掉远处植物，不复制 2048 份场景。展开世界网格的 17.95 GB 原始数组已超过显存限制，因此选择这种精确实例共享方案。

旧场景按 raycheck 的 body pitch、实测链、.54 m base、同一 group/self-hit 筛选，对三个旧位姿重新执行 MuJoCo mj_multiRay；几何 primitive 转三角网格后与 Warp 比较，旧曲面使用 96 圆周段、49 纬度环。初始校验暴露了验证转换器中 capsule/ellipsoid 枚举映射错误，修正后复用已存的独立 MuJoCo 射线重做对照；下表只列最终结果，日志保留前后记录。

{table(['旧站点/feature','射线数','命中状态一致率','共同命中 MAE（mm）','95% 误差（mm）','<1 cm 比例'],[(f"{r['station_m']:.3f}/{r['feature']}",r['rays'],f"{r['hit_agreement']*100:.6f}%",fmt(r['range_mae_m']*1000,5),fmt(r['range_p95_m']*1000,5),f"{r['within_1cm_fraction']*100:.6f}%") for r in v['legacy']])}

仍有 32 条命中状态不一致、4 条共同命中误差超过 1 cm：护栏柱→树冠 9.763 m，树皮→地面 7.073/0.568 m，地面→地面 1.065 m。曲面切线离散和 heightfield 边界可能导致远后景替代，但未逐条证明全部原因，因此**不是逐射线完全一致**。详见 [残差列表](../metrics/r8_p0_legacy_outliers.json)。另外，190 m 站周围 3 m 射程下，10,800 条射线直接比较实例 BVH 与实际展开的 4,380,305 三角形世界网格：命中完全一致，共同命中最大误差 {v['instance_vs_world']['range_max_m']*1000:.4f} mm（alpha 关闭），验证变换与实例查询。

## 3. BEV 定义与比较口径

当前 `LwyEnv/R7` 继承 FastEnv 感知，shape=64×64×6 uint8；分辨率 .125 m，x∈[−2,6)、y∈[−4,4)。原地图 .05 m 是源栅格分辨率，不是 BEV 分辨率。基线脚本在相同静态 state 验证 R7 与 LwyEnv 的 BEV uint8 最大差为 0。state 的 13 维仍为 3 个路线目标的二维偏移、gate 进度、3 维速度、3 维上一动作。

{table(['通道','含义及物理单位','uint8 编码/范围'],[['0','地面高程相对机器人脚下地面 z0，m','clip(h/4+.5,0,1)×255，约 −2…2 m'],['1','最高障碍表面减地面，m','clip(H/2,0,1)×255，0…2 m'],['2','几何支撑判据，非真实道路/足迹可通行标签','坡度<.8、障碍高<.22 m、相对地面>−.55 m；0/255'],['3','未知掩码','未知255；已知0'],['4','公开路线 corridor 先验','由地图和估计定位取得；不从点云语义推断'],['5','最近观测年龄/8 s','clip(age/8,0,1)×255']])}

`priv_trav` 是独立的仅解码目标，不是上述 6 通道：原实现由道路支持边界和碰撞距离做 9×5 footprint erosion，本任务从未把它提供给射线 BEV。将通道 2 称作“可通行 IoU”时，仅指**几何支撑代理**。

射线→BEV：首先只使用点的位置，按同一网格归并最低/最高返回；最低面缺测处用最近邻仅辅助地面估计，再作 5×5（.625 m）灰度开运算；没有点的格仍为未知。障碍高度=max(最高返回−估计地面,0)，通道 2 使用现有相同差分间距 .5 m 和阈值。采用与现有环境相同的首帧 world-aligned max scatter，并保留现有 .49×.29 m 半尺寸的近机身盲区；通道 4 复用同姿态公开路线先验。语义只用于独立归因审计，**不参与地面/支撑估计**。

这是一种可运行的传感器地面估计基线，不是已标定的实机地面分割器。最近邻/开运算、传感器高度更新、真实 3D 遮挡、射线密度都会影响差异，不能把全部误差都归因于植被。正式对照是 memory 为空的首帧，所以通道 5 与未知掩码一致，不能据此评价运动累积和 8 s 遗忘效果。训练基线分别关噪声和按默认开噪声；传感器参考为无测量噪声的几何返回。开噪声 baseline 包含原距离 dropout、.025 m 高度噪声和定位漂移。

IoU 使用阈值：已知格的通道 1≥.22 m 为障碍，已知格的通道 2>.5 为支撑。全图 IoU 将未知视为未检出，同时 JSON 另给共同已知格 IoU，避免掩盖可见性差异。高度主指标在共同已知格按解码后的高程计算；因此有 ±2 m 裁剪和量化。另用**仅用于审计的语义 terrain 回波**对原真值地面计算不依赖地面估计的高度误差。

### 逐站差异（Gemini −15°，全分辨率）

{table(['站点 m / zone','未知比例：净/噪/射线','支撑 IoU：净/噪','障碍 IoU：净/噪','高程 MAE cm：净/噪','地表回波 MAE cm'],[(f"{r['station_m']:g} / {zones[i]['zone']}",'/'.join(fmt(1-x/4096) for x in [r['clean']['reference_known_cells'],r['noisy']['reference_known_cells'],r['clean']['ray_known_cells']]),f"{fmt(r['clean']['support_iou_all'])}/{fmt(r['noisy']['support_iou_all'])}",f"{fmt(r['clean']['obstacle_iou_all'])}/{fmt(r['noisy']['obstacle_iou_all'])}",f"{fmt(r['clean']['height_mae_m']*100,2)}/{fmt(r['noisy']['height_mae_m']*100,2)}",fmt(r['terrain_truth_height_mae_m']*100,2)) for i,r in enumerate(selected)])}

10 站宏平均：净基线支撑 IoU={a['aggregate']['clean']['support_iou_all']:.3f}，障碍 IoU={a['aggregate']['clean']['obstacle_iou_all']:.3f}；共同已知格分别为 {a['aggregate']['clean']['support_iou_common']:.3f}/{a['aggregate']['clean']['obstacle_iou_common']:.3f}。高程 MAE 站点宏平均 {a['aggregate']['clean']['height_mae_m']*100:.2f} cm、按共同格加权 {a['aggregate']['height_cell_weighted_mae_m']*100:.2f} cm；地表真实返回的加权误差 {a['aggregate']['terrain_height_cell_weighted_mae_m']*100:.2f} cm。后者更接近源网格与栅格坐标/采样误差，前者还包含地面估计误差。

![10 分区对照汇总](../figures/r8_p0_summary.png)

逐站逐通道 MAE（全图；H0/H1 为 m，C2…C5 为 0…1；共同已知格版本见 JSON）：

{table(['站点','基线','H0','H1','C2 支撑','C3 未知','C4 路线','C5 年龄'],[(f"{r['station_m']:g}",label,*[fmt(x,4) for x in r[label]['channel_mae_all']]) for r in selected for label in ['clean','noisy']])}

更保守的 9×5 支撑代理侵蚀，与原 priv_trav 分开列出；由于通道 2 在坡面也可能为真，而 priv_trav 明确排除道路外坡面，这些不是同一语义的准确率：

{table(['站点','净 BEV 代理 vs 射线代理 IoU','噪 BEV 代理 vs 射线代理 IoU','射线代理 vs priv_trav IoU'],[(f"{r['station_m']:g}",fmt(r['clean_proxy_vs_ray_iou']),fmt(r['noisy_proxy_vs_ray_iou']),fmt(r['ray_proxy_vs_priv_trav_iou'])) for r in trav])}

### 植被、悬垂物与硬障碍

按传感器估计地面上方 [.22,.80] m 作为临时车体相关高度带，>.80 m 单列悬垂物；.80 m 是审计阈值，尚不是实测机器人扫掠体积。叶片、草、蕨作为可能柔性的类别，木质树枝/树干/石头/护栏/墙/标牌为硬类；灌木为混合类单列。实际软硬、弯折和通行力没有物理验证，不能由光学命中断言“实际可穿过”。

{table(['站点','低处叶/草/蕨格','低处灌木格','低处硬类格','其中地图高度包络缺失','仅高处硬类格'],[(f"{r['station_m']:g}",r['low_soft_cells'],r['low_shrub_cells'],r['low_hard_cells'],r['low_hard_missing_from_truth_cells'],r['overhead_hard_without_low_hard_cells']) for r in a['rows']])}

10 站合计 {a['aggregate']['clearance_counts']['soft_only_low_cells']:,} 格在低处有叶/草/蕨返回而无同高度硬类，{a['aggregate']['clearance_counts']['shrub_only_low_cells']:,} 格有灌木而无同高度硬类；后者绝不能全部当软植被。最高表面式通道 1 还把 {a['aggregate']['clearance_counts']['overhead_hard_without_low_hard_cells']:,} 个只有高处硬回波的格当作高障碍，体现单个高度上包络不能区分“上方有树枝”与“车体会撞上”。同一格允许出现多种类别，类别计数不可直接相加。

低处硬回波而当前真值高度包络<.22 m 的格共 {a['aggregate']['clearance_counts']['low_hard_missing_from_truth_cells']}，其中木枝 {a['aggregate']['hard_missing_class_counts']['branch']}、树干 {a['aggregate']['hard_missing_class_counts']['trunk']}、石头 {a['aggregate']['hard_missing_class_counts']['rock']}、护栏 {a['aggregate']['hard_missing_class_counts']['rail']}，余下为墙/标牌及类别重叠。木枝属于当前仅树干胶囊近似未覆盖的真实几何；树干视觉模型与胶囊投影也有差异。零星石头/护栏/墙缺格可能是栅格边缘、采样和地面估计误差，不能称整个物体都从原地图缺失。另有 {a['aggregate']['clearance_counts']['low_hard_missing_due_unknown_cells']} 格在原地图有足够高度，但当前净 BEV 未看见。这里“缺失”按高度包络定义，不等于碰撞距离字段中完全没有 collider。

去掉所有叶/草/蕨及灌木、保留木质树枝的反事实重投射，在 10 站共改变 {a['aggregate']['counterfactual']['changed_cells']:,} 格，{a['aggregate']['counterfactual']['obstacle_disappeared_cells']:,} 格障碍消失，新增已知 {a['aggregate']['counterfactual']['known_gain_cells']:,} 格。这证明植物对输入影响很大，但其中包含有木质部分的灌木，是影响上界而非“全可穿越”结论。alpha 关闭的双 Airy 对照平均改变 {a['aggregate']['mean_alpha_changed_cells']:.1f} 格；沿用旧雷达高度对照平均改变 {a['aggregate']['mean_legacy_height_changed_cells']:.1f} 格（任一 uint8 通道不同即计数），说明透明贴图和外参不能略去。

## 4. Gemini 俯仰角与覆盖

{table(['俯仰角','相机已知格','相机路线格','融合已知格','融合路线格','相对 Airy 新增格','净基线支撑 IoU'],[(r['pitch_deg'],fmt(r['mean_camera_known_cells'],1),fmt(r['mean_camera_corridor_known_cells'],1),fmt(r['mean_fused_known_cells'],1),fmt(r['mean_fused_corridor_known_cells'],1),fmt(r['mean_added_cells'],1),fmt(r['mean_support_iou_clean'])) for r in a['pitch']])}

以上为 10 站宏平均。−15° 相比 0° 观察到更多路线地面；−30° 相比 −15° 的相机路线覆盖增加有限，却减少总覆盖和相对雷达新增覆盖，融合路线覆盖也没有明显改善。因此建议先用 −15°，并在拿到实际 K/D、横向位置、俯仰测量及下台阶视频后重新比较。上表统计点云是否落入格，不代表有效双目匹配概率或安全通行概率。

![俯仰覆盖](../figures/r8_p0_pitch.png)

每站覆盖范围、深度与 BEV 对比均已保存：

{table(['站点 m','BEV 对比','三档深度','低处/悬垂归因','全部 6 通道差值'],[(f'{int(s)}',f'[图](../figures/r8_p0_bev_{int(s):04d}.png)',f'[图](../figures/r8_p0_depth_{int(s):04d}.png)',f'[图](../figures/r8_p0_clearance_{int(s):04d}.png)',f'[图](../figures/r8_p0_channels_{int(s):04d}.png)') for s in base['stations']])}

BEV 图高程色轴 −1…1 m、障碍色轴 0…1 m，超过显示范围饱和；定量指标使用完整编码范围。BEV 数值图白色为空/未知；6 通道差值图白色为零差值。深度为光轴 Z，白色为无有效返回。

## 5. 2048 环境吞吐与降采样损失

4090，共享 GPU，未清场。每档 2048 个不同位姿均匀分配到 10 站，横移±.25 m、yaw 扰动±.3 rad；用于吞吐的横移地面高度保持站点值，因此不是可行走状态测试。每档 warm-up 后执行三次完整批次；全分辨率分块 16 环境，其余分块 64。每块包含真实输出写入、位姿上传及同步，统计 wall time，输出复用释放。未计 BVH 初建、点云回传、BEV 汇聚、双目匹配、JAX 互操作或训练。

{table(['档位','每颗 Airy','Gemini 高×宽','每环境射线','2048 批次 s：min/median/max','env-step/s'],[(r['name'],f"{r['airy_channels']}×{r['airy_azimuths']}",str(r['camera_shape']) if r['camera_shape'] else '关闭',f"{r['rays_per_env']:,}",f"{min(r['repeats_seconds']):.3f}/{r['median_seconds']:.3f}/{max(r['repeats_seconds']):.3f}",fmt(r['median_env_steps_per_second'],1)) for r in t['rows']])}

全分辨率每批 {full['total_rays']:,} 条射线，若一次保留所有 distance/face/instance/cap 输出就需约 {full['total_rays']*16/1e9:.1f} GB，因此分块是内存约束下的正常实现。测量中场景常驻共享，全部实例保留。共享 GPU 的竞争与功耗状态会影响绝对时间；这些数值只代表本次实测实现。

为了量化降采样代价，从全分辨率返回严格取相同 stride 的子集，重新构建 BEV，而非只测速度：

{table(['双轴 stride','已知格（全分辨率=3569.3）','对全分辨率支撑 IoU','障碍 IoU','共同格地面 MAE cm'],[(r['stride'],fmt(r['ray_known_cells'],1),fmt(r['support_iou_all']),fmt(r['obstacle_iou_all']),fmt(r['height_mae_m']*100,2)) for r in ds['aggregate']])}

这些档位是射线子采样，未声称对应硬件实际的分辨率/FOV/近限组合。极度下采样虽快，缺测和植被采样差异显著，不能直接替代完整参考。

![吞吐](../figures/r8_p0_throughput.png)
![下采样误差](../figures/r8_p0_downsample.png)

## 6. A/B 成本与推荐

**A：在线射线。** 优点是位姿变化、真实多层几何和遮挡自然保留，可直接添加物体；当前全分辨率 {full['median_seconds']:.2f} s/2048 步，仅射线就是 5 Hz 的 .2 s 控制周期的 {full['median_seconds']/.2:.1f} 倍。即使不要求实时，RL 样本吞吐也会显著下降。单独双 Airy 的数字见上表；尚无 ray→GPU BEV→JAX 的端到端接入成绩。可以保留少量在线高分辨率探针作为审计，不建议全训练批次每步在线。

**B：离线烘焙。** 本次实做 190/320/480 m 三站，横偏 −.3/0/+.3 m、yaw 偏移 −15/0/+15°，共 27 个全分辨率位姿，不是只给公式。每姿态“点云下载 + NumPy BEV + 原地图公共先验查询”的流水线平均 {b['bake_mean_seconds_per_pose']:.4f} s，其中射线均值 {b['bake_ray_mean_seconds_per_pose']:.4f} s；不含最终压缩落盘，warm-up 排除。小表驻 GPU 的 2048 环境 gather+原距离 dropout 微基准为 **{b['lookup_median_seconds']*1000:.3f} ms/步**，输出 {b['lookup_output_bytes']/1e6:.2f} MB。它仅覆盖驻留 27 视图的小热表，不含大表 cache miss、查最近位姿、重投影、时间累积或策略，因此不能当生产系统延迟承诺。

按待选设计“沿线 .5 m、5 个横偏、7 个 yaw”估计为 {nposes:,} 个位姿：单份 64×64×6 uint8 表需 {b['proposed_full_bake']['uint8_bev_bytes']/1e9:.3f} GB；按批量射线实测外推纯射线约 **{raybake/60:.2f} 分钟**，按当前串行 Python 烘焙流水线外推约 **{b['proposed_full_bake']['full_serial_pipeline_seconds_estimate']/3600:.2f} 小时**。两者均为明确外推，不是已经完成全 1 km 烘焙；额外置信度/多层高度/视角可见性会增加存储，不能按单份 BEV 大小包办。生成多路线、姿态俯仰/横滚和动态植被亦另计。

B 的关键风险是位姿量化。实测 6 个中间横偏/朝向 holdout，用最近缓存直接取图、没有重投影：

{table(['站点','横偏 m / yaw°','已知掩码 IoU','支撑 IoU','共同格高程 MAE cm'],[(r['station_m'],f"{r['offset_m']}/{r['yaw_offset_deg']}",fmt(r['known_iou']),fmt(r['support_iou']),fmt(r['height_common_mae_m']*100,2)) for r in b['holdout']])}

因此推荐的 B **不能直接把最近 body-frame 图交给策略**。应把命中点/地面估计投到世界坐标，保存支持证据、障碍高度分层与每个采样视角的可见性，查询时按真实姿态重投影；跨视角融合的地图不能把所有历史可见格当作当前帧都可见。通过邻近视角选取或保守可见性融合，再加姿态相关遮挡、距离噪声和 8 s memory；公开路线先验独立查询。此组合尚未实现，P0 给出了其成本依据和必须验证的误差项。

## 7. 下一步与尚未完成事项

1. 获取 Gemini SDK 实机内参、distortion、视差模式及实际外参；测量相机 x/y/pitch、两颗雷达地面高度、整机扫掠体积。用现有三档深度图确定 −15° 是否仍合适。
2. 固定传感器→BEV 的实机同源地面估计算法，显式区分低处障碍、悬垂净空和柔性植被的置信度。先保持 6 通道契约；不要把私有路面支持或植物语义直接写进 actor 通道。当前地面开运算只是 P0 基线。
3. 先做小段 B 的世界坐标、多视角可见性缓存与连续轨迹重投影，针对上表 holdout 误差设验收；再扩到 1 km、多横偏/yaw。禁止用跨视角“看穿遮挡”的全局拼图冒充单帧传感器。
4. 做传感器与机器人自遮挡、pitch/roll、时间同步/扫描运动、双目匹配失败、日照/反射、量程噪声、植被透射/风动等校准。当前只模拟理想光学首表面，未实现这些物理机制；近机身 BEV 盲区也不等于真实机壳的射线自遮挡。
5. 在同一 GPU 时间片下测 GPU BEV 汇聚与 JAX 接口，再决定少量在线 probe 的频率。当前吞吐只到射线阶段，离线大地图烘焙与生产查询尚未完成；没有启动任何训练来填补这些结论。

## 8. 复现与文件索引

代码均为新文件：`nav3/r8_p0_export.py`、`r8_p0_ray.py`、`r8_p0_measure.py`、`r8_p0_validate.py`、`r8_p0_benchmark.py`、`r8_p0_analyze.py`、`r8_p0_downsample.py`、`r8_p0_bake_probe.py`、`r8_p0_finalize.py`。前次留下的 r8_p0_common.py / baseline.py / env.sh 只读复用。

```bash
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source scripts/r8_p0_env.sh
# 下列各命令重建相应 R8 产物；GPU 阶段按顺序执行。
.venv_r8/bin/python -u -m nav3.r8_p0_export > logs/r8_p0_export_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_ray > logs/r8_p0_ray_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_measure > logs/r8_p0_stations_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_validate > logs/r8_p0_validation_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_benchmark > logs/r8_p0_throughput_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_analyze > logs/r8_p0_analysis_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_downsample > logs/r8_p0_downsample_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_bake_probe > logs/r8_p0_bake_reproduce.log 2>&1
.venv_r8/bin/python -u -m nav3.r8_p0_finalize > logs/r8_p0_finalize_reproduce.log 2>&1
```

`fullgeo/r8_airy_XXXX.npz`：有效点的 body/world XYZ、语义，以及每颗 96×900 距离/面/实例 ID；`r8_gemini_XXXX_pitchP.npz`：800×1280 光轴深度与语义、位姿（无返回为 NaN）；`r8_sensor_bevs.npz`：各站/俯仰 BEV 与去植被反事实；`r8_bake_probe.npz`：27 视图缓存。语义审计图为 `figures/r8_p0_semantic_*.png`，低处/悬垂图为 `r8_p0_clearance_*.png`。各 GPU 阶段的 PID、峰值和采样序列见 `metrics/r8_p0_gpu_*.json`；长步骤日志均在 `logs/r8_p0_*.log`。
'''
 (ROOT/'reports/R8_P0_FULLGEO.md').write_text(text)
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');names=[r['name'] for r in t['rows']];vals=[r['median_seconds'] for r in t['rows']]
 axs[0].bar(names,vals);axs[0].set(ylabel='Seconds / 2048 environment steps',title='Measured ray casting, alpha enabled');axs[0].tick_params(axis='x',rotation=20)
 axs[1].bar(['Full ray batch','Hot-table lookup + dropout'],[vals[0]*1000,b['lookup_median_seconds']*1000]);axs[1].set_yscale('log');axs[1].set(ylabel='Milliseconds (log scale)',title='Different scope: see report limitations');fig.savefig(ROOT/'figures/r8_p0_throughput.png',dpi=160);plt.close(fig)
 for k,station in enumerate(base['stations']):
  ref=decode(rays[f'bev_{int(station)}_-15']);fig,axes=plt.subplots(2,6,figsize=(18,6),layout='constrained')
  for row,label in enumerate(['clean','noisy']):
   difference=ref-base[label+'_decoded'][k]
   for ch,(title,lim) in enumerate(zip(['Ground [m]','Obstacle [m]','Support','Unknown','Route prior','Age / 8s'],[2,2,1,1,1,1])):
    ax=axes[row,ch];im=ax.imshow(difference[...,ch],origin='lower',extent=(-4,4,-2,6),vmin=-lim,vmax=lim,cmap='RdBu_r');ax.set(title=f'Ray - {label}: '+title,xlabel='Left [m]',ylabel='Forward [m]');fig.colorbar(im,ax=ax,shrink=.7)
  fig.suptitle(f'All six channels, station {station:g} m, fused pitch -15');fig.savefig(ROOT/f'figures/r8_p0_channels_{int(station):04d}.png',dpi=140);plt.close(fig)
 files=[p for folder,pattern in [('metrics','r8_p0_*.json'),('figures','r8_p0_*.png'),('nav3','r8_p0_*.py')] for p in (ROOT/folder).glob(pattern)]
 acceptance=dict(status='complete',report='reports/R8_P0_FULLGEO.md',stations=len(base['stations']),full_airy_cloud_files=len(list(GEO.glob('r8_airy_*.npz'))),full_gemini_depth_files=len(list(GEO.glob('r8_gemini_*.npz'))),metric_files=len(list((ROOT/'metrics').glob('r8_p0_*.json'))),figure_files=len(list((ROOT/'figures').glob('r8_p0_*.png'))),gpu_peak_mib=peak,benchmark_full_seconds=full['median_seconds'],recommended='B with world reprojection and view-conditioned visibility; camera pitch -15 provisional',no_training_or_policy_evaluation_started=True,protected_hash_mismatches=integ['protected_mismatches'],files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files+[ROOT/'reports/R8_P0_FULLGEO.md'] if p.name!='r8_p0_acceptance.json'])
 dump(ROOT/'metrics/r8_p0_acceptance.json',acceptance);print(json.dumps({k:x for k,x in acceptance.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
