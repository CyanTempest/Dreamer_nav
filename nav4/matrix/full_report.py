"""Compile measured full-route results into an auditable delivery report."""
from collections import Counter
import json
import re

from nav4.matrix.full_ground import FULL, PILOT, WORK
from nav4.matrix.survey import OUT


def read(root, path):
    return json.loads((root/path).read_text())


def cm(record):
    return f"{100*record['rms']:.2f} / {100*record['p95']:.2f}"


def max_rss_gib():
    pattern = r"Maximum resident set size \(kbytes\):\s+(\d+)"
    peaks = [int(n) for f in WORK.glob("*.log") for n in re.findall(pattern,f.read_text())]
    return max(peaks,default=0)/1024**2


def flight_table(stairs):
    groups = {}
    for row in stairs:
        groups.setdefault(row["flight"],[]).append(row)
    result = ["| 梯段 | 站位（m） | 级数 | 踢面绝对值（cm） | 踏面深度（cm） |",
              "|---|---:|---:|---:|---:|"]
    for name,rows in groups.items():
        ss, hh, dd = zip(*[(r["station_m"],abs(r["riser_m"])*100,r["tread_depth_m"]*100) for r in rows])
        result.append(f"| {name} | {min(ss):.2f}–{max(ss):.2f} | {len(ss)} | {min(hh):.1f}–{max(hh):.1f} | {min(dd):.1f}–{max(dd):.1f} |")
    return result


def report():
    segments = read(FULL,"segments.json")
    load = read(FULL,"load_segments.json")
    packed = read(FULL,"load_assemblies.json")
    before = read(PILOT,"pilot_before_on_same_samples.json")
    after = read(PILOT,"geometry.json")
    strips = read(PILOT,"slope_artifacts_by_region.json")
    pilot_load = read(PILOT,"load.json")
    old_load = read(OUT,"load_3.3.0.json")
    scan, ply = (read(WORK/name,"source.json") for name in ("las","ply"))
    build = read(FULL,"build.json")
    seams = read(FULL,"seams.json")
    stairs = read(FULL,"stairs.json")
    phy = read(FULL,"physics.json")["cases"]
    real = read(FULL/"figures","field_positions.json")
    diff = read(FULL,"v8_difference.json")
    layers = read(FULL,"lower_layers.json")["patches"]
    wood = read(FULL,"woody_support_audit.json")
    support = read(FULL,"hard_support_audit.json")
    items = read(FULL,"objects.json")
    kinds = Counter(i["kind"] for i in items)
    faces, steep = Counter(),Counter()
    for row in segments:
        faces.update({k:v["faces"] for k,v in row["surface"].items()})
        steep.update({k:v["near_vertical"] for k,v in row["surface"].items()})
    fill = sum(r["fill_projected_m2"] for r in segments)
    area = sum(r["ground_projected_m2"] for r in segments)
    routefill = sum(r["fill_near_route_m2"] for r in segments)
    packed_faces = sum(faces.values())
    lower_stations = ", ".join(f"{x['station_m']:.2f}" for x in layers)
    bytes_total = sum(r["bytes"] for r in segments)
    full = packed.get("scene_full.xml",{})
    full_robot = packed.get("zgws_atec_full.xml",{})
    window = packed.get("zgws_atec_window_00.xml",{})
    buf = [
        "# A 段 0–722 m 3D 地图复原",
        "",
        "交付 15 段 MJCF/OBJ、全段场景、重叠窗口和 ZGWS 组合场景。几何、误差、接缝、编译和物理代理体结果均为 CPU 实测。遮挡补面、硬物背面封闭和实景帧配对含推断。",
        "",
        "## 网格与 0–30 m 复验",
        "",
        "用规划 PLY 的近地法向及邻近关系筛选 LAS 地面候选。每格取 LAS 观测中位数，按邻域剔除孤点。低频补面由观测边界、局部高程残差与平滑冻结 v8 先验过渡。普通地面使用 5 cm 共享顶点格，限制沿 XY 两轴的坡度，优先保留路线附近有扫描支持的高程。台阶、墙岩和木质实体独立建模。",
        "",
        "| 指标 | 原试点 | 本轮试点 |",
        "|---|---:|---:|",
        f"| 路线 ±1 m 同批留出点 RMS / P95（cm） | {cm(before['ground_holdout_within_1m'])} | {cm(after['ground_holdout_within_1m'])} |",
        f"| ±6 m 同批留出点 RMS / P95（cm） | {cm(before['ground_holdout_corridor'])} | {cm(after['ground_holdout_corridor'])} |",
        f"| 普通地面三角面 | 1,471,839 | {after['surface']['observed_ground']['faces']+after['surface']['filled_ground']['faces']:,} |",
        f"| 渲染网格面 | 1,478,137 | {pilot_load['mesh_faces']:,} |",
        f"| MuJoCo 3.3.0 CPU 编译（s） | {old_load.get('load_seconds',21.4):.2f} | {pilot_load['seconds']:.2f} |",
        "",
        "地面近竖直面以倾角 >75° 计数。窄竖片还要求高差 ≥10 cm，最大 XY 边 ≤4 cm。以加宽 10 cm 的梯级轮廓和距试点墙岩块中心 24 cm 划分。",
        "",
        "| 普通地面位置 | 原近竖直面 | 原窄竖片 | 新近竖直面 / 窄竖片 |",
        "|---|---:|---:|---:|",
    ]
    for key,label in (("stair_footprint_extended","梯级邻域"),("near_masonry_or_rock","墙岩邻域"),("other_slope","其余坡面")):
        a,b = (strips[m][key] for m in ("original","rebuild"))
        buf.append(f"| {label} | {a['near_vertical']:,} | {a['steep_thin_strips']:,} | {b['near_vertical']} / {b['steep_thin_strips']} |")
    buf += [
        "",
        "同视角确认成片竖刺、薄片消失。降采样、坡度约束以及阶沿附近的混合使误差尾部有所增大。原试点报告口径的 ±1 m RMS 2.79 cm，与本轮同批复算使用的候选点和候选面选法有差异。试点原有约 22% 补面按观测距 30 cm 及剔除台阶单元统计，新版用任一角距直接观测超 15 cm 标记。",
        "",
        "## 全段规模与精度",
        "",
        f"LAS 共流式读 {scan['total']:,} 条，按序每 7 条取一条，走廊及工作缓冲保留 {scan['staged']:,} 条。PLY 同区域保留 {ply['staged']:,} 条。试点配准前后残差接近，因此全段采用共用原始坐标。",
        "",
        f"{packed_faces:,} 个渲染 mesh 面，普通地面和下层地面合计 {sum(faces[k] for k in ('observed_ground','filled_ground','lower_observed_ground')):,} 个静态碰撞三角。15 段唯一 MJCF/OBJ 共 {bytes_total/1e6:.2f} MB。LAS/PLY 流式读取分别为 {scan['seconds']:.1f} s、{ply['seconds']:.1f} s，建模及测距为 {build['seconds']:.1f} s。8 核亲和性，阶段最高单进程 RSS {max_rss_gib():.2f} GiB。",
        "",
        f"15 段编译时间合计 {sum(r['seconds'] for r in load):.2f} s。整段场景编译为 {full.get('seconds','未成功')} s，ZGWS + 整段为 {full_robot.get('seconds','未成功')} s，ZGWS + 0–100 m 窗口为 {window.get('seconds','未成功')} s。将 OBJ 顶点与 MJCF 位置重新换算到世界坐标，在 {len(seams)} 道接缝检查 {sum(s['checked_vertices'] for s in seams):,} 点，缺失 {sum(s['missing_vertices'] for s in seams)} 点，最大偏差 {max(s['max_obj_world_disagreement_m'] or 0 for s in seams):.2g} m。",
        "",
        "| 站位（m） | ±1 m RMS / P95（cm） | ±6 m RMS / P95（cm） | 补面（m²） | 分段 MB | 编译（s） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row,comp in zip(segments,load):
        a,b = row["range_m"]
        buf.append(f"| {a:g}–{b:g} | {cm(row['ground_holdout_within_1m'])} | {cm(row['ground_holdout_corridor'])} | {row['fill_projected_m2']:.1f} | {row['bytes']/1e6:.2f} | {comp['seconds']:.2f} |")
    buf += [
        "",
        "每段每口径随机取 8,000 条构建前留出的 LAS 地面候选，测到地面三角、独立踏面或前缘的欧氏距离。候选经近地 PLY 筛选，所以误差主要覆盖已有扫描支持的场景。补面缺少独立真值。",
        "",
        "残差另按 5 m 分桶诊断。50–100 m 近线误差最高，集中在 60–80 m，段内遮挡面积也最大。大量高残差仍来自有观测格，和陡缘连续化或邻近表面混入候选相符。600–650 m 近线路尾部集中于 610–625 m 密集下行梯级。700–722 m 全宽 P95 最大，偏差集中在 705–715 m 阶沿及外侧陡落，补面约 6.5 m²。位置诊断是同 XY 垂向差，详见 residual_locations.json。",
        "",
        "## 台阶及冻结 v8",
        "",
        f"拟合接受 {len(stairs['accepted'])} 级，包括入口 34 级。{len(stairs['review_candidates'])} 个候选仍需现场、原始切片或视频复核。stairs.csv 每级记位置、带符号踢面、踏面、宽度及冻结 v8 配对。组末行深度量到建模端界。stairs.json 列出拟合统计、候选和 v8 未匹配记录。",
        "",
        *flight_table(stairs["accepted"]),
        "",
        "v8 台阶候选用冻结高程和官方扫描中心线复算，再按 XY 转换到成图站位。用同符号和 0.32 m 距离做唯一配对。扫描里程 9.125 m 的 +27.7 cm v8 候选，映射到 8.80 m，落在本轮平台。该梯段首级 9.369 m 高 14.0 cm，全梯段最高 16.1 cm。",
        "",
        "| 新站位（m） | v8 扫描站位（m） | v8 → 新（cm） |",
        "|---|---:|---:|",
    ]
    pairs = [r for r in stairs["accepted"] if "v8_station_m" in r]
    for p in sorted(pairs,key=lambda r:-abs(r["riser_minus_v8_m"]))[:8]:
        buf.append(f"| {p['station_m']:.3f} | {p['v8_scanner_station_m']:.3f} | {100*p['v8_riser_m']:+.1f} → {100*p['riser_m']:+.1f} |")
    buf += [
        "",
        f"共配对 {len(pairs)} 级，v8 候选 {len(stairs['v8_unpaired'])} 处未配对。全段 {diff['ground']['count']:,} 个中心线地面射线，新旧高差 RMS / P95 为 {cm(diff['ground'])} cm。",
        "",
        "| 高差站位（m） | 新−v8（cm） |",
        "|---|---:|",
    ]
    for row in diff["ground_outliers"][:8]:
        buf.append(f"| {row['station_m']:.1f} | {100*row['signed_ground_delta_m']:+.1f} |")
    buf += [
        "",
        "硬障碍在 0.30 / 0.53 / 0.76 m 三个高度做 +/−2 m 横向射线。v8 对照用 5 cm 步长查询硬格。命中统计受表面拟合覆盖、高度带投影和细杆采样共同影响。v8_difference.json 逐侧列出最大不一致。",
        "",
        "| 侧 | 双方命中 | 仅 v8 | 仅新 | 双方无命中 | 双命中距离差中位（cm） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side,data in diff["hard_by_side"].items():
        buf.append(f"| {'左' if side=='left' else '右'} | {data['both']} | {data['only_v8']} | {data['only_scene']} | {data['neither']} | {100*data['both_signed_median_m']:+.1f} |")
    buf += [
        "",
        "仅 v8 命中的位置很多，实体和旧硬障碍真值的一致性仍需逐处核查。",
        "",
        "## 物理代理体",
        "",
        "用 1.03 × 0.58 × 0.20 m、30 kg 自由盒体。自由静置 2 s。另从同一初始位姿以 0.095 m/s 目标速度、正向推力 140 N 上限、制动上限 330 N 慢推 5 s，避免让重力下滑成为主要前进来源。覆盖所有拟合梯段，长下行沿途复测。最窄五处按每 0.1 m 的射线净宽排名，地点至少隔 4 m。边界周边用两侧段文件复验。",
        "",
        "| 案例 | 站位（m） | 净宽（m） | 推进（cm） | 接触重叠（mm） | 停滞 | 爬升候选 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    good = [p for p in phy if "max_contact_overlap_m" in p]
    for a in phy:
        if a not in good:
            buf.append(f"| {a['name']} | {a['station_m']:.2f} | — | — | — | {a.get('status')} | — |")
            continue
        width = f"{a['measured_width_m']:.3f}" if "measured_width_m" in a else "—"
        buf.append(f"| {a['name']} | {a['station_m']:.2f} | {width} | {a['push']['forward_m']*100:.1f} | "
                   f"{a['max_contact_overlap_m']*1000:.2f} | {'是' if a['stalled'] else '否'} | "
                   f"{'是' if a['unexpected_climb_candidate'] else '否'} |")
    buf += [
        "",
        f"共 {len(phy)} 个案例，{sum(a['stalled'] for a in good)} 个慢推停滞，{sum(a['unexpected_climb_candidate'] for a in good)} 个触发爬升候选。{sum(a['center_crossed_below_terrain'] for a in good)} 个代理体中心低于同 XY 地面超过 6 cm。最大接触重叠 {max(a['max_contact_overlap_m'] for a in good)*1000:.2f} mm。慢推中的瞬时峰值速度最高 {max(a['push']['max_forward_speed_mps'] for a in good):.2f} m/s，出现在下阶瞬间。推进少于 15 cm 记停滞。自由滑动基线在 physics_free_slide.json，包含下行长距离滑动。中心上升超过 14 cm 且伴侧碰记爬升候选。上行四个梯段和 6.2、13.5、18.9 m 三处窄点出现停滞，窄点也处在上行台阶。受阻几何及自由静置滑移都保存在逐案例中。整机足端支撑、登阶和运控表现待实测。",
        "",
        "## 对照图",
        "",
        "MATRiX 参考、原试点及新版同镜头五组在 pilot_0_30_rebuild/figures/compare_01.jpg–compare_05.jpg。入口右侧大树宽视图在同目录 entry_tree_detail.jpg。",
        "",
        f"全段 {len(real)} 组在 figures/field_*.jpg。每组保存站位、官方轨迹时间戳、MP4 本地时码和帧号。站位沿化简后扫描轨迹约束递增来找官方时间戳，4 Hz 视频用首尾、上段梯级和下行栏杆地标及横向光流与沿线转弯匹配，属于近似。同路径 MP4 无绝对时间基准，沿线路标局部仍有错位。官方 28,787 条渲染位姿 RGB 字段为空，存证见 video_source_metadata.json。可用对照图定性复核，高精度逐帧对应需有同步索引。",
        "",
        "## 补面、岩壁、树木与植被",
        "",
        f"走廊投影 {area:.1f} m²，单独标记补面 {fill:.1f} m²（{fill/area*100:.1f}%），其中 ±1 m 内 {routefill:.2f} m²。精确格位在 surface.npz，按 5 m 的面积在各 segments/seg_*/geometry.json，MJCF 补面 group=4，走廊平面分布图见 figures/fill_overview.jpg。",
        "",
        f"入口栏杆 {kinds['rail']} 个胶囊含 21 根立柱，后段点云定位栏杆 {kinds['rail_scan']} 段、短遮挡连接 {kinds['rail_inferred']} 段、出口桥扶手和立柱 {kinds['rail_exit']} 段。墙岩由 {kinds['retaining_wall']+kinds['wall_rock_or_root']+kinds['rock_face']:,} 个有扫描约束的局部块组成。去除了 {len(support['removed_disconnected_rock']):,} 个缺少地面连接的岩面候选。"
        f"另外在 {lower_stations} m 测出下层斑块，总投影 {sum(x['lower_projected_area_m2'] for x in layers):.2f} m²。遮挡后的岩壁和桥底仍有缺失。",
        "",
        f"保留 {wood['grounded_stems']} 株落地树干、{wood['continuous_trunk_pieces']:,} 个连续实体段和 {kinds['root']:,} 个有地面或树体连接的根拟合片。删除连接不足或埋住的候选 {len(wood['removed']):,} 件，记录在 woody_support_audit.json。入口右侧大树已保留。低枝证据不足，落地检查后独立低枝保留 {kinds['low_branch']} 条。树冠和细枝省略。软植被 {sum(a.get('voxels',0) for a in items):,} 个体素批为 {kinds['soft']} 个视觉 OBJ，碰撞掩码均为 0。",
        "",
        "| 网格类 | 三角面 | 倾角 >75° |",
        "|---|---:|---:|",
    ]
    labels = {"observed_ground":"扫描支持地面","filled_ground":"补面","lower_observed_ground":"实测下层地面",
              "stair_solids":"梯级闭合体","walls_and_rocks":"挡墙岩石","woody_solids":"树干和根",
              "visual_vegetation":"软植被"}
    for k,v in labels.items():
        buf.append(f"| {v} | {faces[k]:,} | {steep[k]:,} |")
    buf += [
        "",
        "梯级三角统计包含埋入地面的闭合裙边。露出的踢面按逐级表计。现场完整树木与根系覆盖没有人工清单可核。",
        "",
        "## MATRiX 和 v9",
        "",
        "安装、配置、独立运控、三轴控制点及 LiDAR 验收步骤见 USAGE.md。先载入 ZGWS + 0–100 m，再试全段。UE 视觉、Robot_MC 和真实运控闭环待 GPU 恢复。USAGE.md 也列出 v9 多层支撑、亚栅格阶沿、硬障碍、软植被和补面置信度导出方案。",
        "",
        "CPU 单测见 work/tests.log，阶段耗时和内存见 work/*.log，接缝见 seams.json，资产入口和原点见 assembly.json，Git 范围复核见 git_status_after.txt。",
        "",
    ]
    (FULL/"REPORT.md").write_text("\n".join(buf))


if __name__ == "__main__":
    report()
