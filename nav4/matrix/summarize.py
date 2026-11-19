"""Write the pilot report from the persisted CPU and geometry results."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from nav4.matrix.survey import OUT


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def cp(stats):
    return f'{stats["rms"]*100:.2f} / {stats["p95"]*100:.2f}'


def main():
    g=read("geometry")
    objects=read("objects")
    reg=read("source_comparison")
    load=read("load_3.3.0")
    bridge=read("matrix_compile")
    stairs=read("stairs")
    s=stairs["flights"]
    phy=read("physics")
    v8=read("v8_difference")
    hard=read("hard_clearance")["statistics"]
    count=Counter(x["kind"] for x in objects["solids"])
    ra=objects["rails"]
    posts=[x for x in ra if "_post_" in x["name"]]
    spans=[x for x in ra if "_post_" not in x["name"]]
    diameters=read("rail_diameter_check")
    arc_count=sum(field["slices"] for field in diameters.values())
    slope=[np.degrees(np.arccos(x["plane_normal"][-1])) for f in s for x in f["steps"]]
    plane=[x["tread_fit"]["rms"] for f in s for x in f["steps"]]
    report=[
        "# A 段高保真地形 0–30 m CPU 试点",
        "",
        "试点已交付独立 scene.xml、五组参考对照，以及含 ZGWS 本体的 MATRiX 组合 MJCF。主梯段的踏面、立面和实体扶手已重建。遮挡边坡、部分栏杆和树根存在补全，整个 ±6 m 走廊的精度与硬物覆盖仍有差异。UE 视觉、LiDAR 和 MATRiX 闭环待运行验收。",
        "",
        "## 数据与坐标",
        "",
        "官方 LAS 和 PLY 均通过发布 MANIFEST 的 SHA256。水平走廊面积 358.31 m²。laspy 每次读取 100 万条 LAS 点记录。",
        "",
        "| 来源 | 完整点数 | 走廊点数 | 点/m² | 量化步长 |",
        "|---|---:|---:|---:|---|",
    ]
    for name,label,scale in (("removed","map_remov.las","1 mm"),("map0","0.las","0.1 mm"),("ply","规划 PLY","float32")):
        source=read(f"{name}_crop")
        report.append(f'| {label} | {source["full_count"]:,} | {source["points"]:,} | {source["density_per_m2"]:,.0f} | {scale} |')
    report.extend(["","两份 LAS 头部偏移都是零，GPS 时间覆盖相近，试点分类字段均为 0。两者的空间覆盖和密度不同，官方说明没有解释动态物体过滤的区别。选 map_remov.las 是因为本走廊覆盖和重叠残差。按记录顺序每 7 个抽 1 个进入重建工作集，完整冠层回波仍保存在裁剪文件中。",
                   "","以 0.5 m XY 方格划分配准训练和留出样本。近地稳定 PLY 表面和栏杆处的 14,019 个留出点用于评估。最近点误差含采样间距，点到局部平面的误差更接近法向残差。",
                   "","| 对齐 | 10 cm 覆盖 | 欧氏 RMS / P95（cm） | 平面 RMS / P95（cm） |","|---|---:|---:|---:|"])
    for name,label in (("removed","map_remov → PLY"),("map0","0.las → PLY")):
        h=read(f"{name}_registration")["holdout"]["rigid"]
        report.append(f'| {label} | {h["coverage_10cm"]:.2%} | {cp(h["overlapping_euclidean_m"])} | {cp(h["overlapping_signed_plane_m"])} |')
    report += ["",f'两份 LAS 之间的相对平移约 {np.linalg.norm(reg["removed_to_map0_translation_m"])*1000:.2f} mm。直接稳定表面重叠双向欧氏 RMS/P95 为 {cp(reg["removed_to_map0"]["overlap_euclidean_m"])} 和 {cp(reg["map0_to_removed"]["overlap_euclidean_m"])} cm。相对旋转小于 0.003°。试点沿用规划 PLY/v8 坐标，包装版减去 v8 站点 0.40 m 的地面坐标。',
               "", "## 台阶与障碍物",
               "",
               f'34 个踏面以相邻可见踏面的高度为种子拟合局部平面，边线用两踏面之间的密集点拟合为垂直立面。各踏面拟合 RMS 的中位数为 {np.median(plane)*1000:.1f} mm，平面倾角中位数为 {np.median(slope):.1f}°。梯段最后一级的踏面深度量到该梯段模型的端界。入口和弯道末级包含平台。',
               "",
               "| 梯段 | 站位（m） | 踢面（cm） | 踏面深度（cm） | v8 最近踢面站位（m） | v8 最近踢面高度（cm） |",
               "|---|---:|---:|---:|---:|---:|"]
    for flight in s:
        for k,step in enumerate(flight["steps"],1):
            st=step.get("v8_station_m")
            vh=step.get("v8_riser_m")
            v8s=f"{st:.2f}" if st is not None else "—"
            v8h=f"{vh*100:.1f}" if vh is not None else "—"
            report.append(f'| {flight["flight"]} {k:02d} | {step["station_m"]:.3f} | {step["riser_m"]*100:.1f} | {step["tread_depth_m"]*100:.1f} | {v8s} | {v8h} |')
    report += [
        "", "v8 最近级只在 0.28 m 以内配对，有的相邻记录会共用同一个 v8 级。v8 建图报告 data/trail_a/report.json:13492–13522 的 5/12/4 级来自旧 0.1 m 地面图沿原扫描中心线的离散检测。试点入口增加 3 级，弯道分出 7 级，后续两个梯段各 12 级。",
        "",
        f'第二段最高 {max(x["riser_m"] for x in s[2]["steps"])*100:.1f} cm。v8 报告最大 27.707 cm 在 9.125 m。新场景在 8.75 m 的平台比 v8 高 {v8["route_ground"]["7.5-9.25 m"]["worst_signed_m"]*100:.1f} cm。两者的曲线形状支持 v8 将平台低估并计入上升这项解释。无独立现场尺量。',
        "",
        f'栏杆共 {len(ra)} 个胶囊，含 {len(posts)} 根立柱。用与重建交错分离的 LAS 记录，从薄截片 RANSAC 拟合 {arc_count} 个圆弧截面，按每条杆件轨迹取径向中位数。水平和斜杆直径 {min(x["geom"]["size"][0] for x in spans)*200:.1f}–{max(x["geom"]["size"][0] for x in spans)*200:.1f} cm，立柱直径 {min(x["geom"]["size"][0] for x in posts)*200:.1f}–{max(x["geom"]["size"][0] for x in posts)*200:.1f} cm。分轨径向分位数见 rail_diameter_check.json。遮挡片段按相邻截面插值，立柱按所属栏杆轨迹分组，未做现场尺量。',
        "",
        f'闭合硬物包含 {count["retaining_wall"]} 个挡墙块、{count["wall_rock_or_root"]} 个高置信表面凸体、{count["trunk"]} 个树干分段和 {count["root"]} 个树根段。根部来自附着在树干周边的低矮凸起，根背面和墙厚由可见面封闭。通用硬物提取按 v8 剔除了类型 2，软植被和细枝未输出。挡墙、根和岩壁尚无覆盖全走廊的人工复核。',
        "", "## 地面误差与规模", "",
        "| 条件 | n | 几何距离 RMS / P95（cm） |","|---|---:|---:|"
    ]
    for key,title in (("corridor","全走廊地面候选"),("within_1m","路线 ±1 m"),
                      ("well_supported_within_1m","路线 ±1 m 且距 PLY 地面支持点 ≤3 cm")):
        val=g[f"ground_holdout_{key}"]
        report.append(f'| {title} | {val["count"]:,} | {cp(val)} |')
    report += ["",
               f'误差基于构建前留出的原始 LAS 地面候选记录，计算到地面或踏面三角形的欧氏距离。地面候选用 PLY 近地表筛选。网格节距 2.5 cm，{g["v8_fill_cells"]:,}/{g["fitted_ground_cells"]:,} 个一般地面网格由 v8 补齐，路线 ±1 m 的一般地面网格无需该补齐。近路高支持区域达到约 2 cm RMS。全走廊尾部误差及遮挡区超过同一标准。',
               "",
               f'独立模型共 {load["mesh_faces"]:,} 个 mesh 面，其中一般地面 {g["triangles_terrain"]:,}，台阶 {g["triangles_stairs"]:,}，硬物 {g["triangles_solids"]:,}。地面同一组面另用于 {load["flex_faces"]:,} 个静态碰撞三角。MJCF 与 OBJ 总文件大小 {load["bytes"]/1e6:.2f} MB，MuJoCo {load["version"]} CPU 加载 {load["load_seconds"]:.2f} 秒。最终几何构建含误差测量 {g["build_seconds"]:.2f} 秒。组合 ZGWS + 场景加载 {bridge["seconds"]:.2f} 秒，有 {bridge["actuators"]} 个原机器人执行器。',
               "", "## 对照图和物理检查","",
               "figures/compare_01.jpg–compare_05.jpg 左侧是视频画面，右侧是 OSMesa 着色。匹配按地标对齐，雾化、视野和画面进度存在差异。figures/depth_01_m.npy–depth_05_m.npy 保存米制深度通道。camera_poses.json 保存世界位姿。",
               "",
               "M1 尺寸 1.03 × 0.58 m、30 kg 自由方盒，静置 2 秒，随后以约 0.095 m/s 为速度目标、140 N 为上限推 5 秒。用完整模型的局部碰撞子集做各接触试验。负接触间隙为接触求解的最大重叠。",
               "",
               "| 案例 | 推进（cm） | 阻挡步数 | 最大重叠（mm） | 阻挡几何 |",
               "|---|---:|---:|---:|---|"]
    for name,d in phy["cases"].items():
        p=d["push"]
        hits=", ".join(p["blocking_geoms"]) or "—"
        report.append(f'| {name} | {p["forward_m"]*100:.1f} | {p["blocking_steps"]} | {-p["lowest_contact_m"]*1000:.2f} | {hits} |')
    report += ["","四个案例均有承重接触与侧向阻挡。入口、转弯入口和第二段阶梯均明确碰到独立踢面。转弯平台横向试验在三角边坡上受阻。测试未覆盖整机登阶和所有扶手、岩壁或根的碰撞。",
               "", "## 与 v8 的差异","",
               "中心线路径每 5 cm 取地面射线。平台 8.75 m 的最大差异为约 +26 cm。5.10、11.85 和 17.75 m 有约 13–14 cm 的局部差异，集中在两模型的台阶边线位置不同处。20–30 m 的绝对 P95 见下表。",
               "",
               "| 区间（m） | 绝对高差 RMS / P95（cm） | 最大带符号差（cm）及站位 |",
               "|---|---:|---|"]
    for label,val in v8["route_ground"].items():
        report.append(f'| {label} | {cp(val)} | {val["worst_signed_m"]*100:+.1f} @ {val["worst_station_m"]:.2f} m |')
    report += ["",
               "每 10 cm 沿 v8 路线左右各射 2 m，在 0.30–0.80 m 身体带内采样硬物。对照量是 v8 硬格距离。两种表示分别依赖本地地面高度和射线采样，比较含体素、栏杆细截面和坡面误差。",
               "",
               "| 方向 | 双方命中 | 仅 v8 命中 | 仅试点命中 | 双方无命中 | 试点减 v8 的中位数（cm） |",
               "|---|---:|---:|---:|---:|---:|"]
    for side,item in hard.items():
        d=item["paired_delta_m"]
        report.append(f'| {side} | {d["count"]} | {item["v8_only"]} | {item["pilot_only"]} | {item["both_clear"]} | {d["median"]*100:+.1f} |')
    report += ["",
               "逐站差异、最近物体名及最大不一致见 hard_clearance.csv 和 hard_clearance.json。仅 v8 命中的站位包含局部漏拟合，需与原始点云和遮挡区逐段复核。已生成硬物在沿线占据上的吻合程度尚不足以直接替代 v8 真值。",
               "", "## 722 m 推广估计与 MATRiX", "",
               f'按试点长度比 722/30 线性外推，约 {load["mesh_faces"]*722/30/1e6:.1f} 百万 mesh 面，一般地面另有约 {load["flex_faces"]*722/30/1e6:.1f} 百万碰撞三角，OBJ/MJCF 约 {load["bytes"]*722/30/1e9:.2f} GB，约 {round(g["terrain_tiles"]*722/30):,} 个地形块。单次流式裁整段并按走廊窗口处理，纯构建约 12 分钟，含源扫描、打包和分段 CPU 加载预计约 20–40 分钟。解压、人工标注、遮挡区复核和 UE 调试另计。这些都是试点外推。',
               "",
               "全段应以 25–50 m 分区输出和验证，避免一次加载数千万静态三角。处理流程和 MATRiX 路径见 ../matrix_sim/evidence/map_import.md。试点包装保存在 matrix/，原点平移和 CPU 编译结果在 matrix_compile.json。此轮仅处理 0–30 m。"]
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n")


if __name__=="__main__":
    main()
