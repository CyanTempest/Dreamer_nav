# A 段 0–30 m MuJoCo 试点

从项目根运行。所有实测数字和几何资产放在 Git 忽略的 data/trail_a_mujoco/pilot_0_30/。CPU 启动器设置 OSMesa、CPU 亲和性、地址空间限制和独立缓存目录。

已有本任务的环境和校验过的原始数据时，可以执行以下命令重建。

    bash nav4/matrix/cpu.sh -m nav4.matrix.survey ply
    bash nav4/matrix/cpu.sh -m nav4.matrix.survey removed
    bash nav4/matrix/cpu.sh -m nav4.matrix.register removed
    bash nav4/matrix/cpu.sh -m nav4.matrix.survey map0
    bash nav4/matrix/cpu.sh -m nav4.matrix.register map0
    bash nav4/matrix/cpu.sh -m nav4.matrix.sources
    bash nav4/matrix/cpu.sh -m nav4.matrix.build --source removed
    bash nav4/matrix/cpu.sh -m nav4.matrix.rail_refine --apply
    export PYTHONPATH=../envs/mjmap/mujoco330
    bash nav4/matrix/cpu.sh -m unittest discover -s nav4/tests -p 'test_matrix_*.py'
    bash nav4/matrix/cpu.sh -m nav4.matrix.verify
    bash nav4/matrix/cpu.sh -m nav4.matrix.clearance
    bash nav4/matrix/cpu.sh -m nav4.matrix.package
    bash nav4/matrix/cpu.sh -m nav4.matrix.summarize

独立环境依赖 NumPy 1.26、SciPy、laspy、trimesh、rtree、Matplotlib、Pillow 和 MuJoCo。MuJoCo 3.3.0 路径指向 venv 内单独安装的版本，与 MATRiX 运行库对齐。verify 读取用户授权的参考视频帧，放在输出目录 figures/reference/ref_{秒数}.jpg。

settings.json 保存试点人工标注的梯段、栏杆轨迹和大树位置。逐级边线与平面采用 LAS 点拟合。更换路段时需要新的标注。批量重跑脚本会归档已有几何目录，便于审核。

## A 段 0–722 m

全段重建代码使用同一个 5 cm 米制顶点阵列，各段按站位归属面。输出在被 Git 忽略的 `data/trail_a_mujoco/full_v1/`，使用 `REPORT.md` 查看验收，用 `USAGE.md` 选分段、重叠窗或 ZGWS 组合场景。

从项目根依次执行 `full_scan ply`、`full_scan las`、`full_scene --pilot` 检查试点，再执行 `full_scene` 建全段。每个 Python 模块由 `bash nav4/matrix/cpu.sh -m nav4.matrix.<模块>` 启动，渲染和检查使用 `PYTHONPATH=../envs/mjmap/mujoco330`。扫描暂存及审计使用独立输出目录，复建时应归档已有 `full_v1/`，以免复用上次已验的阶段标记。

全段几何阶段依次用 `full_wood` 连接树干并检查根部，`full_hard_audit` 检查岩面连接和短栏杆缺口，`full_decks` 恢复扫描支持的下层地面，`full_exit_rails` 用原始 LAS 拟合出口桥栏杆。随后用 `full_package` 组合。`full_audit seams` 对照已序列化的接缝坐标，`full_checks compare` 和 `full_checks physics` 测路线与接触，`full_frames align` 和 `full_frames render` 出实景对照。

编译可用 `full_checks assemblies --entry scene_full.xml` 和 `--entry zgws_atec_full.xml` 分别启动独立受限进程。`full_visual pilot` 和 `full_visual tree` 使用旧视角及补充树木视角渲染新试点。使用 `full_report` 汇总最终验收数据。
