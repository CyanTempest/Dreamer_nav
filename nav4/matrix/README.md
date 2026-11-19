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
