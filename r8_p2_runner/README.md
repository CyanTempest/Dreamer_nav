# R8-P2 原版赛道试训

2026-11-13 用户授权继续训练。本轮是单场景试训，不是完整 R8 泛化实验。

## 运行

远端工作目录 `/home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3`，tmux `r8_p2_pilot`。运行 `bash r8_p2_runner/run.sh`。

队列：运动/重置检查 → 16 回合新 BEV 基线 → 256 环境 Dreamer 试训（首个更新后 3600 秒）→ 16 回合配对复测。模型从 3 小时旧快照初始化，保留模型和优化器状态，但清空回放，学习率 1e-5，回放容量 20 万。每 10 分钟及正常退出保存可恢复 checkpoint。

25% 训练重置从 2 m 出发，75% 从 2–900 m 随机位置出发；评估一律从 2 m 出发。已有门进度不计为随机起点之前的导航成功。评估统计全程完成率。

## 范围和限制

使用 P1 atlas 与 canonical `test_862428` 对齐。这个原 test 场景本轮明确改作训练场景，因此复测只是同场景开发评估，不能声称盲测或泛化成绩。未接多地图，随机障碍关闭。静态碰撞仍用已有 LWY 几何；植被观测并不意味着新增叶片刚性碰撞。保留 P1 未达标的遮挡/高程近似，不能据此宣称实机可部署。

## 状态与文件

- 阶段：`metrics/r8_p2_status.json`
- 训练日志：`logs/r8_p2_train.log`
- 更新指标：`metrics/train_dreamer_lwy_r8_pilot_20261113.jsonl`
- 检查：`metrics/r8_p2_preflight.json`
- 前后评估：`metrics/r8_p2_baseline.json`、`metrics/r8_p2_final.json`
- 模型：`checkpoints/dreamer_lwy_r8_pilot_20261113/`

独立新增 runner，不修改 P0/P1/R7 或 upstream 源码。两端 runner 文件 SHA256 已逐项核对，记录在 source_hashes.json。历史实验目录本身不受 git 管理。
