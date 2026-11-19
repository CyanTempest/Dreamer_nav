# Colab A100 训练

在服务器项目根运行。将 `--map` 指向冻结地图目录，将 `--name` 设为新运行的名字。地图需要包含 `manifest.json` 和其中 `forward` 记录所指的 NPZ。

```bash
python3 nav4/colab/launch.py --map data/trail_a_v9_frozen --name v9-s0 \
  --seconds 2400 --envs 256 --model size12m --batch 16 --seed 0 --eval-minutes 10
```

启动器检查当前账户的会话，在 Colab 创建单个高内存 A100。VM 从 GitHub 获取 `6405fc4`，再应用服务器当前的训练代码、策略包装和配置。它在临时磁盘上解包训练地图，以发行版 Python 的 venv 引导 uv，安装 Python 3.11.18 和锁定的 JAX 0.4.33 CUDA 12 依赖，检查 JAX GPU 后端。所有训练和 GPU 采样都在 VM 运行。

`train.sh` 先在独立目录跑 10 秒优化器冒烟，再从头启动正式训练。两次运行共用 JAX 编译缓存。训练秒数从第一个优化器更新开始计，包含周期评估。编译、预热、最终落盘和回传还需要墙钟时间。训练使用按需分配的显存，XLA 上限为设备容量的 90%。`--batch` 指序列条数，默认每条 64 步。仓库训练默认值仍是 256 环境、size12m 和 16×64。

- 加 `--weights /绝对路径/best.pkl` 或 `latest.pkl`，可从模型档位及观测定义匹配的 nav4 agent 继续优化。该方式恢复 agent 参数、优化器与内部计数，重新建立 replay、环境状态、训练时钟和本轮更新计数。
- VM 内需从完整状态继续时，对已有运行目录使用 `/content/nav4-env/bin/python -m nav4.train_dreamer --logdir /content/runs/名字 --seconds 累计目标秒数 --resume`。完整恢复依赖原 VM 的官方检查点和 replay。
- 服务端每 10 分钟及训练结束时把日志、指标、配置和已发布的权重同步至 `../nav4_colab/runs/名字/`。回放留在 VM 临时磁盘上。单次手动同步为 `bash nav4/colab/sync_to_server.sh 名字 --once`。
- 启动器根据实际费率计算 13 单元的收尾时间，预留同步时间，并在退出时关闭会话、查询会话列表和余额。若复用已分配的同名高内存 A100，在命令中加 `--allocated-at` 和原始分配请求的含时区 ISO 时间。预算从该时间起算。


### 独占 A100 的吞吐配置

服务器启动器和 VM 内的 `train.sh` 均接受 `--fast-a100`。该开关启用 90% 显存预分配、XLA 默认自动调优、无压缩内存回放和批量写入。每 60 分钟以及正常结束时各评估一次，每次 8 个整段。可用 `--eval-minutes` 调整间隔，评估种子、协助模式和确定性动作沿用训练配置。

VM 入口另支持 `--save-minutes`。10 分钟验收使用 60 分钟保存间隔，最后一次评估和完整检查点在训练计时结束后完成。`metrics.jsonl` 中的 `allocator.peak_bytes_in_use` 表示 XLA 活跃分配峰值，`gpu.csv` 中的显存读数包含预留池。

在 `nav4.train_dreamer` 上使用 `--set profile_steps=true` 可记录逐批量步的同步耗时。每条 `profile` 的 `seconds` 为前台各阶段的秒数，`background_sample` 为同期预取线程的采样耗时，两者有重叠。`benchmark_only=true` 供一次性标定使用，会跳过最后一次检查点。
