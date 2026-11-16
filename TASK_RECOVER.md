# 任务：修复 nav3 Dreamer 训练的主机内存溢出，恢复「Dreamer s0 累计 5 小时 → PPO s0 累计 5 小时 → 评估」队列

用中文汇报。你看不到之前的对话。先读本目录的 `TASK.md` 开头两节「用户更新」「用户重申的最终设计」、`README.md`、`ACCEPTANCE_20261110.md`，它们是当前有效的范围。本文件只管一件事：**让被打断的训练队列尽快、可靠地恢复，并跑完**。用户在睡觉，期望早上看到两个方法各 5 小时的结果。

## 现状（已由验收方核实）

- 代码以 Mac `/Users/amiao/Desktop/ATEC_NAV/nav3/` 为准，运行目录是远端 4090：`ssh hpf@100.108.114.64`，`~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`（`scripts/sync.sh` 推代码，`scripts/fetch.sh` 拉指标）。Mac 与远端的 `nav3/train_dreamer.py` 目前一致。
- G1（规划器留出集 100/100）和 G2 已通过。队列由 `python -m nav3.runner` 在远端 tmux 里顺序执行：Dreamer s0 累计 18000 s → PPO s0 累计 18000 s → 评估 → 报告。
- **远端现在没有任何训练在跑**（没有 tmux 会话，GPU 空闲）。最后一个 `formal_dreamer_s0` 作业在 2026-11-11 01:00 CST 被内核 OOM killer 杀掉（`journalctl -k`：`Out of memory: Killed process 609102 (python) ... anon-rss:58873628kB`，主机共 62 GB 内存、1 GB 交换），runner 遇到失败后整个队列退出，PPO 没有开始。
- Dreamer s0 最后一个完好的检查点：`checkpoints/dreamer_s0/latest.pkl`，`progress.json` 为 1,299,872 步、40,621 次更新、累计 `wall_seconds` 3295（约 0.92 小时）。被杀前最后一段（1.30M→1.35M 步）因为换页，3000 多秒只走了约 4 万步。留出集评估：75 万步 6%、100 万步 8%、125 万步 42% 全通过（平均 7.14/8 门，失败全是碰撞）。
- 配置：`configs/runtime.json` 里 `dreamer_envs=4608`（用户要求显存占用接近 90%，实测 89.3%）；batch 16×64、size12m、train_ratio 32、replay 容量 100 万、`chunksize` 用官方默认 1024。

## 我对原因的分析（请你验证，不要直接当结论）

1. 每步存储约 29 KB（`bev` 64×64×6 + `priv_trav` 64×64×1 + 向量），100 万步容量本身约 29 GB。
2. 官方 `embodied.Replay` 按 worker 分流写 chunk，chunk 只有在写满 `chunksize` 且其中所有 item 被淘汰后才释放。训练受 train_ratio 限制，总吞吐约 600 环境步/秒（`metrics/memory_tuning_summary.json`：1024 环境 596 步/秒，3072 环境 599 步/秒，**环境数不改变吞吐**），4608 个 worker 时每个 worker 约 0.13 步/秒，写满一个 1024 步的 chunk 要两个多小时。于是常驻内存 ≈ 容量 + worker 数 ×（接近一个 chunk），远超 62 GB，必然 OOM。
3. 每 900 秒 `replay.save()` 可能把同一个未写满的 chunk 以不同长度反复落盘（`replay/dreamer_s0` 已有 25,344 个文件、13 GB），续训 `replay.load()` 时可能重复加载。请确认并处理（不要丢掉有效数据的前提下清理重复/过期文件，或改成不会重复的保存方式）。
4. `latest.pkl` 有 1.79 GB（改环境数前是 152 MB），里面包含 4608 个环境的完整状态，保存时还有一次性内存峰值。

## 要做的事（按顺序，第 1–3 步的目标是 30–40 分钟内让训练重新跑起来）

1. **确认原因**：读 `dreamerv3/embodied/core/replay.py`、`chunk.py`（本目录 `dreamerv3/` 是 `nav3` 分支工作树）和 `nav3/train_dreamer.py`，核实上面第 2、3、4 点；给出常驻内存随（容量、环境数、chunksize、已写步数）变化的估算式。
2. **修复主机内存**，要求：
   - 不改算法语义：仍是官方 Agent、官方 Replay 的均匀采样、batch 16×64、size12m、train_ratio 32、replay 容量 ≥100 万步、序列长度不变。可以调 `chunksize`、保存方式、环境状态的存档方式等工程参数。对 `dreamerv3/` 的改动只在 `nav3` 分支工作树里做并提交，不动 main，不 push。
   - **尽量保留用户要求的 4608 个环境 / 显存约 90%**。只有当实测+估算表明 4608 个环境下训练进程常驻内存无法稳定低于 **45 GB** 时，才把环境数降到满足该上限的最大值，并在汇报里写明原因和降后的显存占比；不允许用占位数据凑显存。
   - 用一次 ≥10 分钟的实跑（从现有检查点续训即可，时间计入累计预算）记录 RSS 曲线，确认与估算式吻合、并外推到 5 小时不超过上限。
   - 在 `train_dreamer.py` 里加 RSS 看门狗：RSS 超过 50 GB 时先正常存档再以专用退出码退出，日志写明原因。
   - 续训必须从 1,299,872 步的检查点继续，保留权重、优化器、replay（能保留多少保留多少，说明实际加载了多少步）和累计时间；不允许从头开始。
3. **让队列更耐操并重启**：
   - `nav3/runner.py`：单个作业失败后从最近检查点自动重试（同一配置最多 3 次，之后跳到下一个配置而不是整个队列退出）；失败作业的墙钟不应挤占后面作业的预算——现有 `--budget-seconds 43200` 已经不够跑完剩余的 Dreamer 约 4.1 小时 + PPO 约 5 小时 + 评估，请把预算设到足够并留余量。
   - PPO：环境数对 PPO 是真实吞吐，请用一次短探针选一个吞吐高、显存不超过 90%、主机内存安全的环境数（默认 128 太小），写进 `configs/runtime.json`。如果因此需要丢弃现有 11 万步的冒烟检查点重新开始，可以（它只训练了 42 秒），在汇报里说明；累计 5 小时从新的起点算。
   - 在远端 tmux（会话名 `nav3_queue`）里重启 runner，确认 Dreamer 在跑、RSS 和显存正常、步数在涨。
4. **守到跑完**：之后每 20–30 分钟用 ssh 看一次（RSS、显存、步数、有没有新的失败），直到 Dreamer 5 小时、PPO 5 小时、评估和 `nav3.report` 全部结束，再 `scripts/fetch.sh` 把指标、图、日志、`REPORT.md` 拉回 Mac。中途再出故障就按同样原则修复并续训，不要停在那里等人。**不要因为等待而提前结束会话。**
5. 评估按 `TASK.md` 4.4：留出集 ≥100 回合/条件、mini_trail ≥30 回合/条件，条件为随机障碍开/关 × 安全过滤开/关；规划器、Dreamer s0、PPO s0 三行都要有。

## 约束

- 远端只写 `~/wang-sm/ATEC_NAV_WSM/research/exp_nav3/`；不要 sudo；不要动别的目录和别人的进程；启动前看 `nvidia-smi`。
- 不要碰 `/Users/amiao/UniLab`、`a2_trail`、`go2w_trail`。Mac 上另有用户自己的训练进程（`m1_unilab.train`）在占 CPU，与本任务无关，不要动。
- 不改 G1/G2 判定、不改环境和奖励、不改评估集；`environment_fingerprint` 必须保持 `a3350106…896d`，否则与已有 Dreamer 检查点和评估不可比。
- 不启动额外种子、消融、随机预填对照（用户已暂缓）。
- 在 `ACCEPTANCE_20261110.md` 末尾追加一节记录这次事故：原因、证据、修复、损失的训练时间、配置是否有变。

## 汇报

最后一条输出写：OOM 的确认原因和证据；改了什么（文件、参数）；最终环境数与显存/主机内存实测；Dreamer、PPO 各自的累计步数/更新数/墙钟；三种方法在各评估条件下的表（来自 `metrics/evaluation_summary.json`）；没完成的项；远端是否还有任务在跑。
