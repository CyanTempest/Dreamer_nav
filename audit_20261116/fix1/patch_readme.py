"""Fix1: exact, asserted string replacements in r9_official_runner/README.md."""
from pathlib import Path
path=Path('/home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3/r9_official_runner/README.md')
NOTE='（仅存在于原 Mac 工作区，本服务器上没有此文件）'
edits=[
("![测试结果](metadata/r9_results.png)\n",
 "![测试结果](metadata/r9_results.png)"+NOTE+"\n"),
("区域副本和上游校验记录在 `local_data/planning_diagnostic_crop.json`。",
 "区域副本和上游校验记录在 `local_data/planning_diagnostic_crop.json`"+NOTE+"。"),
("- `metadata/r9_summary.json`：最终统计；",
 "- `metadata/r9_summary.json`"+NOTE+"：最终统计；"),
("源码同步与校验记录保存在 `metadata/source_hashes_local.json`。\n",
 "源码同步与校验记录保存在 `metadata/source_hashes_local.json`"+NOTE+"。\n"
 "\n"
 "## 审计补记（2026-11-16）\n"
 "\n"
 "- **跟踪器偏航增益**：`ScanEnv` 传入 `configs/lwy_v2_tracker.json` 的 `yaw_tracking_gain=4.0`，但该值被忽略：`ScanEnv._step` 经 `LwyEnv._step` 调用 `FastEnv._step`，后者的增益硬编码为 2。因此本文所有数字都是在增益 2 下得到的。在 9 个评估窗口上对比增益 2 与 4（每组 144 回合：随机种子 914901 与 20261116 各 72 回合），未见可测差异：规划器 141/144 对 143/144；Dreamer（采样动作）89/144 对 92/144。\n"
 "- **Dreamer 动作**：本文 Dreamer 结果均为采样动作（上游 `Agent.policy` 忽略 `mode`，始终采样）。确定性动作 `--action mode`（策略头 tanh(均值)）在增益 2 下为 100/144，采样为 89/144。该差异仅有提示意义，不能作为结论：只有 9 张地图，且回合之间相关。`diagnose_policy.py` 现支持 `--action {sample,mode}`，默认 `sample`，默认结果不变。\n"
 "- **单元测试**：`cd r9_official_runner && PYTHONPATH=. ../.venv_lwy/bin/python -m unittest discover -s tests`。\n"
 "- **原始输出**：A/B 原始结果在 `audit_20261116/results/`（由 `audit_20261116/eval_ab.py` 生成，汇总见 `audit_20261116/logs/summary.md`）。\n"),
]
text=path.read_text()
for old,new in edits:
    assert text.count(old)==1,(text.count(old),old)
    text=text.replace(old,new)
path.write_text(text);print('patched README',len(edits))
