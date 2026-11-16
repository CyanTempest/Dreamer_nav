"""Chinese final report, built from the round-five evidence files."""
import json
from pathlib import Path


def main():
    read=lambda p:json.loads(Path(p).read_text())
    summary=read('metrics/fix_summary.json');assert summary['complete'],'Final matrix is incomplete'
    assert len(list(Path('metrics').glob('fix_diagnostic_*.json')))==101,'100 diagnostic conditions plus summary required'
    assert all(Path(f'metrics/fix_repeat100_{name}.json').exists() for name in ['dreamer_s0','ppo_s0','dreamer_s0_ft','ppo_s0_ft'])
    assert Path('videos/fix/mini_trail_comparison.mp4').exists()
    assert read('metrics/fix_integrity_final.json')['changed']==[]
    assert Path('metrics/fix_remote_final.json').exists()
    final_state=read('metrics/fix_remote_final.json')
    assert not final_state['nav3_sessions'] and not final_state['remaining_project_processes']
    census=read('metrics/fix_census.json');gate=read('metrics/fix_g1.json')
    initial=read('metrics/fix_initial_checkpoints.json');jobs=read('metrics/fix_jobs.json')
    training={}
    for method in ['dreamer','ppo']:
        name=method+'_s0';before=initial[name]['source_progress'];after=read(f'checkpoints/{name}_ft/progress.json')
        training[method]=dict(before=before,after=after,fine_tuning_seconds=after['wall_seconds']-before['wall_seconds'],
            new_steps=after['steps']-before['steps'],new_updates=after['updates']-before['updates'],
            job_wall_seconds=sum(j.get('wall_seconds',0) for j in jobs if j['label'].startswith(f'train_{name}_ft_')))
        exposure=read(f'metrics/fix_exposure_{name}_ft.json');flags=exposure['new_combination']
        training[method]['exposure']=dict(combination_maps=sum(flags),
            combination_maps_seen=sum(flag and count>0 for flag,count in zip(flags,exposure['near_log_frames'][1])),
            online_near_combination_frames=sum(count for flag,count in zip(flags,exposure['near_log_frames'][1]) if flag),
            prefill_near_combination_frames=sum(count for flag,count in zip(flags,exposure['near_log_frames'][0]) if flag))
        Path(f'metrics/fix_config_{name}_ft.json').write_text(Path(f'metrics/config_{name}_ft.json').read_text())
    Path('metrics/fix_training_summary.json').write_text(json.dumps(training,indent=2))
    lookup={(r['method'],r['split'],r['ob'],r['sf']):r for r in summary['rows']}
    mainrow=lookup[('dreamer_s0_ft','test',0,0)];base=lookup[('dreamer_s0','test',0,0)]
    regressions={k:v for k,v in summary['paired_bootstrap'].items() if k.startswith('dreamer_') and
        (v['success']['significant_regression'] or v['gates']['significant_regression'])}
    ppo=lookup[('ppo_s0_ft','test',0,0)]
    changed=mainrow['rate']>base['rate'];primary_ok=changed and not regressions
    raw_repro='\n'.join(line for line in Path('logs/fix_repro_legacy_a.log').read_text().splitlines() if line.startswith(('REPRO ','AssertionError:')))
    raw_gate=Path('logs/fix_g1_attempt1.log').read_text().strip()
    ci=mainrow['wilson'];text=[f'''# nav3 第五轮：训练分布、可复现评估与续训

本报告由 `python -m nav3.fix_document` 从保存的 JSON 自动生成。完整数据在 `metrics/fix_*.json`，主评估共 60 个条件，每条件 100 回合；诊断条件另列，不混入完整路线成绩。

## 结论与验收

Dreamer 无随机障碍、无过滤的 mini-trail 全通过率：**{base['success']}/100 → {mainrow['success']}/100**；续训模型 Wilson 95% 区间 **{ci[0]:.1%}–{ci[1]:.1%}**。PPO 同条件续训结果为 **{ppo['success']}/100**。

按“mini-trail 无过滤通过率提高，旧留出和新盲测各条件的通过率及平均门数没有显著回退”判定：**{'达到本报告的主要指标判据' if primary_ok else '未达到全部主要指标判据'}**。Dreamer 显著回退条件：{', '.join(regressions) or '未检出'}。区间包含零只是未检出差异，不是已证明严格不劣；速度、接管率及 IoU 的全部配对区间也在下方披露，不能据主要指标宣称所有指标均改善。

新 G1：规则规划器、无随机障碍、无过滤，**{round(gate['success_rate']*gate['episodes'])}/{gate['episodes']}**，平均 **{gate['mean_gates']:.3f}/8**，闸门 **{'通过' if gate['passed'] else '未通过'}**。未修改生产环境的碰撞阈值、足迹、膨胀、支撑判定或旧地图。

## 1. 24/30 与 20/30 的根因及修复

原 PPO 单独运行 mini-trail 两次均为20/30，逐回合完全一致；按原命令先执行四组留出评估，再执行 mini-trail，得到24/30，且与旧JSON逐回合完全一致。初始失败命令是 `scripts/fix_repro.py --tag legacy_a --compare metrics/eval_ppo_s0_test_ob0_sf0.json`，原始输出包含：

```text
{raw_repro}
```

相同输入的独立数值探针排除了权重、初始环境状态、观测、策略随机键差异：这些数组的SHA256完全相同。冷/热入口动作最大绝对差为2.2351741790771484e-7。编译HLO显示第三层卷积的cuDNN tuning knob 2由3变4。根因是GPU自动调优受执行/编译历史影响，细小数值差被闭环轨迹放大；不是两次随机种子未固定。原始HLO、各单变量运行以及全部对照见 `logs/fix_numeric_*.hlo`、`metrics/fix_repro_summary.json`。

| 排查项 | 证据与结论 |
| --- | --- |
| checkpoint/config 不一致 | 原两个模型SHA256与第4轮/原视频记录一致；所有PPO对照的权重哈希一致，env_config逐值一致。 |
| 环境噪声/随机障碍种子 | 初始环境state与公共观测哈希一致；环境seed=9137，主故障条件随机障碍关闭。随机生成只用显式JAX key。 |
| 策略采样种子 | 所有对照先清carry，再设置PRNGKey(9137)；同输入探针的分裂后key相同。 |
| 环境数/批量随机流错位 | 故障对照均为30环境；预热使用100环境，但随后目标评估重新初始化30环境和全部key。批量形状是协议的一部分，100回合不会冒称与旧30回合逐回合相同。 |
| GPU数值与编译历史 | 相同输入/权重/key下动作仍不同，HLO卷积调优参数不同；先运行原四组条件可以逐回合恢复旧结果。 |

正式入口 `nav3.fix_eval` 在JAX初始化前显式设置 `--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true`；每个条件运行于独立进程，固定地图顺序、批量大小、环境/策略seed，记录模型与地图哈希、运行配置，以及每个5Hz动作/位姿的SHA256。两种旧策略各两次30回合完全一致；完整100回合对照也通过单独重复检查（`fix_repeat100_*`），全部逐回合字段和轨迹哈希差异为0。新旧模型的唯一正式比较使用 `fix_eval_v1_*`。

兼容性试验保留：关闭自动调优（level0）使PPO冷/热均24/30，但同权重Dreamer首帧解码退化为常数0.48828125，IoU变为0，因此弃用；level1首帧恢复约0至1的分布，并恢复原Dreamer 1/30、29次碰撞。没有用level0试验当正式基线。第一次向elements.Config设置不存在的jax.autotune键报 `KeyError: 'jax.autotune'`，随后改到JAX初始化参数处；日志未删除。

重复性结论限于记录的4090、软件栈、编译设置和固定批量协议，不保证跨硬件逐位一致。原环境按批量生成随机数，且终止回合重置会影响共享随机流，因此不主张不同批量大小或不同策略间逐时刻噪声完全相同。所有主对比均固定100环境；按地图配对的重采样保持相同地图与回合索引。

## 2. 生成器与地图隔离

旧 `maps/` 不动。`repair_bypass(..., preserve_feasible=False)` 默认保留旧行为；新模式先使用原rasterize（含原栅格膨胀），以1.03×0.58 m包络、11×7足迹采样、5 cm中心格和四连通验证倒木前后通路，能通就保留右肩石块，只有不通才逐块修复。该连通检查是充分路径筛查，不是完整SE(2)可达性证明；仍由真实规划器G1验收。

第一次G1为92/100，8次碰撞来自blind/50006和50010的右侧倒木及对侧原有石块，几何连通不等于带噪跟踪有足够余量。训练按闸门停止，没有复制或更新模型。随后对所有新采样组合统一增加10 cm的障碍距离验收余量，不满足就继续随机采样；不按失败地图编号选择性修补。这个余量仅用于生成筛查，不改变生产栅格、实际足迹或终止判定。首次地图完整保存在 `maps_v2_attempt1/`，指标在 `fix_g1_attempt1.json`、`fix_g1_eval_attempt1.json` 和 `fix_census_attempt1.json`；首次生成代码也保留在日志目录。

首次G1原始输出：

```text
{raw_gate}
```

新图以0.8概率尝试组合；宽度、邻近门和连通性限制可能使尝试回退到基础场景，实际频次见下表。倒木长度1.2–1.85 m、半径0.10–0.17 m、相对路线角65–115°，左右侧随机；石块1–3块，半尺寸、yaw和前后位置独立采样，包含对侧/后方组合。组合分布在不同门附近，未读取mini-trail或诊断地图几何。

训练种子 **40000–40199（200张）**，新盲测 **50000–50019（20张）**，完整清单在 `maps_v2/manifest.json` 与 `metrics/fix_map_manifest.json`。它们与旧训练10000–10199、旧留出20000–20019、mini-trail 871002和诊断30000–30003不相交。训练manifest为200旧图+200新图的均匀采样，旧/新各50%。

特征统计复用第4轮 `diag_geometry.feature_row`；“右侧邻近石块”沿用 |Δs|<3.5 且 −1.6<横向<0，另增加镜像后的任意对侧计数；三件组合要求任意对侧石块且门距<3 m。

| 集合 | 张数 | 右侧邻近石块 | 任意对侧石块 | 倒木+对侧石块+门 | 左/右倒木 |
| --- | --- | --- | --- | --- | --- |''']
    for key,label in [('old_train','旧训练'),('new_train','新训练'),('blind','新盲测')]:
        d=census[key];c=d['counts'];text.append(f'| {label} | {d["n"]} | {c["right_rock"]} | {c["either_opposing_rock"]} | {c["triple"]} | {c["left_log"]}/{c["right_log"]} |')
    text.append('''
![组合频次](figures/fix_combinations.png)

## 3. 奖励与续训

原贴近项为 `-.04*max(0,.5-min(-center_edge, center_obstacle_distance))`，用机体中心距离；机体前端/侧边已接近障碍时，中心距离可能仍超过0.5 m而惩罚为零。新奖励保留全部原项，并额外减去 `.25*clip((.20-min_foot_obstacle_distance)/.20,0,1)`。距离来自原 `FastEnv.truth`，覆盖同一静态占据栅格及回合随机障碍，使用同一25个足迹点；20 cm以外新增代价为零。两算法完全相同，开关默认权重0。

原生物理step先独立执行，随后另一个JIT只改reward及累计return，避免把奖励算子混入物理编译图。近倒木回归测试实际触发新惩罚，同时检查其余物理状态逐值不变。没有改变碰撞、支撑、膨胀、成功或终止定义；正式评估仍用原奖励以便比较。

两个原checkpoint全目录复制到对应`_ft`目录并核对latest.pkl哈希。模型、优化器和计数继续沿用；首次重置采样回合，Dreamer重建混合训练图replay并用规划器预填，避免旧奖励标签与新目标混用。旧replay和原checkpoint不动。Dreamer保持4608环境、size12m、16×64、ratio32、百万Replay；PPO保持640环境及原GRU/rollout/BPTT/优化器参数。训练始终无安全过滤。

| 方法 | 续训记账小时 | 作业实际墙钟小时 | 新在线环境步 | 新更新步 |
| --- | --- | --- | --- | --- |''')
    for method,d in training.items():text.append(f'| {method} | {d["fine_tuning_seconds"]/3600:.4f} | {d["job_wall_seconds"]/3600:.4f} | {d["new_steps"]:,} | {d["new_updates"]:,} |')
    text.append('\n实际训练曝光单独记账，倒木窗口为各训练图自身倒木前6 m至后7 m。下表仅统计主动组合采样成功的地图，不含仅由修复器保留下来的既有组合，因此分母与特征统计的三件组合数不同。生成了组合不代表有限墙钟内已到达该位置：\n\n| 方法 | 实际到达主动增广地图/主动增广地图总数 | 在线窗口帧 | 预填窗口帧 |\n| --- | --- | --- | --- |')
    for method,d in training.items():
        e=d['exposure'];text.append(f'| {method} | {e["combination_maps_seen"]}/{e["combination_maps"]} | {e["online_near_combination_frames"]:,} | {e["prefill_near_combination_frames"]:,} |')
    text.append('''
各方法按累计3小时目标运行，约30分钟一段，15分钟存档，单个训练进程远低于3小时。记账沿用原训练progress口径，初始化、预填和存档计入训练进程墙钟；独立评估另计。精确命令、退出码、进程和时间见 `metrics/fix_jobs.json`。每段后在20张新盲测图各跑1回合，无随机障碍、无过滤，画下图。仅使用预先约定的最终checkpoint，不依据监控曲线挑选最优时点；盲测没有进入梯度/replay，但被定期查看，不能称为从未查看的最终测试集。

![学习曲线](figures/fix_learning_curves.png)

## 4. 完整路线评估

每条件旧留出20×5、新盲测20×5、mini-trail100回合，随机障碍×原安全过滤各开/关。过滤使用原 `control.safety_filter`，不是第4轮高接管的margin候选。8门通过沿用原协议 `gates==8`，失败原因另记；若同一步出现8门与非成功原因，不隐去该边缘情况。Wilson区间描述固定地图条件下的回合变异，mini-trail区间不能解释为地图泛化区间。PPO/规划器没有可通行解码头，IoU为不适用；分母为0也为不适用。
''')
    text.append(Path('metrics/fix_evaluation_table.md').read_text())
    text.append('## 5. 地图配对差异、失败分类与诊断\n\n20张地图作为重采样单位，前后使用同一地图索引，10000次bootstrap，seed9042。以下为续训减原模型；速度/接管率/IoU采用分子分母聚合后的配对比率差。IoU随策略轨迹分布变化，不能仅据它判断模型权重本身改善或退化。\n')
    for file in ['fix_bootstrap_table.md','fix_secondary_bootstrap_table.md','fix_failures_table.md']:
        text.append(Path('metrics',file).read_text())
    text.append('![mini-trail失败路线位置](figures/fix_failure_s.png)\n\n诊断全用原/第4轮只读地图：倒木短段s=158→171，冷启动组s=150→171分别使用seed9137和9141、各30回合，独立seeds30000–30003的base/rocks从各自倒木前6 m到后7 m。段通过不是8门完整成绩，冷启动记忆也不同于全路线历史。\n')
    text.append(Path('metrics/fix_diagnostic_table.md').read_text())
    text.append('''
## 6. 视频、完整性与限制

对比视频：`videos/fix/mini_trail_comparison.mp4`；两个续训模型使用同seed9137、100环境批次中的第0回合，无随机障碍、无过滤。录像是快环境真实策略推理轨迹的MuJoCo运动学显示，不是A2带腿闭环验收。单方法视频和原始rollout也在该目录。

完整保护哈希清单为 `metrics/fix_protected_hashes.json`，初始共2222文件；最终逐文件检查见 `logs/fix_protect_final.log`。`tests/check_diag.py`继续通过，`tests/check_fix.py`覆盖旧指纹、原指标哈希、新旧种子隔离、实际训练日志种子、两算法共同奖励及重复评估。最终资源状态见 `metrics/fix_remote_final.json`。

软件、GPU驱动、相关依赖版本及代码SHA256见 `metrics/fix_runtime.json`。

限制：只有一个训练种子；2.5D栅格仍是保守碰撞语义，不等于真实接触概率；没有更改A2、UniLab或上游Dreamer main，也未完成原项目尚缺的官方训练循环接入、独立安全进程、慢环境闭环或TensorRT导出。本轮没有把任何诊断/mini地图加入训练。原生产指纹仍为 `a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d`；新增地图、奖励和评估运行选项另有显式记录。

## 7. 代码清单与复现

| 文件 | 说明 |
| --- | --- |
| nav3/evaluate.py | 可选地图根、策略重置、每步哈希、有限值检查和短段评估；默认保留旧入口。 |
| nav3/fix_eval.py / nav3/dreamer_adapter.py | 新评估协议固定JAX启动配置，记录输入/模型/地图与运行证据。 |
| nav3/maps.py / nav3/maps_v2.py | 可选保留可绕行石块，生成独立随机组合及M1包络连通性筛查。 |
| nav3/diag_geometry.py | 第4轮特征脚本支持新地图根，原默认不变。 |
| nav3/fix_prepare.py | 全量特征统计、G1、混合训练manifest和checkpoint复制核验。 |
| nav3/fix_training.py | 默认关闭的共享足迹余量奖励与G1前置检查。 |
| nav3/train_dreamer.py / nav3/train_ppo.py | 新地图根、首次重置/新replay和续训分段接口，默认保持旧训练行为。 |
| nav3/fix_queue.py | 3小时/方法预算、约30分钟可恢复作业、独立盲测学习曲线。 |
| nav3/fix_repro_report.py / scripts/fix_repro.py / scripts/fix_numeric_probe.py | 冷/热入口与数值核对，保留失败断言、输入哈希和HLO。 |
| nav3/fix_report.py / nav3/fix_document.py | Wilson、地图bootstrap、全量表、诊断表、图和中文报告。 |
| nav3/inference_video.py | 支持续训模型、新协议和独立输出目录，旧默认不变。 |
| tests/check_diag.py / tests/check_fix.py | 保护旧结果输出；新增隔离、哈希、重复性和奖励物理不变回归。 |
| scripts/fix_protect.py | 初始/最终流式校验全部受保护文件。 |
| scripts/fix_status.py / scripts/fix_runtime.py / scripts/fix_final_state.py | 只读进度轮询、运行依赖/源码指纹和最终会话/进程清理核验。 |
| scripts/fix_verify_eval.sh / fix_after_baselines.sh / fix_maps_stage.sh / fix_training_stage.sh / fix_final_evaluation.py | 有闸门的顺序调度、失败日志和可恢复评估矩阵。 |

远端执行均从 `exp_nav3/` 开始并 `source scripts/env.sh`。新评估命令示例：

```bash
"$NAV3_PYTHON" -m nav3.fix_eval --name dreamer_s0_ft --split test --episodes 100 --tag independent_repeat
"$NAV3_PYTHON" tests/check_fix.py --reward
JAX_PLATFORMS=cpu "$NAV3_PYTHON" tests/check_diag.py --output metrics/fix_diag_checks_final.json
JAX_PLATFORMS=cpu "$NAV3_PYTHON" scripts/fix_protect.py --check
```

评估拒绝覆盖同名JSON，重跑需使用新tag。训练调度器可从`_ft`的progress及最新checkpoint继续，已完成3小时则不再训练。所有失败/中断保留在日志，不计作成功验收。
''')
    Path('FIX_REPORT.md').write_text('\n\n'.join(text))
    print(json.dumps(dict(primary_criterion_passed=primary_ok,dreamer_mini_success=mainrow['success'],regressions=list(regressions))),flush=True)


if __name__=='__main__':main()
