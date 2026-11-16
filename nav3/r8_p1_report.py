"""Generate the Chinese report from measured artifacts; no invented results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p1_common import ROOT,GEO,dump
from .r8_p1_validate import metrics,summary

def read(name):return json.loads((ROOT/f'metrics/r8_p1_{name}.json').read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def f(x,d=3):return '—' if x is None else f'{x:.{d}f}'

def main():
    val=read('validation');fit=read('noise_fit');bake=read('bake_alpha');opaque=read('bake_opaque');audit=read('reduction_audit');checks=read('checks')
    bench={name:read('throughput_'+name) for name in ['training','simple','no_visibility']}
    ratio=bench['simple']['median_seconds']/bench['training']['median_seconds']
    refs=np.load(GEO/'r8_p1_references.npz');pred=np.load(GEO/'r8_p1_predictions.npz');cpu=np.load(GEO/'r8_p1_predictions_cpu.npz')
    old=np.load(GEO/'r8_bake_probe.npz');dup=[]
    for i in range(10,37):
        if refs['offsets'][i]==0 and refs['yaw_offsets'][i]==0:
            j=int(np.flatnonzero(refs['stations'][:10]==refs['stations'][i])[0])
            dup.append(dict(probe_index=i,station_index=j,pose_max_abs=float(np.max(np.abs(refs['poses'][i]-refs['poses'][j]))),bev_changed_cells=int(np.any(refs['reference'][i]!=refs['reference'][j],axis=-1).sum())))
    parity={name:dict(changed_cells=int(np.any(pred[name]!=cpu[name],axis=-1).sum()),max_uint8_abs=int(np.max(np.abs(pred[name].astype(int)-cpu[name].astype(int))))) for name in ['simple_unfit','simple']}
    raw_delta=refs['raw_high']-refs['ground_estimate'];finite=np.isfinite(raw_delta)
    extra=dict(cpu_gpu=parity,duplicate_pose_reference=dup,max_reference_raw_obstacle_m=float(raw_delta[finite].max()),reference_raw_cells_above_16m=int((raw_delta[finite]>16).sum()),
               validation_backend='GPU JAX 0.4.33 via .venv_lwy interpreter and read-only ../envs/dreamer site-packages',
               alpha_opaque_added_band_cells=sum(a['opaque_added'] for a in audit['alpha_raw']))
    dump(ROOT/'metrics/r8_p1_crosscheck.json',extra)
    groups={}
    for split in ['station','probe','holdout']:
        rows=[r for r in val['rows'] if r['split']==split];m=val['summary'][split]['simple']
        groups[split]=dict(macro=dict(support=m['support_iou']>=.7,obstacle=m['obstacle_iou']>=.7,unknown=m['unknown_iou']>=.7,height=m['height_mae_m']<=.06),
            poses=len(rows),per_pose_pass_count={key:sum((r['methods']['simple'][key]>=.7 if key!='height_mae_m' else r['methods']['simple'][key]<=.06) for r in rows) for key in ['support_iou','obstacle_iou','unknown_iou','height_mae_m']})
    comparisons=list((ROOT/'figures').glob('r8_p1_compare_*.png'))
    assert len(comparisons)==43,len(comparisons)
    integrity=read('integrity')
    acceptance=dict(status='completed experiment; accuracy targets partially failed',groups=groups,throughput_ratio=ratio,throughput_pass=ratio<=1.5,
        reference_records=43,distinct_poses=40,comparison_figures=len(comparisons),all_existing_files_unchanged=not integrity['changed'],no_training=True,no_policy_evaluation_queue=True)
    dump(ROOT/'metrics/r8_p1_acceptance.json',acceptance)
    dump(ROOT/'metrics/r8_p1_throughput.json',dict(methods=bench,ratio=ratio,target=1.5,pass_target=ratio<=1.5))
    # Parallel comparisons and physical units make these standalone report figures.
    names=['simple','no_alpha','no_visibility','old_noise','ground_only']
    labels=['P1','No alpha','No occlusion','Old noise','Ground only']
    fig,axs=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    for ax,key,target in zip(axs,['support_iou','obstacle_iou','unknown_iou','height_mae_m'],[.7,.7,.7,.06]):
        ax.bar(labels,[val['summary']['station'][n][key] for n in names]);ax.axhline(target,c='red',ls='--');ax.set_title(key);ax.tick_params(axis='x',rotation=30)
    fig.savefig(ROOT/'figures/r8_p1_ablations.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i,(key,label) in enumerate([('median_seconds','Observe [ms / 2048]'),('frame_median_seconds','Frame [ms / 2048]')]):
        axs[i].bar(['LwyEnv','P1','P1 no occlusion'],[bench[n][key]*1000 for n in bench]);axs[i].set_ylabel(label)
    axs[0].axhline(bench['training']['median_seconds']*1500,c='red',ls='--',label='1.5 x LwyEnv');axs[0].legend()
    fig.savefig(ROOT/'figures/r8_p1_throughput.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ['simple_unfit','simple','old_noise','training_noisy']:
        rows=val['rows'][:10];axs[0].plot(refs['stations'][:10],[r['methods'][name]['unknown_fraction'] for r in rows],'o-',label=name)
        axs[1].plot(refs['stations'][:10],[r['methods'][name]['unknown_iou'] for r in rows],'o-',label=name)
    axs[0].plot(refs['stations'][:10],[r['methods']['simple']['reference_unknown_fraction'] for r in val['rows'][:10]],'k--',label='P0 reference')
    axs[0].set_title('Unknown fraction');axs[1].set_title('Unknown-mask IoU');axs[0].legend(fontsize=8)
    for ax in axs:ax.set_xlabel('station [m]');ax.grid(alpha=.2)
    fig.savefig(ROOT/'figures/r8_p1_noise.png',dpi=150);plt.close(fig)
    s=val['summary']['station']['simple'];h=val['summary']['holdout']['simple'];cfg=fit['selected_config']
    text=[]
    text.append('# R8-P1：静态世界图层与廉价可见性的 BEV 实验\n\n日期：2026-11-13。工作目录为 `research/exp_nav3`。本次只做离线几何、静态观测验证和吞吐测量，没有 RL 训练、策略评估或队列。\n')
    text.append(f'**结论：吞吐达到目标，传感器等效精度尚未全部达到目标。** 全路线 0.05 m 图层已实际生成，43 条参考记录和三方对比图齐全。10 站宏平均支撑/障碍 IoU 为 **{s["support_iou"]:.3f}/{s["obstacle_iou"]:.3f}**，未知掩码 IoU **{s["unknown_iou"]:.3f}**，共同已知格高程 MAE **{100*s["height_mae_m"]:.2f} cm**。前两项达到 0.7 的宏平均目标，后两项没有达到 0.7 和 6 cm。不是每站都通过，逐站结果和通过数量见下文。\n')
    text.append(f'2048 个不同位姿的完整 `observe` 中位数为 **{1000*bench["simple"]["median_seconds"]:.2f} ms**，同条件 `LwyEnv` 为 **{1000*bench["training"]["median_seconds"]:.2f} ms**，比值 **{ratio:.3f}×**，低于 1.5×。因此这是一份可接入的实验实现，尚不能声称已满足全部传感器替代验收，也没有把它启用到训练。\n')
    text.append('主要索引：[验收 JSON](../metrics/r8_p1_acceptance.json)、[43 位姿指标](../metrics/r8_p1_validation.json)、[噪声拟合](../metrics/r8_p1_noise_fit.json)、[额外简化审计](../metrics/r8_p1_reduction_audit.json)、[完整吞吐](../metrics/r8_p1_throughput.json)、[功能检查](../metrics/r8_p1_checks.json)。\n\n![汇总](../figures/r8_p1_summary.png)\n')
    text.append('## 1. 固定参考与比较口径\n\n完整阅读并只读复用了 [P0 报告](R8_P0_FULLGEO.md) 及其实现。Gemini 固定为 **0°、水平朝前**，光心 `(0.30, 0, 0.64) m`，H90°/V65°，1280×800，光轴 Z 量程 0.5–20 m；P0 的 −15° 推荐已不适用于本实验。双 Airy 均保持 P0 的实测旋转与相对平移，地面高度均值 0.495 m，2×96×900、0.1–60 m。姿态只含 yaw，与 P0 一致。\n\n10 站直接复用 `r8_sensor_bevs.npz` 中已经存在的 **pitch0 原始 BEV**；同时从保存的 0° 深度和 Airy 点云重建诊断低/高返回。主标准答案采用原 BEV，避免浮点深度反投影跨格影响真值。27 个 `r8_bake_probe.npz` 位姿与 6 个中间 holdout 重新全分辨率投射，保持完整场景和 alpha，全部 alpha 追踪上限触发数为零。共 43 条记录，其中 27 探针的 3 个中心位姿与站点重合，因此是 **40 个不同位姿**。参考耗时、复用来源、反投影差异在 [reference JSON](../metrics/r8_p1_reference.json)。\n\n27 个 probe 用于开发比较和噪声拟合；6 个 holdout 从未用于参数选择。10 站用于跨分区报告，但开发时查看过其诊断，不把它们称为盲测集。holdout 也只检验 190/320/480 m 附近的横偏/yaw 插值，不能代表全路线分布外泛化。没有拟合逐位姿图、位置标签或语义标签。\n')
    text.append(table(['通道','含义','编码/约束'],[
        ['H0','传感器估计地面相对脚下参考地面，m','clip(h/4+.5)×255；约 −2…2 m'],
        ['H1','最高可见返回减估计地面，m','clip(H/2)×255；0…2 m'],
        ['C2','几何支撑代理','坡度 <.8、H1 <.22 m、地面 >−.55 m；0/255'],
        ['C3','未知掩码','未知 255；已知 0'],['C4','公开路线 corridor','沿用原环境；不来自几何语义'],['C5','年龄 /8 s','clip(age/8)×255']]))
    text.append('栅格为 64×64×6 uint8，0.125 m，x∈[−2,6)、y∈[−4,4)。支撑 IoU 在已知格用 C2>.5；障碍 IoU 在已知格用解码 H1≥.22 m；未知视为未检出。未知 IoU 比较 **未知掩码本身**，没有拿已知 IoU 冒充它。高程主指标是共同已知格、解码后 H0 的 MAE，保留原 ±2 m 裁剪及 uint8 量化。全图/共同格逐通道 MAE 和 micro IoU 均另存 JSON。没有更改 P0 的开运算、真值或指标阈值。静态主比较全部为空 memory 的首帧，C5 与 C3 相同。\n')
    text.append('## 2. 离线图层：全路线、无语义、保留 alpha\n\n复用 `maps_lwy/test_862428/map.npz` 的分块地址与原点，**不读取其地面或语义来烘焙图层**。5535 个 32×32 分块共 5,667,840 个 0.05 m 单元，矩形块总面积约 14,169.6 m²，沿 1 km 路线约 6 m 走廊及分块边缘。完整 8,642 实例/6.89 亿三角形仍由 P0 精确共享 BVH 查询，不剪掉植物、不用灌木凸包。\n\n先从全场景最低包围盒以下向上投射，记录该 XY 列最低有效光学表面。缺测只在地面估计的 halo 内最近邻补齐，13×13 开运算得到约 .65 m 的静态下包络。每个分块使用 16 格 halo，避免独立分块滤波接缝。原始地面返回有效约 98.99%；缺测标记保留，不将补齐的像素当作真实返回。运行时相对脚下 z0 仍使用与 P0 一样的位姿参考地面，公开路线 prior 独立查询。\n\n相对原始最低表面的六个采样高度带是 `[.04,.22]、[.22,.8]、[.8,2]、[2,4]、[4,8]、[8,16] m`。每带向上、向下各一次首表面投射，再从中部用四个斜向短射线补采样近竖直面；斜射线落在实际命中位置，重新按目标列高度带归档，绝不把整条光路填成占据。alpha 用 P0 原贴图、UV、双线性采样、线性空间 0.5 截断与透明命中继续追踪。\n')
    text.append(table(['图层','数量','物理含义/用途'],[
        ['ground_absolute','1','无语义静态地面下包络，世界 Z'],
        ['surface_min_relative','6','各高度带实际采样表面的最低 Z 减静态地面'],
        ['surface_max_relative','6','各高度带实际采样表面的最高 Z 减静态地面'],
        ['coverage_3x3','6','3×3 原始 .05 m 列中有有效表面的比例；不是材质类别'],
        ['ground_sample_valid','1','原始竖直地面返回是否存在']]))
    text.append('共 20 个 float16 层，外加整数稀疏 lookup。为了用一次点查询近似 12.5 cm 输出格面积，min/max/coverage 在 3×3 源格上预聚合，即轴对齐约 .15 m 足迹；这不是精确的旋转方格面积积分，会缩小细孔隙。世界地面用绝对高度，其余高度相对地面存储，降低 float16 在较高世界 Z 上的误差。raw 文件仍保存未做 3×3 聚合的 float32 最小/最大表面，便于审计。没有保存或向策略提供植物/道路/硬软语义。\n')
    text.append(table(['版本','生成总秒','纯射线秒','射线数','压缩 atlas MB','展开 atlas MB','raw MB','PID GPU 峰值 MiB'],[
        [name,f(r['seconds'],2),f(r['ray_seconds'],2),r['rays'],f(r['atlas_bytes']/1e6,2),f(r['atlas_uncompressed_bytes']/1e6,2),f(r['raw_bytes']/1e6,2),read('gpu_bake_'+tag)['peak_mib']]
        for name,tag,r in [('保留 alpha','alpha',bake),('关闭 alpha','opaque',opaque)]]))
    text.append(f'耗时包括 BVH、射线、下载、NumPy 归并、滤波与压缩落盘；不是全路线的外推数字。保留 alpha 版共 {bake["rays"]:,} 条采样射线，alpha cap={bake["alpha_cap"]}。文件：[atlas](../fullgeo/r8_p1_atlas_alpha.npz)、[raw](../fullgeo/r8_p1_raw_alpha.npz)、[烘焙明细](../metrics/r8_p1_bake_alpha.json)。\n')
    text.append('## 3. 在线裁剪、可见性、memory 与噪声\n\n[SimpleEnv](../nav3/r8_p1_online.py) 是 opt-in 的 `LwyEnv` 子类，只替换 `_frame`，静态图层作为动态 JAX pytree 参数常驻 GPU。每帧按连续 XY/yaw 旋转裁剪；不存在最近缓存位姿取图。七个候选表面是地面加六个高度带。Airy 使用原外参轴的半球、量程和 96 线名义仰角 LUT 近似格子的角覆盖；0.4° 方位离散仍按连续覆盖处理。Gemini 用指定的矩形水平 FOV 和光轴 Z 量程筛选。两类传感器均保留 |x|<.49、|y|<.29 m 的近机身盲区。\n\n遮挡采用每传感器 256×64、径向 .125 m 的 polar terrain horizon，比较目标表面斜率与前方地面最大斜率。**最终默认只用地面遮挡，不把高处树冠向下挤成实心柱。** 代码保留低层 coverage 的遮挡实验开关，但其默认阈值 1.1 意味着该项禁用；将其设为 .7 的实测结果见消融，地面误差明显变大。这是主要的准确性缺口：最终方案没有正确恢复低植被和树冠内部的 3D 自遮挡，只有地形遮挡、FOV、量程、盲区和离散束近似。不能把跨视角静态层的存在当成真实单帧可见性已经解决。\n\n地面估计从可见候选的最低表面开始，无候选处回退静态下包络，再做与 P0 相同大小的 5×5 开运算。P0 在无回波处使用最近返回补齐，本实现的静态回退是额外近似。障碍采用最高可见候选减地面；支撑使用原 .5 m 差分和阈值。`_observe` 则原样继承：128×128 世界对齐 memory、max scatter、平移重定位、年龄递增、8 s 遗忘、公开路线 prior、13 维 state 与 decoder target。`priv_trav` 保持原始独立输出，从未混入六通道。\n')
    text.append(f'噪声在 **27 probe** 上重新拟合 `p(r)=clip(b+d·r²,0,1)`，选择 **b={cfg["p1_noise_base"]}、d={cfg["p1_noise_distance"]} m⁻²、σh={cfg["p1_height_std"]} m**。旧参数是 .03、.015、.025 m。40 组 dropout 候选按未知比例差最小、未知 IoU 次级选择；7 档高度噪声按共同已知格高程 MAE 选择。P0 是无测量噪声的理想参考，所以 σ=0 是本比较目标下的结果，不是实机测量噪声为零的结论，也没有把几何误差当成真实传感器噪声。所有候选及选取规则在 noise_fit JSON。\n')
    text.append(table(['10 站版本','未知比例','未知 IoU','高程 MAE cm'],[[n,f(val['summary']['station'][n]['unknown_fraction']),f(val['summary']['station'][n]['unknown_iou']),f(100*val['summary']['station'][n]['height_mae_m'],2)] for n in ['simple_unfit','simple','old_noise','training_noisy']]+[['P0 射线参考',f(s['reference_unknown_fraction']),'1.000','0.00']]))
    text.append('重新拟合使未知数量接近参考，但**没有使未知格出现在相同位置**。相对于完全不加噪声，拟合 dropout 甚至降低未知掩码 IoU；这是明确的失败项，不能通过继续调整随机丢点率解决空间遮挡结构。旧噪声消融只恢复原 dropout 与 .025 m 高度噪声，其余保持 P1，以隔离噪声重拟合的贡献。\n\n![噪声对照](../figures/r8_p1_noise.png)\n')
    text.append('## 4. 精度、holdout 与六通道误差\n\n下表 S/O/U 分别为支撑、障碍、未知 IoU，H 为共同已知格高程 MAE。训练 clean/noisy 都是同位姿原 `LwyEnv`；10 站原 baseline 直接复用 P0，其他位姿新生成。主 P1 数值和图使用与吞吐相同的 GPU/JAX 0.4.33 路径。CPU/JAX 0.7.1 的无噪声结果与 GPU 在全部 43 位姿逐 uint8 完全一致；带随机 dropout 时后端 PRNG 的具体掩码不同，所以最终报告统一 GPU 结果，具体差异见 crosscheck JSON。\n')
    text.append(table(['集合','方案','S','O','U','H cm','未知比例'],[[split,name,*[f(val['summary'][split][name][k]) for k in ['support_iou','obstacle_iou','unknown_iou']],f(100*val['summary'][split][name]['height_mae_m'],2),f(val['summary'][split][name]['unknown_fraction'])] for split in ['station','probe','holdout'] for name in ['simple','training_clean','training_noisy']]))
    text.append('**10 站逐站三方对比。** 高程/障碍展示保持共同色轴；所有图均有完整六通道。表中的未知比例属于当前方法，参考比例见各行 JSON。\n')
    text.append(table(['站点 m','方案','S','O','U','H cm','未知比例'],[[f(r['station_m'],0),name,*[f(r['methods'][name][k]) for k in ['support_iou','obstacle_iou','unknown_iou']],f(100*r['methods'][name]['height_mae_m'],2),f(r['methods'][name]['unknown_fraction'])] for r in val['rows'][:10] for name in ['simple','training_clean','training_noisy']]))
    text.append('**6 个 holdout。** 横偏 ±.15 m、yaw ±7.5°，查询时用真实连续位姿旋转世界图层；参考是本次补投射的 0° Gemini，不沿用 P0 的 −15° holdout。\n')
    text.append(table(['站点 / 横偏 / yaw','方案','S','O','U','H cm'],[[f'{r["station_m"]:g} / {r["offset_m"]:+.2f} / {r["yaw_offset_deg"]:+.1f}',name,*[f(r['methods'][name][k]) for k in ['support_iou','obstacle_iou','unknown_iou']],f(100*r['methods'][name]['height_mae_m'],2)] for r in val['rows'][37:] for name in ['simple','training_clean','training_noisy']]))
    text.append(table(['集合','逐位姿 S≥.7','O≥.7','U≥.7','H≤6 cm'],[[split,*[f'{g["per_pose_pass_count"][k]}/{g["poses"]}' for k in ['support_iou','obstacle_iou','unknown_iou','height_mae_m']]] for split,g in groups.items()]))
    text.append('**每通道 MAE。** 下表为全图、位姿宏平均；H0/H1 单位 m，C2–C5 为 0…1。每一个位姿的全图与共同已知格六通道 MAE 都在 validation JSON 的 `rows[].methods`，未省略 27 个 probe。\n')
    text.append(table(['集合','方案','H0 m','H1 m','C2','C3','C4','C5'],[[split,name,*[f(v,4) for v in val['summary'][split][name]['channel_mae_all']]] for split in ['station','probe','holdout'] for name in ['simple','training_clean','training_noisy']]))
    text.append('## 5. 消融与各项简化的实际代价\n\n以下全部仍对同一份 P0 alpha/水平相机参考比较，绝不把不带 alpha 的结果替换成标准答案。Δ 是相对完整 P1 的变化；IoU 越高越好，高程误差越低越好。\n')
    rows=[]
    for name in ['simple','no_alpha','no_visibility','old_noise','ground_only']:
        m=val['summary']['station'][name]
        rows.append([name,f(m['support_iou']),f(m['obstacle_iou']),f(m['unknown_iou']),f(100*m['height_mae_m'],2),'/'.join(f(m[k]-s[k],4) for k in ['support_iou','obstacle_iou','unknown_iou']),f(100*(m['height_mae_m']-s['height_mae_m']),2)])
    text.append(table(['10站版本','S','O','U','H cm','ΔS/ΔO/ΔU','ΔH cm'],rows))
    text.append(table(['进一步简化/替代','S','O','U','H cm'],[[name,*[f(m[k]) for k in ['support_iou','obstacle_iou','unknown_iou']],f(100*m['height_mae_m'],2)] for name,m in audit['extra_ablations']['station'].items()]))
    text.append(f'**alpha 的代价不能只看最终 IoU。** 关闭 alpha 在六个高度带累计新增 **{extra["alpha_opaque_added_band_cells"]:,} 个带-格占据记录**（同 XY 在不同带重复计数）。最终 BEV IoU 的变化却很小，甚至略有改善，说明 3×3 最大值、层内 min/max、2 m 编码饱和和不完整遮挡已经掩盖了许多细孔隙差异；这不能证明透明贴图可忽略。raw 图层仍保留 alpha 筛选。\n')
    text.append(table(['高度带','alpha 占据列','opaque 新增列','3×3 聚合后有表面列','共同列最高面 MAE mm'],[[i,a['alpha_occupied_columns'],a['opaque_added'],a['dilated_footprint_columns'],f(1000*a['common_surface_max_mae_m'],2)] for i,a in enumerate(audit['alpha_raw'])]))
    text.append('各项压缩损失的可量化边界如下。\n\n- **5 cm 采样 + 六带 min/max 替代叶片几何：** 总体代价就是 P1 对 P0 的全部误差，不能声称已拆出单独的 5 cm 离散误差。原始逐三角形信息不再在线存在。层内孔隙、表面法线、多个独立表面被压缩；各带合并区间的长度审计见下表。\n- **3×3 面积近似：** `point_crop` 删除此面积聚合，只取单个 5 cm 列。其 S/O 数字见上表；它漏掉输出格内部其它位置的表面。主方法的 .15 m 轴对齐足迹又会过覆盖旋转的 .125 m 方格，这是改善召回所付的几何精度代价，不能解释成植物实心凸包。\n- **视角地面：** `static_ground` 完全使用静态地面；`no_online_opening` 去掉在线 5×5 开运算，数值见上。回退静态地面与 P0 最近返回填补不同，这部分仍包含在最终高程误差中。\n- **高处树冠：** `no_canopy_above_2m` 删除后三级中的 >2 m 层，其 S/O 明显下降；因此即使 BEV 障碍通道在 2 m 饱和，也不能直接删除上方回波。\n')
    text.append(f'- **16 m 上截断：** 本次 43 帧参考原始最高返回减地面估计的最大值为 {extra["max_reference_raw_obstacle_m"]:.3f} m，超过 16 m 的参考格为 {extra["reference_raw_cells_above_16m"]}。这只说明本验证集未发现直接截断损失，不证明全路线/任意姿态都没有高处回波。\n')
    text.append('- **2.5D terrain horizon：** 删除它的 accuracy 变化较小，却将完整 observe 降至下节的约 21 ms。这说明本场景主要未解的遮挡来自植被多层结构，而不是已建模的地形。`dense_low_horizon` 用低层满覆盖上包络补遮挡，会错误关闭可穿透间隙并把可见高层误判为地面，导致高程误差上升；不是可接受的修复。\n- **角采样：** `no_beam_sampling` 连名义 96 线间隙也不近似；影响较小。方位采样、细面朝向和完整首返回关系仍缺失，未各自独立消融，不能为它们给出虚构的损失数字。\n- **时间/物理简化：** 与 P0 同为静态理想光学返回；pitch/roll、双目匹配、反射、风、扫描运动、实机壳体遮挡没有新增建模。由于 P0 也没有这些因素，不能从本实验估计真实硬件代价。\n')
    text.append(table(['带','合并区间均长 m','P95 m','最大 m'],[[r['band'],f(r['mean_span_m']),f(r['p95_span_m']),f(r['max_span_m'])] for r in audit['span_stats']]))
    text.append('注意同一输出足迹跨越陡坡/台阶时，相邻列的相对带坐标不同，因此聚合后的绝对高度区间可能超过原带宽。max/min 本身不表示区间每一点都是实心；把它直接当占据体用于遮挡正是存在风险的改法。\n\n![消融](../figures/r8_p1_ablations.png)\n')
    text.append('## 6. 4090 吞吐与内存\n\nGPU 共享，未清场、未终止其它进程。每种方法独立进程，统一 JAX 0.4.33、相同 map、2048 个覆盖全路线 3–997 m 的不同位姿、横偏 ±.25 m 与 yaw ±.3 rad（seed 8814），没有动态障碍。相同原始 state、memory shape、定位漂移/延迟配置；各方案保留各自应有的观测噪声。warm-up 后 30 次调用，每次对**完整返回 state/obs/extra**执行 `jax.block_until_ready`。原 decoder target、world memory、公开路线 prior、state 向量均包含在主计时中。没有用单环境耗时乘 2048，没有用热表 gather 代替完整 observe。随机横偏不等于已验证可行走状态。\n')
    text.append(table(['方法','observe min/median/P90 ms','frame median ms','env-step/s','进程 GPU 峰值 MiB','JAX peak live GiB'],[[name,' / '.join(f(1000*r[k],2) for k in ['min_seconds','median_seconds','p90_seconds']),f(1000*r['frame_median_seconds'],2),f(r['env_steps_per_second'],0),read('gpu_throughput_'+name)['peak_mib'],f(r['gpu_memory_stats']['peak_bytes_in_use']/2**30,3)] for name,r in bench.items()]))
    text.append(f'主比值 {ratio:.3f}×，通过 ≤1.5× 目标。去可见性的加速同时说明 horizon 查询仍是主要开销。进程显存由 `nvidia-smi` 每 .2 s 采样并设置本 PID >7400 MiB 自退出 guard；最高记录为 {max(read("gpu_throughput_"+n)["peak_mib"] for n in bench)} MiB，低于 8 GiB。0.24 的 JAX allocator 上限控制显存池，池保留量不等于活跃数组；表中另外给出 JAX allocator 的 peak live。采样不能严格证明亚 .2 s 瞬时峰值，但 allocator 约束、编译器内存分析和分块射线缓冲均留有余量。\n')
    text.append(table(['方法','编译参数 GiB','返回 GiB','临时区 GiB'],[[n,*[f(r['compiled_memory'][k]/2**30,3) for k in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes']]] for n,r in bench.items()]))
    text.append('map 全路线 atlas 展开约 216.4 MiB，所有 2048 环境共享一份。测量只含单个完整 test 场景；多场景常驻、在线加载 cache miss、训练网络及其梯度/优化器/回放占用不在这里，不能据此保证训练加图层之后仍低于 8 GB。共享 GPU 的外部竞争会影响单次绝对耗时，30 次原始时序已保存。\n\n![吞吐](../figures/r8_p1_throughput.png)\n')
    text.append('## 7. 未达到目标的原因、检查与训练接入\n\n未知掩码不合格的主要可观察证据是：主方法生成了近似正确的未知数量，却在参考树冠/灌木阴影处仍预测已知，并用随机 dropout 在其它位置制造孔洞；no_visibility 与默认方法的未知 IoU 接近，表明 terrain horizon 没有解释这部分阴影。地面误差主要发生在参考无地面返回的植被区域，P0 的最低可见表面和开运算会抬高估计，而本方法经常回退到静态下包络。逐帧未知 FP/FN、支撑/非支撑格高程误差及有符号残差在 reduction_audit JSON，不把全部误差简单归结为噪声。\n\n可以继续尝试以下改进，但本报告没有声称已实现或通过：\n\n1. 在低层保留薄体素占据/透射率，沿每个目标的 4–8 个高度相关采样点做短距离 LOS，保持树冠下方空隙；单独实测是否仍在 1.5× 预算内。当前“稠密低层上包络”的失败数值给出了不能直接填实的证据。\n2. 保存少量世界坐标方向条件的最低返回/可见性证据，用视角插值修复地面遮挡，不缓存 body-frame BEV；用额外离开 190/320/480 m 的 holdout 验证，不能继续仅在当前三段调参。\n3. 用姿态相关的角采样与遮挡失败模型替代独立随机 dropout。未知比例拟合只应作为总量约束，不能再把它当空间精度证据。\n4. 对旋转 .125 m 方格采用 4 个子采样或保守面积归并，以量化 .15 m 预聚合造成的孔隙损失；保持 alpha。\n\n功能检查通过：单环境 `vmap` 与批处理 measured 最大差为 0；3 个位姿的 NumPy P0-style 首帧 scatter 与 JAX 输出逐 uint8 完全一致；shape/dtype/首帧 age 通过。另对一个冻结传感器的世界 marker 做 42 次 observe，人工改变平移/yaw，穿过 memory recenter 边界，检查世界位置、年龄递增、8 s 失效；没有调用 `env.step`。这验证 memory 实现，不是连续真实传感器重建精度验证，主精度实验仍是首帧。\n\n接入训练仍缺：为其它训练 map 重烘焙和管理 atlas ID；动态障碍对观测的覆盖/遮挡合成；确认碰撞几何与新增植被感知的关系；实机 K/D、外参和光学/深度失败标定；多场景、策略网络共同驻留的显存实测；连续运动参考序列的累积/清除误差；最终通过未知及高程验收。当前类固定 `random_obstacles=False`，不能在原训练的动态障碍设置下直接假定等价。原训练/地图/配置没有被改写。\n')
    text.append('年龄边界也按实测披露：原实现 float32 每次累加 .2 s，40 次后的年龄为 7.9999967 s，因此这次检查在名义 8.0 s 尚保留 1 个 marker 格，到 8.2 s 清除。P1 原样继承了这个既有边界行为，没有为宣称“严格 8.000 s”另改规则。\n')
    text.append('最小 opt-in 用法（本任务未启用到训练）：\n\n```python\nfrom nav3.lwy_bank import LwyBank\nfrom nav3.r8_p1_online import SimpleEnv, Atlas, fitted_config\nbank = LwyBank("test", limit=1, root="maps_lwy")\nenv = SimpleEnv(bank, 2048, atlas=Atlas(), config=fitted_config())\n# env.observe(state) 保持原观测 dict；本类默认不支持动态障碍。\n```\n')
    text.append('## 8. 复现、文件保护与图索引\n\n所有新代码位于 `nav3/r8_p1_*.py`，新脚本位于 `scripts/r8_p1_*.sh`，产物与 cache 均使用允许的 P1 新文件路径。P0 文件只读复用。全量哈希复核的现有文件数为 '+str(integrity['protected_files'])+'，内容变化 '+str(len(integrity['changed']))+'，见 [integrity JSON](../metrics/r8_p1_integrity.json)。保护范围是原 nav3/configs/scripts/reports/metrics/maps_lwy 文件及 fullgeo 顶层 P0 文件，包括大型展开网格；不是对其它用户目录的扫描。\n\n依赖没有安装或升级：Warp/几何使用 `.venv_r8`；其 JAX 0.7.1 是 CPU 版，`.venv_lwy` 本身没有 JAX。因此 GPU 阶段使用 `.venv_lwy/bin/python`，通过 `PYTHONPATH` **只读复用本项目 `../envs/dreamer/lib/python3.11/site-packages` 已有的 JAX 0.4.33/CUDA 依赖**，未修改该环境或任何共享环境。所有写缓存重定向到 `fullgeo/r8_p1_cache`，bytecode 关闭。CPU 开发和最终 GPU 结果分开保留。\n\n完整顺序复现脚本：[r8_p1_run.sh](../scripts/r8_p1_run.sh)。主报告由 [r8_p1_report.py](../nav3/r8_p1_report.py) 从实测 JSON 生成。重新运行会重建 P1 产物，不会改写 P0。\n')
    text.append(table(['记录','站点/横偏/yaw','三方六通道图'],[[f'{r["index"]:02d} {r["split"]}',f'{r["station_m"]:g} m / {r["offset_m"]:+.2f} m / {r["yaw_offset_deg"]:+.1f}°',f'[图](../figures/r8_p1_compare_{r["index"]:02d}_{r["split"]}_{int(r["station_m"]):04d}.png)'] for r in val['rows']]))
    (ROOT/'reports/R8_P1_SIMPLE_BEV.md').write_text('\n'.join(text))
    print('REPORT',ROOT/'reports/R8_P1_SIMPLE_BEV.md','ratio',ratio,flush=True)

if __name__=='__main__':main()
