"""Academic report of FULL-target validation, limitations and blind freeze."""
import statistics as st
from experiments.tma_reduction_layout.phase11.c33 import common as c


def short(layout): return layout.split(':',1)[1]


def build():
    t=c.read(c.OUT/'theta_H100.json');v=c.read(c.OUT/'cross_validation.json')
    d=c.read(c.OUT/'development.json');b=c.read(c.OUT/'baselines.json')
    diag=c.read(c.OUT/'mechanism_diagnostics.json');blind=c.read(c.OUT/'D1_predictions.json')
    resource=c.read(c.OUT/'resource_calibration.json');rp=resource['full_single_pass_model']
    q=c.read(c.prior.OUT/'matrix.json');cv=v['full']['metrics']
    lines=['# Stage C3.3 — 单次 FULL 主目标模型与十个 D1 预测冻结','',
        '## 1. 数据、停点与目标','',
        '旧 C3/C3.1/C3.2 与 C2c 证据不动。本版使用 C2d 九个几何的 61/61 个无循环 FULL：原 55 个候选补齐 18 个 within，另加入授权移出盲测的 `16×64/w8`（R=4）六个候选。每个几何所有合法候选都有实测真值；没有“该候选未实测”的选择。',
        '', '主目标是 FULL 的稳态 tile 服务与候选相对默认的配对 FULL 差分，不再将 K64 MC 强制等同 K1 FULL 的片上项。MC 与固定 R 净对照是可消融结构协变量；固定 R 的 packing 干预仍未成功独立辨识裸 packing 成本，未恢复成已知指令时延。host descriptor 路径固定。实验中用 grid 做 ns/tile/SM 归一化，静态模型不接收运行时 grid，不使用 26.6 µs 或全图截距。',
        '', '主要标签为 128/256/384 MiB 的 time-versus-grid 斜率。128 MiB 的原始 FULL 时间亦在 C2d 报告中；有限 grid 的 launch/尾波截距不等于一个可直接套用的 tile 底座。因测量批次和 footprint 拟合协议更新，本版 ns 与旧 C3.2 的绝对 ns 不作无配对的误差改善百分比。',
        '', 'D1 保留十个历史 strict-zero-mention（五核心、五边界）。当前仅执行静态 G_L、模型推断与预测冻结；未对任何剩余 D1 几何做 GPU 编译、PTX/NCU 检查或计时。',
        '', '## 2. 从静态结构到 FULL 响应','',
        '预先绑定的候选族为直接 FULL 结构回归，以及 `H + max(0, A_L - eta*H)` 的部分重叠响应（eta∈{0,.25,.5,.75,1}）；H 为独立操作型 TMA 对照响应，A_L 的非负系数主要由 FULL 与配对差分标定。H 包括 wait、首行读取和输出，不称作纯 TMA 硬件 latency。q 指 CUDA occupancy capacity（在寄存器/shared reservation 限制下的驻留上限），不声称动态全过程始终恰有 q 个 CTA 活跃。',
        '', f'最终开发模型选择 `{t["selected_recipe"]["family"]}`，eta={t["selected_recipe"]["eta"]}，ridge={t["selected_recipe"]["ridge"]}。所有超参数只由开发集内部 whole-geometry CV 选择；完整 eta/ridge profile 在 theta_H100.json。eta 是响应模型参数，未独立识别物理重叠比例或唯一 B。',
        '', 'G_L 输入为 ML 理想/冲突波前、加载 warp 指令、BF16 local max/partial widening、MR shuffle/共享通信/barrier、输出重分布需求与预测 q。输出重分布部分是静态结构代理，不把未经验证的代理计数说成精确 SASS。固定 R 表使用 native−fixture 与 wide−fixture_wide 同语义净对照；不使用旧 `(p−1)/p × Pp` 变体积惩罚。',
        '', '系数是经非负约束、ridge 与 FULL/pair 联合目标拟合的统计响应参数。并非将几个手工分数直接相加，也不宣称一个系数就是独立硬件指令延迟。所有 feature_names、coefficients、训练几何、排除几何和 inner scores 可直接审计。',
        '', '## 3. 九折嵌套留一几何验证','',
        '外层一次留出整个几何，同时排除其 FULL、T_control、MC 和 FULL-register 资源记录；内层在剩余几何中选择 recipe。独立 fixed-R/primitive 微基准作为共享硬件标定保留。训练使用已知 q capacity 标定响应；内层评分与外层主结果使用静态寄存器模型预测 q。另报已知真实 q 的 conditional 指标以分离资源误差。',
        '',f'外层 natural FULL：RMSE **{cv["FULL_RMSE_ns"]:.3f} ns/tile/SM**，相对 RMSE **{cv["FULL_relative_RMSE_percent"]:.2f}%**，绝对相对误差中位 **{cv["FULL_median_absolute_percent"]:.2f}%**，最大 **{cv["FULL_max_absolute_percent"]:.2f}%**。配对差分 RMSE **{cv["paired_RMSE_ns"]:.3f} ns**；归一化配对比例 RMSE **{cv["paired_relative_RMSE_percent"]:.2f} 个百分点**。同一几何多个候选和 worker 不是新的独立几何样本。',
        '', '| 留出几何 | FULL 相对 RMSE | FULL 最大绝对误差 | paired RMSE ns | 已知 q 的 paired RMSE ns | recipe eta/ridge |',
        '|---|---:|---:|---:|---:|---|']
    for f in t['outer_folds']:
        m=f['metrics'];cm=f['conditional_metrics'];rr=f['selected_recipe']
        lines.append(f'| {f["held"]} | {m["FULL_relative_RMSE_percent"]:.2f}% | {m["FULL_max_absolute_percent"]:.2f}% | {m["paired_RMSE_ns"]:.3f} | {cm["paired_RMSE_ns"]:.3f} | {rr["family"]} {rr["eta"]}/{rr["ridge"]} |')
    lines+=['','## 4. 名义预测、实测收益与保守选择','',
        '下表每个几何由不含该几何的外层模型预测；实测收益来自该轮完整 FULL 候选的同条件真值。oracle 只用于事后评分，不输入静态选择器。','',
        '| 几何 | nominal 布局 | 预测 FULL 收益 | 同布局实测收益 | oracle 收益 | 保守布局 | 保守实测收益 |',
        '|---|---|---:|---:|---:|---|---:|']
    for r in d['outer_evaluation']:
        geo=r['geometry'];name=f'{geo[0]}×{geo[1]}/w{geo[2]}'
        lines.append(f'| {name} | {short(r["nominal"])} | {r["nominal_predicted_gain_percent"]:.2f}% | {r["nominal_observed_gain_percent"]:.2f}% | {r["oracle_gain_percent"]:.2f}% | {short(r["selected"])} | {r["selected_observed_gain_percent"]:.2f}% |')
    lines+=['',f'全部开发数据最终拟合模型的保守改选：{d["development_switches"]}/9（样本内说明，不作泛化成绩）。训练集内交叉标定区间的外层保守改选：**{d["outer_switches"]}/9**；实测恶化超过 3% 的外层选择 **{d["outer_adverse_selections_over_3_percent"]} 个**。每个候选都可按实测真值评分。全部开发数据拟合的逐几何表见 development.json 的 development_evaluation，外层表为 outer_evaluation。',
        '', '## 5. 单侧配对区间与资源场景','',
        '残差使用 `(FULL_L−FULL_default)/FULL_default` 的预测误差，按同 worker 的 measured default 归一；直接在收益比例上构造上尾，避免大 tile 的绝对 ns 误差无差别罚小 tile。保留残差符号，不作零点反射。候选/default 的公共 H 不作为两个独立上下界相减；其是否抵消由同几何 FULL 配对预测与 CV 残差反映。',
        '', '分层键依次为包络支路（both_platform/crossing/both_onchip；直接响应单列）、布局族、low/high vec、q-change，另寻找 R 相差不超过两倍的尺度邻域。每层要求至少三个不同几何，不足则明确回退到更粗层。上界为该层 cross-fit 正向最差残差 + 至少 1% 或正尾 RMS 的裕度，再按有上限的 leverage 与 R 支持情况放大。负残差不会制造虚假的正向极值。',
        '', '外层每一决策的区间库仅来自其余八个训练几何的 cross-fit，条件于该外层 inner-selected recipe；不是把当前留出几何自身残差加入自己的阈值，也不是分布无关的置信区间。最终 D1 使用九折 outer residual bank。分别记录每个候选实际采用的 conditioning_keys、distinct_geometries、R 支持与回退。',
        '', f'外层逐候选（worker 中位）上尾覆盖 **{d["outer_covered"]}/{d["outer_total"]}**。该覆盖来自 9 个相关几何组，不能将候选数当作独立样本量。真实决策另传播 shared/register 合法性与相关 q 场景；每个场景共享 default 的 register residual，再叠加同尺度候选−default 的 allocation residual。最后仅当所有受支持场景的 FULL 改善仍 >3% 才改选。',
        '', f'FULL 物理资源标定使用 {rp["observations"]} 个去重可用 CUBIN，R 范围 {rp["R_support"]}、W 支持 {rp["W_support"]}；单次 FULL 与 MC/局部 fixture 分开。n_regs 由 cuobjdump/driver 实测回归，区间来自 held-geometry residual 加 register margin，不再使用 `2R+2columns+48` 未标定硬门。资源模型区间是经验边界，不是无 spill 的形式证明。',
        '', f'九个开发几何的 held-geometry 静态 q 点预测与 driver capacity 完全一致 {diag["held_geometry_resource_q_exact"]}/{len(diag["held_geometry_resource_checks"])} 个候选；逐项 n_regs/q 在 mechanism_diagnostics.json。寄存器预测使用同一种 allocation rounding，CV 与最终选择器不先各自取整后跨越不同的寄存器分配门限。',
        '', '## 6. 基线与消融','',
        '固定 vec 使用 N-first；该 vec 在几何上不合法时保留默认。fanout-only 只比较 u*w，平局优先更大 vec、再 N-first。zero-base 使用当前 MC/control 标定，但仍 `max(H,MC)`，作为截断结构诊断。final 为九个几何全量拟合的样本内结果；outer 为未见几何预测。','',
        '| 策略 | 改选/9 | 平均实测收益 | >3% 恶化 | 平均 oracle regret |','|---|---:|---:|---:|---:|']
    for r in b['baselines']:
        lines.append(f'| {r["policy"]} | {r["switches"]} | {r["mean_gain_percent"]:.2f}% | {r["adverse_over_3_percent"]} | {r["mean_oracle_regret_percent"]:.2f}% |')
    lines+=['','固定 vec1 与 fanout-only 在这九个几何上已经很强：平均收益约 21.36%，完整模型 nominal 约 21.93%。时间预测误差改善不等价于选择收益显著优于简单基线；当前开发集尚不足以证明复杂选择器的必要性。保守模型平均收益较低，反映了为配对误差与资源不确定性付出的弃权代价。']
    z=b['zero_base_outer_metrics']
    lines+=['',f'同九折、同完整矩阵的 zero-base 诊断：FULL 相对 RMSE {z["FULL_relative_RMSE_percent"]:.2f}%，paired RMSE {z["paired_RMSE_ns"]:.3f} ns，paired 比例 RMSE {z["paired_relative_RMSE_percent"]:.2f} 个百分点。它沿用当前 control/MC 标定与静态 q，未用 FULL 修正包络。']
    lines+=['','消融均独立重新标定并执行同样的嵌套 whole-geometry CV；误差不是仅删除预测时一个已拟合项。消融改善也如实保留，不将其隐藏为模型有效。','',
        '| 消融 | FULL 相对 RMSE | paired RMSE ns | paired 比例 RMSE | held nominal 平均收益 | >3% 恶化 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in b['ablations']:
        lines.append(f'| {r["policy"]} | {r["FULL_relative_RMSE_percent"]:.2f}% | {r["paired_RMSE_ns"]:.3f} | {r["paired_relative_RMSE_percent"]:.2f} pp | {r["mean_gain_percent"]:.2f}% | {r["adverse_over_3_percent"]} |')
    full=next(r for r in b['ablations'] if r['policy']=='full')
    fixed=next(r for r in b['ablations'] if r['policy']=='Fixed-R-blind')
    lines+=['',f'Fixed-R-blind 的 FULL 相对 RMSE 为 {fixed["FULL_relative_RMSE_percent"]:.3f}%（完整模型 {full["FULL_relative_RMSE_percent"]:.3f}%），paired RMSE 为 {fixed["paired_RMSE_ns"]:.3f} ns（完整模型 {full["paired_RMSE_ns"]:.3f} ns）。因此固定 R 协变量的总体增量证据仍混合，不能声称已独立证明 packing 项必要。加载消融的差异也较小；通信与资源/包络消融的退化更明显，具体以表为准。']
    lines+=['','## 7. 两个原先假 0% 点与 within FULL 诊断','']
    for r in diag['false_zero_cases']:
        g=r['geometry'];lines.append(f'- {r["scope"]}，{g[0]}×{g[1]}/w{g[2]}：nominal FULL 预测收益 {r["nominal_predicted_gain_percent"]:.2f}%，同布局实测 {r["nominal_observed_gain_percent"]:.2f}%，oracle {r["oracle_gain_percent"]:.2f}%。不使用 MC 打破 FULL 0% 平局来冒充正收益。')
    lines+=['','`32×64/w8 vec8:within` 与默认 vec8:N-first 的 FULL 整核计数如下。它不是 ML 64-conflict 的那个布局；不能把其旧正残差归因于并不存在的 64 个输入加载 bank conflicts。','',
        '| 布局 | FULL ns | regs/q | shared WF/CTA | BC/CTA | SHFL warp inst | BAR warp inst |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in diag['FULL_counter_audit']:
        if r['geometry']==[32,64,8] and r['vec']==8:
            inst=r['dynamic_warp_instructions_per_CTA']
            lines.append(f'| {short(r["layout"])} | {r["service_ns"]:.2f} | {r["n_regs"]}/{r["q"]} | {r["shared_load_wavefronts_per_CTA"]:.3f} | {r["shared_load_bank_conflicts_per_CTA"]:.3f} | {inst["shuffle"]} | {inst["barrier"]} |')
    lines+=['','within 将跨 warp MR 共享通信降为零，但仍有高 fanout shuffle/max 和输出转换；q capacity 相同，寄存器无 spill。整核计数与纯 ML/MR 分量的差不能直接换算成一个固定 MO latency。另一方面，32×128 的 within vec2/4/8 仍能在整核 BC 看到静态 ML 64-conflict 之上的小量附加流量；所有原值保留在 mechanism_diagnostics.json。现有计数可排除一些错误归因，不能声称独立分解了每一纳秒。',
        '', '低 R 的 T_control 已有 R2 与新 R4（同 W8）的完整 q 曲线；q1→2→4→8 的对照服务明显变化，证明自然平台不应只按 bytes 外推。控制仍是操作型响应，不能仅凭这些点认定最小 CTA 周期严格等于 15.6 ns×W。',
        '', '## 8. 十个 D1 静态预先预测','',
        '| 几何 | 历史/分层 | nominal | nominal 预测收益 | 冻结选择 | 选择的最差预测收益 | 改选 |','|---|---|---|---:|---|---:|---|']
    for r in blind['cases']:
        p=r['prediction'];g=r['geometry'];lines.append(f'| {g[0]}×{g[1]}/w{g[2]} | {r["history_category"]}/{r["stratum"]} | {short(p["nominal_choice"])} | {p["nominal_gain_percent"]:.2f}% | {short(p["selected"])} | {p["selected_worst_gain_percent"]:.2f}% | {p["changed"]} |')
    lines+=['','所有选择及每个候选的 q/寄存器区间、配对上尾、资源场景和回退原因已冻结于 D1_predictions.json。核心与边界分别统计；边界超出 FULL R/W 支持时不把保留默认算作性能验证成功。历史三级分类沿用 C3.2 原始审计 cutoff；当前预测文本出现不算实际运行暴露。',
        '', '## 9. 交付与可复核边界','',
        '最终 validator 核对旧证据 SHA、C2d commit 绑定、61/61 真值、九个 outer holdout 的排除、inner recipe 选择、FULL/pair 误差算术、全部政策评分、单侧区间库不含 held geometry，以及十个 D1 预测逐项重算。代码、theta、history、预测与报告一同 SHA 冻结。无新增 GPU 分配、无 D1 编译/计时、无 C++ 修改、无 PR。',
        '', '这是 H100 开发域上的静态响应模型验证；九个几何的 outer 结果与十个尚未执行的 D1 预测分开。消融是否改善、支持稀疏处的区间回退、资源估计误差及不利选择均见表与 JSON。没有将物理 B/eta 未识别或全部回退默认解释为有效性证明。',
        '', '当前仍有明确缺口：开发集 R4 与 R2 的实测收益未通过保守选择；D1 仅 1/10 触发改选（核心 1/5，边界 0/5），两个 R4 核心虽 nominal 约 25%–27%，在 q 场景及上尾误差后仍不足以稳健改选。上尾 49/52 覆盖的三个失败候选逐项见 development.json；这不是保证。C3.3 修复了完整真值缺失与 FULL 假 0% 表征，但尚未证明所有小 tile 都能安全改选，也未证明跨硬件有效。',
        '', '### 图：外层预测与选择收益',
        '', '![61 个 FULL 的留一几何预测](figures/FULL_LOGO_predictions.png)',
        '', '![九个几何的名义与保守选择收益](figures/FULL_LOGO_gains.png)',
        '', '相同科学图的 PDF 位于 figures/，便于导出。']
    c.derived(c.OUT/'REPORT.md',('\n'.join(lines)+'\n').encode())
    print('C3.3 report written')


if __name__=='__main__': build()
