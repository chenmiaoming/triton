"""Render the complete preregistered held-out report without selecting models."""
from experiments.tma_reduction_layout.phase11.d1 import common as c

ORDER=('C34_resource_only_conservative','C34_full_auto_conservative','C33_conservative',
       'upstream_default','fixed_vec1','fixed_vec2','fixed_vec4','fanout_only',
       'C34_resource_only_nominal','C34_full_auto_nominal','C33_nominal')


def name(g):return f'{g[0]}×{g[1]}/w{g[2]}'
def short(s):return ':'.join(s.split(':')[1:])
def ratio(s):return f'{s["covered"]}/{s["total"]}'


def run():
    p=c.protocol();truth=c.read(c.OUT/'truth.json');scores=c.read(c.OUT/'scorecards.json')
    coverage=c.read(c.OUT/'coverage.json');gate=c.read(c.OUT/'artifact_gate.json')
    counter=c.read(c.OUT/'counter_gate.json');a=c.read(c.OUT/'analysis.json')
    top={r['policy']:r for r in scores['tables']['all']}
    main=top['C34_resource_only_conservative'];fixed=top['fixed_vec1']
    lines=['# Stage D1 — H100 未见几何盲测','',
        f'预选主模型在 10 个未见几何上改选 {main["switches"]} 个，平均实测收益 **{main["mean_gain_percent"]:.2f}%**，oracle regret **{main["mean_oracle_regret_percent"]:.2f}%**；其中两个 R4 案例获约 20% 收益，第三次改选仅 1.53%，未达到冻结收益下界。>3% 恶化为 {main["adverse_over_3_percent"]}，改选但收益≤3% 为 {main["switched_gain_le_3_percent"]}。fixed_vec1 平均收益 {fixed["mean_gain_percent"]:.2f}%，当前盲测没有证明复杂选择器在收益上优于该简单基线。','',
        '本阶段只评估启动前绑定的预测，未重拟合、未交换主次模型、未改动 C3.4 或更早证据。',
        '', '## 1. 预绑定协议与测量口径','',
        f'- 主模型：**C34_resource_only**；自动内层选族对照：**C34_full_auto**；历史对照：C33。主模型由开发集九折外层证据预选，因此旧外层结果属于开发选模依据，D1 才是本次独立检验。',
        f'- 协议：`preregistration/protocol.json`，SHA256 `{c.sha(c.encode(p))}`，冻结时间 `{p["frozen_at_UTC"]}`；冻结时尚未编译或计时 D1。全部原预测以字节副本绑定，角色与 11 组决策在首个远程编译之前确定。',
        f'- 编译器 `{p["pinned_upstream"]}`，沿用 C2d 不可变基础镜像和原生库 SHA `{p["baseline"]["native_SHA256"]}`；无需重建 Triton，CPU preflight 的 ccache 前后统计一致。',
        '- 原回归的 host-created BF16 rank-3 descriptor、128B swizzle、Gluon 显式 register layout；直接复用 C2c 的 loop-free one_pass FULL/T_control，未改 C++。',
        '- 主指标：自然驻留下，128/256/384 MiB 连续输入的时间对 grid/SM 的 OLS 斜率，单位 **ns/tile/SM**。grid 只用于测量归一化，未输入静态选择器。128 MiB 原始时间另列。',
        '- 三个新 H100! invocation；每个 cell 3 visits × 4 event samples，visit 前 2 次 warmup，随机交错顺序；图中只有一个 kernel 和两个 event。先通过三种输入正确性检查，再执行正式计时；计时过程不导入 Triton/JIT，NCU 独立运行。',
        f'- 测得 {truth["FULL_candidates"]}/59 FULL 候选与 {truth["controls"]}/10 T_control；{gate["checks"]} 项 eligibility 检查通过，记录 {a["event_samples"]} 个原始 event samples。spill/local memory 即使出现也不作为删除候选的理由。',
        '- 各案例收益/遗憾先在同一 worker 内配对，再取三个 worker 的中位数；分层平均对几何等权。附带的 t 区间为三 worker 的描述性小样本区间（df=2），重复 launch 不当作独立实验。oracle 是同 worker 已测候选的最小斜率，受测量噪声影响。',
        '- 3% 是预注册实用差异阈值；所有记分表固定展示核心5、边界5、全部10，未按结果剔除边界或弃权案例。可选的低 q 重拟合诊断本轮未执行。',
        '- strict_zero_mention 是先前历史审计截点上的分类；本次预注册与 D1 采集自然会记录这些形状，不把当前文件的字面出现当作此前 GPU 暴露。',
        '', '## 2. 十个几何的 oracle 与主模型逐项结果','',
        '| 几何 / 分层 | 默认 | oracle 点布局 | oracle 收益 | 主模型冻结选择 | 主模型实测收益 | 自动对照实测收益 |',
        '|---|---|---|---:|---|---:|---:|']
    allscores={r['policy']:r for r in scores['tables']['all']}
    primary={tuple(r['geometry']):r for r in allscores['C34_resource_only_conservative']['items']}
    secondary={tuple(r['geometry']):r for r in allscores['C34_full_auto_conservative']['items']}
    for row in truth['cases']:
        key=tuple(row['geometry']);pr=primary[key];se=secondary[key]
        lines.append(f'| {name(key)} / {"核心" if row["stratum"]=="empirically_unseen_core" else "边界"} | {short(row["default"])} | {short(row["oracle_median_point_layout"])} | {row["oracle_gain_percent"]["median"]:.2f}% | {short(pr["layout"])} | {pr["gain_percent"]:.2f}% | {se["gain_percent"]:.2f}% |')
    lines+=['','“oracle 点布局”按候选三 worker 斜率中位数最小值展示；oracle 收益/遗憾按每 worker 的最小实测斜率计算。两者不保证对应同一布局，完整 worker 值见 truth.json。','',
        '### 主模型实际改选：启动前预期与盲测','',
        '| 几何 | 冻结选择 | 同布局点预测收益 | 冻结最差收益下界 | 实测收益 |',
        '|---|---|---:|---:|---:|']
    frozen_primary=c.read(c.OUT/'preregistration/bindings/C34_resource_only/predictions.json')
    for r in frozen_primary['cases']:
        pred=r['prediction']
        if not pred['changed']:continue
        selected=next(v for v in pred['candidates'] if v['layout']==pred['selected'])
        observed=primary[tuple(r['geometry'])]
        lines.append(f'| {name(r["geometry"])} | {short(pred["selected"])} | {selected["nominal_gain_percent"]:.2f}% | {selected["worst_gain_percent"]:.2f}% | {observed["gain_percent"]:.2f}% |')
    lines+=['',
        '## 3. 预锁定模型与简单基线并列记分','']
    for stratum,label in (('core','核心域：5 个 empirically_unseen_core'),('boundary','边界域：5 个 legacy_boundary'),('all','全部 10 个')):
        lines += [f'### {label}','','| 冻结策略 | 改选 | 平均实测收益 | 平均 oracle regret | >3% 恶化 | 改选但收益≤3% | 失败 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        table={r['policy']:r for r in scores['tables'][stratum]}
        for key in ORDER:
            r=table[key]
            lines.append(f'| {key} | {r["switches"]}/{r["cases"]} | {r["mean_gain_percent"]:.2f}% | {r["mean_oracle_regret_percent"]:.2f}% | {r["adverse_over_3_percent"]} | {r["switched_gain_le_3_percent"]} | {r["failures"]} |')
        lines.append('')
    lines+=['fixed_vec 的非法候选按冻结规则回退默认；fanout_only 只最小化 u×w，再偏好大 vec 和 N-first。所有决定已在预注册中逐形状写死。记分差异仅来自这些原始决定，不用 D1 挑选赢家。','',
        '## 4. 冻结时间预测与资源区间的真实覆盖','',
        '时间点预测未使用实测 q 修正。FULL 误差按每个候选的三 worker 中位数计算；配对纳秒/比例误差按同 worker 相对默认的差分计算。“上尾”检验 nominal-q 残差边界，“传播下界”检验冻结资源场景与性能上界联合给出的收益下界。','',
        '| 模型 / 分层 | FULL 相对 RMSE | FULL RMSE ns | 配对 RMSE ns | 配对比例 RMSE pp | 性能上尾 | 传播收益下界 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('C34_resource_only','C34_full_auto','C33'):
        for s in ('core','boundary','all'):
            r=coverage['models'][key]['summary'][s]
            lines.append(f'| {key} / {s} | {r["FULL_relative_RMSE_percent"]:.2f}% | {r["FULL_absolute_RMSE_ns"]:.2f} | {r["paired_RMSE_ns"]:.2f} | {r["paired_ratio_RMSE_pp"]:.2f} | {ratio(r["performance_upper"])} | {ratio(r["propagated_gain_lower"])} |')
    lines+=['','| 模型 / 分层 | 连续 n_regs 上界 | 向上取整上界（辅助） | 边际 q 区间 | 离散 q 集合 | 联合 q 对 | 超出资源校准 R / 新 W 的候选数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('C34_resource_only','C34_full_auto','C33'):
        for s in ('core','boundary','all'):
            r=coverage['models'][key]['summary'][s]
            lines.append(f'| {key} / {s} | {ratio(r["n_regs_upper"])} | {ratio(r["n_regs_integer_ceil_upper"])} | {ratio(r["q_interval"])} | {ratio(r["q_discrete"])} | {ratio(r["joint_q"])} | {r["R_outside_count"]} / {r["new_W_count"]} |')
    lines+=['','全部 59 个 FULL 候选、49 个非默认配对；资源覆盖分母按候选，不把重复 worker 当作独立形状。C33 无离散 q 集合时以 0/0 标记，不伪造该覆盖指标。各失败项、资源预测、实测值及原选择门原因均保存在 coverage.json。','',
        '### 未覆盖项清单','', '| 模型 | 几何 / 布局 | 未覆盖项目 |', '|---|---|---|']
    failures=0
    for key in ('C34_resource_only','C34_full_auto','C33'):
        for r in coverage['models'][key]['records']:
            missed=[label for field,label in (('n_regs_upper_covered','n_regs'),('q_interval_covered','q interval'),
                ('q_discrete_covered','discrete q'),
                ('joint_q_covered','joint q'),('upper_tail_covered','performance upper'),
                ('propagated_lower_gain_covered','gain lower')) if r[field] is False]
            if missed:
                failures+=1;lines.append(f'| {key} | {name(r["geometry"])} / {short(r["layout"])} | {", ".join(missed)} |')
    if not failures:lines.append('| 全部 | — | 无 |')
    lines+=['','## 5. 完整候选真值矩阵与独立传输对照','',
        '以下服务斜率、收益采用冻结的 worker 配对中位数；括号区间为收益的三 worker 描述性 95% t 区间。原始各 worker 样本、格点拟合、截距、三点残差见 analysis.json / truth.json。local_bytes 是 CUDA driver 的每线程 local allocation；它与 SASS local load/store 站点同时保存，不能仅凭 local allocation 就声称发生了 spill 流量。','',
        '| 几何 | 候选 | FULL ns/tile/SM | 相对默认收益（描述性区间） | 物理 n_regs | 自然 q | driver local / CUBIN stack bytes/thread |',
        '|---|---|---:|---|---:|---:|---:|']
    for case in truth['cases']:
        for r in case['candidates']:
            res=r['worker_resources'][0];lo,hi=r['gain_percent']['descriptive_t95']
            stack=r['static_audit']['stack_bytes_from_cuobjdump'][0]
            lines.append(f'| {name(case["geometry"])} | {short(r["layout"])} | {r["slope_ns"]["median"]:.3f} | {r["gain_percent"]["median"]:.2f}% [{lo:.2f}, {hi:.2f}] | {res["n_regs"]} | {res["q"]} | {res["local_bytes"]} / {stack} |')
    lines+=['','| 几何 | 独立 T_control 服务斜率 ns/tile/SM | 128 MiB 原始时间 μs（辅助） | n_regs / q |',
        '|---|---:|---:|---|']
    for case in truth['cases']:
        r=case['control'];rs=r['worker_resources'][0]
        lines.append(f'| {name(case["geometry"])} | {r["slope_ns"]["median"]:.3f} | {r["raw128_us"]["median"]:.3f} | {rs["n_regs"]} / {rs["q"]} |')
    lines+=['','### 128 MiB 原始时间辅助表（不混入主记分）','',
        '| 几何 | 默认 μs | 主模型 μs | 主模型原始时间收益 | 主模型斜率收益 |',
        '|---|---:|---:|---:|---:|']
    for case in truth['cases']:
        rows={r['layout']:r for r in case['candidates']};pr=primary[tuple(case['geometry'])];r=rows[pr['layout']]
        lines.append(f'| {name(case["geometry"])} | {rows[case["default"]]["raw128_us"]["median"]:.3f} | {r["raw128_us"]["median"]:.3f} | {r["aux_raw128_gain_percent"]["median"]:.2f}% | {pr["gain_percent"]:.2f}% |')
    lines+=['','## 6. NCU 与证据链','',
        f'- 独立采集 {len(counter["reports"])} 份 NCU 二进制报告，{len(counter["dynamic"])} 个 grid=1 K1 动态指令记录、{len(counter["hardware"])} 个自然/受控 q 硬件计数记录；相同 CUBIN 用 SHA 去重并保留 profile_aliases。',
        '- 收集 shared load/store wavefronts 与 bank conflicts、动态 SASS 指令、DRAM 字节/利用率及活动 warp。计数不代替图 event 计时；FULL 的 shared 流量含输入、规约通信和输出转换，未强行归因于单独 ML。',
        '- 每份 .ncu-rep 均在本地 CPU 重新导入，核对远端导入的全部数值与 per-PC 动态 SASS。原始 report、collect/import 输出、命令行、GPU UUID、CUBIN hash 和正确性绑定均保存。',
        '- protocol/bindings 在编译前 write-once；每次 dispatch 记录同一 protocol SHA 和 call ID；cpu_preflight、eligibility、timing、profiles 原始返回及所有 compiler artifacts 保留。preregistration/prior_manifest.json 锁定此前所有阶段文件；最终 validator 重新核对，未覆盖任何旧证据。',
        '']
    summaries={r['tag']:r for r in c.read(c.OUT/'counter_summary.json')['records']}
    lines+=['| 几何 | shared-load 波前/CTA 默认→主选择 | load bank conflicts/CTA 默认→主选择 | 动态 SHFL/CTA 默认→主选择 | 动态 BAR.SYNC/CTA 默认→主选择 | DRAM 利用率 默认→主选择 |',
        '|---|---:|---:|---:|---:|---:|']
    for case in truth['cases']:
        pr=primary[tuple(case['geometry'])]
        d=summaries[case['default']+':ONE_FULL'];v=summaries[pr['layout']+':ONE_FULL']
        dh=next(r for r in d['hardware'] if r['q_condition']=='natural')
        vh=next(r for r in v['hardware'] if r['q_condition']=='natural')
        def pair(field):return f'{dh["per_CTA"][field]:.1f}→{vh["per_CTA"][field]:.1f}'
        def dyn(field):return f'{d["grid1_dynamic_warp_instructions"][field]}→{v["grid1_dynamic_warp_instructions"][field]}'
        lines.append(f'| {name(case["geometry"])} | {pair("shared_load_wavefronts")} | {pair("shared_load_bank_conflicts")} | {dyn("shuffle")} | {dyn("barrier")} | {dh["DRAM_throughput_percent"]:.2f}%→{vh["DRAM_throughput_percent"]:.2f}% |')
    for r in summaries.values():
        if r['grid1_local_load_store_warp_instructions']:
            lines.append(f'\nlocal memory 证据：{r["tag"]} 的 grid=1 动态 LDL/STL 共 {r["grid1_local_load_store_warp_instructions"]} 条 warp 指令；对应 STACK/LOCAL、driver local allocation 和 SASS 站点在 truth.json 中逐项保留。该候选保留在全部服务斜率、oracle 和模型误差分母中。')
    for r in coverage['models']['C34_resource_only']['records']:
        pr=primary[tuple(r['geometry'])]
        if r['layout']==pr['layout'] and r['nondefault'] and not r['propagated_lower_gain_covered']:
            lines.append(f'\n主模型收益下界未覆盖：{name(r["geometry"])} / {short(r["layout"])}，冻结下界 {r["propagated_lower_gain_percent"]:.2f}%，实测 {r["observed_gain_percent"]:.2f}%。这一失效保留为盲测反例，未调整区间或改选规则。')
    lines.append('\n16×256/w8 的独立传输对照约 389.23 ns，默认 FULL 约 396.65 ns，所选布局约 390.84 ns，实测均接近传输平台。所选 within 布局减少 SHFL/BAR.SYNC，同时增加 shared-load bank conflicts；这些整核计数用于核对现象，尚不足以单独识别时间模型高估 19.19% 收益的全部原因。其 128 MiB 原始时间收益为约 4.00%，主服务斜率收益为 1.53%；两种口径按预注册分别保留，不改用较有利的原始时间评分。')
    lines+=['','## 7. 解释范围与停点','']
    for s in ('core','boundary','all'):
        table={r['policy']:r for r in scores['tables'][s]};r=table['C34_resource_only_conservative'];b=table['fixed_vec1']
        lines.append(f'- {s}：预选主模型收益 {r["mean_gain_percent"]:.2f}%，regret {r["mean_oracle_regret_percent"]:.2f}%；fixed_vec1 收益 {b["mean_gain_percent"]:.2f}%，regret {b["mean_oracle_regret_percent"]:.2f}%。这只是本次冻结 H100 集合的并列结果，不据此改变主模型或宣称跨硬件普适。')
    lines+=['- 覆盖失败、模型弃权、边界域外推和简单基线优势均完整保留。10 个几何不足以给出普遍的安全改选保证；未开展跨硬件确认、编译器修改或下一阶段实验。',
        '- 本阶段完成最终离线 validator、commit 与 push 后停止，等待复核；不创建 PR。','']
    c.derived(c.OUT/'REPORT.md','\n'.join(lines).encode())
    print('D1 report rendered; all 11 policies in each of three strata')


if __name__=='__main__':run()
