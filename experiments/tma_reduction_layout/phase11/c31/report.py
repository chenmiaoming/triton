"""Generate C3.1 report and figures from frozen development calculations."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tma-c31-mpl')
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c31 import common as c


def run():
    theta=c.read(c.V1/'theta_H100.json');dev=c.read(c.V1/'development.json')
    blind=c.read(c.V1/'D1_predictions.json');model=theta['models']['full']['model']
    selections=[r for r in dev['selections'] if r['mode']=='full']
    lines=['# Stage C3.1 — MC 直接校准、配对差分与 D1 预测更新','',
        '**开发集目标已达到：32×64/w8 改选 vec1:N-first，32×128/w4 保留默认 vec8:N-first。** '
        '本轮只有 CPU 分析；GPU 作业、D1 编译/计时、native rebuild、C++ 改动均为 0。'
        '没有把开发集结果称为盲测成功，也没有创建 PR。','',
        '旧 C3 原样保留为 [c3_v0_overconservative](../c3_v0_overconservative/REFERENCE.json) 的历史引用，'
        '对应原目录 `stage_c3`、commit `'+c.PARENT+'`。没有覆盖旧 REPORT、参数、失败记录或八个旧预测。','',
        '## 1. 本版改变了什么','',
        '训练目标直接取 C2 的 MC D1 稳态 ns/tile/SM；不是 ML/MR/MO 探针和，也没有 fixture/MC 全局乘性比值。'
        'K16/64/256、三档 G/SM 的斜率归一沿用 C2。15 布局×4 个 q 条件形成 60 个 cell median；'
        '保留全部 180 个 worker service estimate，用于同条件配对。natural 属于 C2a，受限 q 属于 C2b，'
        '不将两个 cohort 的同编号 worker 当成配对。','',
        '静态 G_L 及其原始证据全部复用：ML actual WF 已包含地址冲突附加波前，MR 的 SHFL/WF/BAR、p、e 与输出结构可直接计算。'
        '输入仍为 host-created descriptor、BF16→FP32 max、沿 M 规约。选择器没有 runtime grid、最终 binary、实际盲测寄存器数或计时入口。','',
        '模型为 `S_MC(L,q) = Σ β_j x_j(G_L,q)`，约束 `β_j ≥ 0`。'
        '原始需求特征是波前、warp 指令、线程内 max 等价操作与树深度；P/U/W 仅提供可学习权重的形状/q 回归项。'
        '它们没有以权重 1 相加成预测时间。驻留压力项随 q 非增；D=1 原语探针的微小非单调性先做 PAVA 投影，'
        '原始表值和投影值均保留，未改原始数据。','',
        f'所选 recipe：`{theta["recipe"]}`。共比较 16 个预登记组合，按 LOGO 配对差分 RMSE 选择，'
        '不以得到指定选择为损失函数。全部比较保留在 [model_search.json](model_search.json)。'
        '由于只有两个几何，选择 recipe 与校准误差范围复用了开发 LOGO，属于非嵌套开发分析，不是独立置信保证。','',
        f'设计矩阵 rank={model["diagnostic"]["rank"]}，columns={model["diagnostic"]["columns"]}。'
        '系数可预测组合服务，但相关特征的独立物理代价未必可辨识；不能将下表的系数解释为裸指令延迟。','',
        '| 非零共享系数 | 数值 | 单位/含义 |','|---|---:|---|']
    for name,beta in zip(model['feature_names'],model['coefficients']):
        if beta<=1e-9:continue
        unit='ns / 对应静态需求单位'
        if name.endswith('_shape_q'):unit='无量纲；乘有效原语形状 ns 特征'
        if '__inverse_q' in name:unit='ns / (对应需求 / q)'
        lines.append(f'| {name} | {beta:.6f} | {unit} |')
    lines += ['',
        'P 项为 `Pp(q) × W*columns/4 × (p−1)/p`；U 项为 `Uu(q) × W*columns/(4u)`（u>1）；'
        'W 项为 `Ww(q) × W*columns/(8w)`（w>1）。这些是 fixture 形状回归量，β 从 MC 学习；'
        '不是独立 max/shuffle/barrier 的实测纯耗时。每个候选保存 feature 值及 `β×feature` 的 ns 贡献，可逐项复算。','',
        '## 2. 配对误差与资源区间','',
        'LOLO 一次留掉某布局全部 q/worker；LOGO 一次留掉整个几何。'
        '同一个 fold 模型同时预测候选和默认，观测按 `(cohort, GPU_UUID, invocation, q_condition, geometry)` 配对。'
        '计算 `r_D = S_D_obs−S_D_pred` 和 `r_Δ = (S_C_obs−S_D_obs)−(S_C_pred−S_D_pred)`。'
        '每个残差对连同符号、来源和 fold 保留于参数文件。','',
        '比较时采用同一个 `(r_D,r_Δ)`：`S_D*=S_D+r_D`、`S_C*=S_C+r_D+r_Δ`，并使用同一个传输平台。'
        '保留 nominal `(0,0)`，并将联合残差关于零反射，避免缺项模型仅因误差单边而得到乐观偏差修正。'
        '没有将候选 marginal 上界与默认 marginal 下界独立相减。','',
        '误差按 q1/q2/q≥3 分组，避免把低驻留延迟响应直接套给高驻留比较。'
        '静态 q 区间内的所有整数容量组合均被枚举；q 的交叉取值是对资源未知的保守处理。'
        '寄存器/共享资源仍只来自静态估计；可能 spill 的候选不能触发改选。','',
        '| q 区间 | 原始配对差分残差范围 ns | 使用方式 |','|---|---:|---|']
    for name,b in theta['models']['full']['paired_error_bands'].items():
        lo,hi=b['delta_residual_ns_interval']
        lines.append(f'| {name} | [{lo:.3f}, {hi:.3f}] | nominal + 联合残差及其反射 |')
    lines += ['',
        '该范围是有限开发数据的经验误差包络，不是独立 GPU 置信区间、形式化硬件上界或盲测非回归保证。'
        '新几何/warp 数的外推会明确标记；不能把它们当成已经实测。','',
        '## 3. 独立 TMA 平台和主决策','',
        '`s_TMA = event_us × 1000 × SMs / calibration_grid`。grid 只用于校准归一，不进入编译期选择函数。'
        'C1.6 与 C2a 控制分开保留，不混成配对 worker；同几何的公共传输区间覆盖这两批控制。'
        '已测几何使用自己的控制范围，未测几何按 tile 输入+输出字节的共享速率区间外推。','',
        'C1.6 的 warm median：32×64/w8 为 `58.688 µs × 132 / 32768 = 236.414 ns/tile/SM`；'
        '32×128/w4 为 `54.672 µs × 132 / 16384 = 440.472 ns/tile/SM`。'
        'C2a 对应控制较低，全部原始 worker 值及差异保留于 θ。','',
        '**本版采用 `T_tile=max(s_TMA,S_MC)`，即显式设定 `s_base=0` 的稳态瓶颈近似。** '
        '没有从 fixture 截距或 MC 残差拟合新的固定加数，也没有引入历史时间偏置。'
        '这不是精确完整 kernel 总时间模型；未辨识的单次 setup/output 工作仍可能导致绝对时间偏差，下面单列诊断。'
        '该假设连同新形状的传输速率外推，均需要 D1 独立检验。','',
        '仅当候选在静态 q 与联合残差/公共传输场景下，最差 `T_C/T_D−1 < −3%` 才改选。'
        '多个稳健候选若 point roofline 在 3% 内可比，先保留 upstream placement，再选 MC point 较低者。'
        '这个 tie-break 不改变相对默认的 >3% 安全条件，不使用实测赢家。K1/MO/grid1 不参与训练或硬门限。','',
        '| 几何 | 默认 MC point ns / ρ | 决策 | 相对默认的配对包络变化 |','|---|---|---|']
    for s in selections:
        d=next(p for p in s['predictions'] if p['id']==s['default'])
        change='[0%, 0%]：传输平台内可比'
        if s['switched']:
            comp=next(r for r in s['comparisons'] if r['candidate']==s['choice'])
            change=f'[{100*comp["relative_change_interval"][0]:.2f}%, {100*comp["relative_change_interval"][1]:.2f}%]'
        lines.append(f'| {s["geometry"]} | {d["MC_point_ns"]:.3f} / {d["rho_point"]:.3f} | {s["choice"].split(":",1)[1]} | {change} |')
    lines += ['',
        '32×64/w8 的 vec1/vec2 均有稳健改进；多个候选落到公共传输平台后选择 vec1:N-first。'
        '32×128/w4 的候选差异被公共平台覆盖，保留默认是有模型依据的决定。'
        '旧完整 N-first kernel 的 vec8→vec1 为 111.120→60.592 µs，观察收益约 45.47%；'
        '这是已知开发数据的回放，不能叫新测量或外部验证。','',
        '## 4. 消融与基线','',
        '| 模型 | 拟合 RMSE ns | LOLO 配对 RMSE ns | LOGO 配对 RMSE ns | 32×64/w8 | 32×128/w4 |',
        '|---|---:|---:|---:|---|---|']
    for mode,item in theta['models'].items():
        cv=item['cross_validation'];rms={}
        for level in ('LOLO','LOGO'):
            errors=[r['delta_residual_ns'] for r in cv['paired_residuals'] if r['level']==level]
            rms[level]=float(np.sqrt(np.mean(np.square(errors))))
        choices=[next(r for r in dev['selections'] if r['mode']==mode and r['geometry']==g)['choice'].split(':',1)[1]
                 for g in ([32,64,8],[32,128,4])]
        lines.append(f'| {mode} | {item["model"]["diagnostic"]["training_rmse_ns"]:.3f} | {rms["LOLO"]:.3f} | {rms["LOGO"]:.3f} | {choices[0]} | {choices[1]} |')
    extra=next(r for r in dev['selections'] if r['mode']=='Envelope-only-blind' and r['geometry']==[32,128,4])
    lines += ['',
        '三个主消融均用相同 cell/recipe 重新拟合，并重新计算自己的配对残差：','',
        '- Comm-blind 删除 SHFL/WF/BAR 与 U/W 形状项；拟合和留出误差显著增大，错失 32×64 的稳健改选。',
        '- Load-blind 的总体配对 RMSE 没有变差，两个主决策也相同；不能声称当前数据已证明加载特征带来独立选择收益。'
        '预先按静态 bank conflict 定义的 within N128 vec2/4/8 同 vec 对照结果全部列于下表。',
        '- Resource/Envelope-blind 的拟合/配对误差扩大，32×64 选择 placement 不同；32×128 的 nominal 会偏好小 vec，'
        '但其重新校准后的宽误差范围仍阻止稳健改选。不能隐藏这一点，只展示一个人为不确定性较小的错误选择。','',
        f'额外的 Envelope-only 诊断保持 full 拟合、q 区间和配对残差不变，只移除公共传输平台，'
        f'32×128 即改选 `{extra["choice"].split(":",1)[1]}`。这直接展示传输包络为何会改变主决策。','',
        '| Bank-conflict 子集（同 vec、同条件） | full MAE ns | Load-blind MAE ns |',
        '|---|---:|---:|']
    for level in ('fit','LOLO','LOGO'):
        rows={r['mode']:r for r in dev['bank_conflict_subset_metrics'] if r['level']==level}
        lines.append(f'| {level}，每模型 {rows["full"]["pairs"]} 个 worker 对照 | {rows["full"]["MAE_ns"]:.3f} | {rows["Load-blind"]["MAE_ns"]:.3f} |')
    lines += ['',
        '上述子集是由 G_L 的 BC>0 定义，非按结果挑选。所有 q、正负误差和三种验证条件都保留，'
        '若某验证条件未改善，不解释成加载项已得到唯一因果识别。','',
        '| 几何 | 基线/选择器 | 选择 | MC natural ns | 旧 FULL µs | 旧 FULL 相对默认收益 |',
        '|---|---|---|---:|---:|---:|']
    for r in dev['baselines']:
        if r['method'] in MODES_FOR_TABLE:continue
        full='未测' if r['FULL_event_us'] is None else f'{r["FULL_event_us"]:.3f}'
        gain='未测' if r['FULL_improvement_vs_default_percent'] is None else f'{r["FULL_improvement_vs_default_percent"]:.2f}%'
        lines.append(f'| {r["geometry"]} | {r["method"]} | {r["choice"].split(":",1)[1]} | {r["MC_natural_ns"]:.3f} | {full} | {gain} |')
    worst=max(abs(r['absolute_error_percent']) for r in dev['absolute_FULL_diagnostics'])
    lines += ['',
        f'**绝对完整 kernel 时间并未闭合：** base=0 的瓶颈模型在这八个旧 FULL 点上最大绝对相对偏差为 {worst:.2f}%。'
        '这部分没有反向用于拟合 MC、缩窄配对区间或补一个固定加数。'
        '当前达成的是稳态选择的开发集目标，不能据此声称完整机制或跨形状有效性已经验证。','',
        '## 5. 同一批八个 D1 预测','',
        '没有新增 D1 编译或计时。旧历史审计及 split hash 沿用，八个几何完全相同。'
        '所有候选的静态特征、q 曲线、β×feature、ρ、配对变化区间、资源门和 nominal 选择均冻结于 '
        '[D1_predictions.json](D1_predictions.json)。','',
        '| 几何 | c3_v1 保守选择 | nominal 选择 | 原因 |',
        '|---|---|---|---|']
    for r in blind['cases']:
        lines.append(f'| {r["geometry"]} | {r["choice"].split(":",1)[1]} | {r["nominal_choice"].split(":",1)[1]} | {r["reason"]} |')
    reasons={}
    for r in blind['cases']:reasons[r['reason']]=reasons.get(r['reason'],0)+1
    lines += ['',
        f'保留默认的原因分布：`{reasons}`。保留默认不等于已知默认最优。'
        '传输受限的可比决定、静态资源上界未决和差分区间未分离的回退分开报告；'
        '四个大 tile 的寄存器上界可能超过 255，因此即使无 spill 的时间模型落在传输平台内，也不能称为已经确认传输受限。'
        '它们不再由 K1/grid1 一票否决或旧乘性比值导致。'
        '小 BM case 的 nominal/保守差异也保留；其中 8×128/w8 nominal 选 vec1，而保守规则仍保留默认，供 D1 检查是否过保守。','',
        '## 6. 冻结与停点','',
        '保留全部原始证据及旧 C3；新版本仅更新派生模型和预测。'
        '最终离线 validator 检查旧文件 hash、G_L/计数匹配、拟合与交叉验证重算、q 单调性、同条件配对、'
        '辅助数据不进入主目标、代码/θ/预测冻结，以及同一八个盲测 case 零新增 binary/计时。','',
        '复现：`.venv/bin/python -m experiments.tma_reduction_layout.phase11.c31.validate --committed`。'
        '已停在 C3.1 commit；没有自行启动 D1，也没有提交 PR。','',
        '[参数与 CV](theta_H100.json) · [开发/消融](development.json) · [MC 拟合图](figures/MC_fit.png) · [原始来源绑定的校准 cell](calibration_cells.json) · '
        '[辅助诊断](auxiliary_diagnostics.json) · [冻结清单](freeze.json)','']
    (c.V1/'REPORT.md').write_text('\n'.join(lines))
    (c.OUT/'REPORT.md').write_text('# Stage C3.1\n\n[完整 c3_v1 报告](c3_v1/REPORT.md)。'
        '开发集：32×64/w8 改选 vec1:N-first，32×128/w4 保留默认；D1 未编译/计时。\n')
    figures(theta,dev)


MODES_FOR_TABLE={'full','Load-blind','Comm-blind','Resource/Envelope-blind','Envelope-only-blind'}


def figures(theta,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.tma_reduction_layout.phase11.c31.data import development
    from experiments.tma_reduction_layout.phase11.c31.calibrate import predict
    _,_,cells=development();model=theta['models']['full']['model']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for ax,n in zip(axes,(64,128)):
        for q in ('1','2','4','natural'):
            rs=[r for r in cells if r['case']['BN']==n and r['q_condition']==q]
            ax.scatter([r['service_ns'] for r in rs],
                       [predict(model,r['case'],r['q'],theta['primitive_shape_tables']) for r in rs],label='q='+q,s=25)
        ax.plot([0,470],[0,470],color='0.5',ls='--',lw=1)
        ax.set_title(f'BM32 BN{n}: direct MC fit')
        ax.set_xlabel('Measured ns / tile / SM');ax.set_ylabel('Predicted ns / tile / SM')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    path=c.V1/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'MC_fit.png',dpi=180);plt.close(fig)


if __name__=='__main__':run()
