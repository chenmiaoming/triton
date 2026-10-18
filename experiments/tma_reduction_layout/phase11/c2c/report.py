"""Academic calibration report with explicit identification failures."""
from collections import defaultdict
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c2c import common as c


def run():
    a = c.read(c.OUT/'analysis.json')
    composition = c.read(c.OUT/'composition.json')
    gate = c.read(c.OUT/'artifact_gate.json')
    counters = c.read(c.OUT/'counter_gate.json')
    compiled = c.read(c.OUT/'compile_gate.json')
    meta = gate['metadata']
    lines = ['# Stage C2c — 单次组合、固定 R 局部对照与资源补测', '',
        '日期：2026-10-18。H100 sm90，原 host-created descriptor、BF16→FP32 max，沿 M 规约。', '',
        '**采集与证据校验已完成；独立物理底座与纯传输分量尚未辨识。** 相同 q 下，允许每个几何/worker 单独拟合一个公共非负 B，仍有 '
        f"{sum(not r['all_points_within_3_percent'] for r in composition['best_common_base_diagnostics'])}/{len(composition['best_common_base_diagnostics'])} 组不能在全部布局上闭合到 3%。"
        '因此不把完整对照减去一个经验常数，不把全图截距加入静态选择器，也不把拟合残差命名为纯硬件延迟。', '',
        '## 1. 实验与保护范围', '',
        f"- CPU 预检覆盖 {len(meta)} 个逻辑程序；{compiled['reused']} 个原 CUBIN 逐字节复用，其余使用持久 Triton 缓存。native SHA 与 pinned compiler 保持一致，ccache 统计未改变，没有重新构建 Triton。",
        f"- GPU 正确性检查 {gate['checks']} 项全部通过；包含 quarter/random/boundary 输入。固定 R 的随机 FP32 checksum 允许已记录的舍入容差，其余检查采用数值相等。",
        '- 3 次固定调用，每测量单元 12 个单 kernel Graph 内事件样本，3 次交错访问；不使用 NCU duration 训练。自然驻留别名和相同 CUBIN 复用，未按结果重复测量。',
        f"- 原始 NCU 报告 {len(counters['reports'])} 份，软件动态采集 {len(counters['dynamic'])} 次，硬件计数采集 {len(counters['hardware'])} 次；全部重新导入并绑定 CUBIN。",
        '- 保留首次 CPU 预检失败、所有技术修订和源快照。第一次失败是 fixed-R 的嵌套 SliceLayout 扩维轴顺序，发生在分配 GPU 之前。',
        '- 11 个 D1 几何已在补测前保护；未编译、加载、采集或计时其候选布局。旧 C3/C3.1 和全部既有 raw/compiler artifacts 保持原样。', '',
        '标定几何：`8×64/w8, 8×128/w4, 8×256/w4, 16×64/w4, 16×128/w4, 32×64/w8, 32×128/w4, 64×64/w8`。另两个非 D1 的 R128/256 几何只用于单次程序资源标定。', '',
        '## 2. 小 BM 的完整 TMA 对照', '',
        '下表为 genuine loop-free transfer control：完整 TMA tile、wait、首行读取、BF16→FP32 与同尺寸写回。它不是纯 HBM 延迟。'
        '全图归一化包含全图截距；grid slope 与截距分别保留，后者不进入选择器。四分之一/半图可能有缓存非线性，原始三点和拟合残差未删除。', '',
        '| BM×BN / W | 全图 median µs | 全图归一化 ns/tile/SM | grid slope ns/tile/SM | 全图截距 µs |',
        '|---|---:|---:|---:|---:|']
    controls = defaultdict(list)
    for row in a['oneshot']:
        if row['family'] in ('ONE_TRANSFER', 'ORIGINAL_TRANSFER') and row['q_condition'] == 'natural':
            controls[tuple(row['geometry'])].append(row)
    for (m,n,w), rows in sorted(controls.items()):
        lines.append(f"| {m}×{n} / {w} | {st.median(r['full_us'] for r in rows):.3f} | {st.median(r['full_normalized_ns'] for r in rows):.2f} | {st.median(r['tile_slope_ns'] for r in rows):.2f} | {st.median(r['global_intercept_us'] for r in rows):.2f} |")
    lines += ['', '同为 128 MiB 输入，8×64/w8 的单 tile 只有 1 KiB，但全图耗时约 129 µs；8×128/w4 约 60 µs。'
        '这证明当前范围内仅按 tile 字节数外推不充分，不能将所有平台都解释成 HBM 带宽屋顶。W、CTA 数、TMA 指令/等待和驻留条件均需纳入解释；此对照本身没有独立区分每一种原因。', '',
        '## 3. 单次与稳态的组合检验', '',
        '`MC` 采用驻留双槽与可观察的逐次输出；K1 只搬入一个 tile，K>1 搬入两个固定 tile。'
        '`FULL/TRANSFER` 的多 K 版本逐次搬入新 origin，槽位复用前的 K−1 次 barrier 属于显式 fixture；真正单次程序没有运行时 K 循环。'
        '单独测 ROW 与 direct-row SCAFFOLD，不将其当作已知纯底座。', '',
        'K16/64/256 拟合稳态斜率；K1/2/4 与全图 K1 单列。先对 G/SM 取斜率，分离全图启动/尾部截距。'
        '同 worker、相同实际 q 比较 FULL、MC 与 transfer control；自然条件实际 q 不相同的记录只列为资源响应，未混入公共底座检验。', '',
        '| 几何 / 条件 | 公共 B 最优拟合后的最大绝对误差，3 worker 范围 |', '|---|---:|']
    grouped = defaultdict(list)
    for row in composition['best_common_base_diagnostics']:
        grouped[tuple(row['geometry']), row['q_condition']].append(row['max_abs_error_percent'])
    for (geometry, q), values in sorted(grouped.items()):
        lines.append(f"| {geometry} / q={q} | {min(values):.2f}%–{max(values):.2f}% |")
    lines += ['', '这个检验已经允许事后、每几何、每 worker 的 B，仍无法闭合时，失败不能仅归因于跨形状底座估计不足。'
        '稳态重复子图与单次程序的流水线、资源和依赖调度差异仍在其中。支持的结论是组合关系需要修正；没有证据给出唯一的 B_tile 与纯 T_transfer。'
        'C3.1 的 18.80% 偏差不能被简单宣称已由补 B 消除。后续模型必须报告组合选择的交叉验证误差。', '',
        '## 4. 固定 R 的同语义配对局部对照', '',
        'R4/8/16/32，共 18 组合法 `(p,columns)`，每组四个版本：native、wide、fixture、fixture_wide。'
        '每线程逻辑元素数与输入状态更新固定，四个版本使用相同 checksum consumer，并在循环后观察全部最终状态。'
        'native/wide 对同一分组计算相同 max；wide 采用精确的 ×2、max、×0.5 来阻断 BF16 max 识别，其额外操作通过对应 fixture 对照保留。', '',
        '跨 p 改变的是待研究的局部分组，不声称不同分组具有相同完整算子语义。'
        'native−fixture、wide−fixture_wide 和双重差分为有效子图对照，包含指令调度/打包/转换交互，不能直接命名为裸 max 指令延迟。'
        '自然条件若寄存器分配导致 q 不同，明确标为驻留不匹配；受控 q 对照另列。', '',
        '| R32，q=4 | native−fixture ns | wide−fixture_wide ns | 双重差分 ns |',
        '|---|---:|---:|---:|']
    local = defaultdict(list)
    for row in a['local_pairs']:
        if row.get('contrast_available') and row['R'] == 32 and row['q_condition'] == '4':
            local[row['p'], row['columns']].append(row)
    for (p, columns), rows in sorted(local.items()):
        lines.append(f"| p={p}, columns={columns} | {st.median(r['native_net_ns'] for r in rows):.2f} | {st.median(r['wide_net_ns'] for r in rows):.2f} | {st.median(r['difference_in_differences_ns'] for r in rows):.2f} |")
    lines += ['', 'p=1 的 max-deleted 对照应接近零差值，保留微小正负值，不截断成纯正惩罚。'
        '各 R、各 p、全部 q 与 worker 数值在 analysis.json 中保留。', '',
        '### BF16 标量干预未成功，不能据此辨识 packing 单项', '',
        '另外新增 4 个 p=2 BF16 scalar combine 程序，复用 16 个 native/wide/fixture CUBIN，'
        '重新配对采集 3 次调用、34,560 个 Graph 事件样本、2 份 NCU 报告与 180 项正确性检查。'
        'PTX 的 scalar max site 数确实为 native packed site 的 2 倍，但 NCU 显示实际 max 执行次数相同：', '',
        '| R / p=2 | native 动态 max/iteration/CTA | scalar-source 动态 max | wide FP32 动态 max |',
        '|---|---:|---:|---:|',
        *[f'| {r} | {r} | {r} | {2*r} |' for r in (4, 8, 16, 32)], '',
        '后端将 scalar PTX 重新合并为 HMNMX2.BF16_V2，原先期望的 scalar max = 2×native 检查明确失败，'
        '保存在 packing_supplement/validation_attempts，未通过改名把它包装成成功干预。'
        '另 8 个 CPU-only 修订（max+cvt 同 asm、显式复制 BF16 halves）仍生成打包 SASS，未继续 GPU 计时。'
        '保留源快照、一次上传失败和一次 SliceLayout 编译失败，不修改原始 compiler artifacts。', '',
        '**因此尚未独立辨识 packing 与 widening 的裸成本。** fixed-R native/fixture 对比是可用的净局部子图对照；'
        'packing supplement 的 scalar−native 微小计时差只反映最终实现/调度差异，不能解释成打包收益。'
        'C3.2 使用净子图响应，不给 packing 指令人为指定延迟。全阶段共 305,748 个 Graph 事件样本、24 份原始 NCU 报告并逐份本地重导入。', '',
        '## 5. 物理寄存器与边界', '',
        '| 非 D1 单次程序 | R | vec8 n_regs | local bytes |', '|---|---:|---:|---:|']
    for tag, item in meta.items():
        if item.get('resource_only') and item['vec'] == 8 and item['placement'] == 'n_first':
            resource = gate['resources'][tag]['attributes']
            r = item['M']*item['N']//(32*item['num_warps'])
            lines.append(f"| {item['M']}×{item['N']}/w{item['num_warps']} | {r} | {resource['num_regs']} | {resource['local_bytes']} |")
    lines += ['', '固定 R 的人工 state/consumer fixture 与单次完整程序具有不同活跃区间；资源模型必须按程序族校准。'
        '例如 R32/p16/columns2 的 native 与 wide 静态分配分别为 72 与 111 个寄存器，而 R256 的单次 vec8 程序只分配 45 个。'
        '这些是实际编译分配与 CUDA attribute 证据，不等价于逻辑元素存储量，也不能把“未观察到 spill”外推成任何布局均无 spill。', '',
        '## 6. 交付与停点', '',
        '本阶段只新增 C2c 证据。C3.2 的方法在正式 GPU 数据之前通过 C32_method_binding.json 绑定；'
        '下一阶段更新静态组合与资源模型、构造单侧配对不确定性，并冻结分层预测。D1 尚未开始。', '',
        '[原始来源绑定分析](analysis.json) · [组合检验](composition.json) · [NCU 计数与报告绑定](counter_gate.json) · '
        '[编译审计](compile_gate.json) · [资源/正确性](artifact_gate.json) · [D1 保护清单](preregistration/D1_reservation.json)', '']
    c.derived(c.OUT/'REPORT.md', '\n'.join(lines).encode())
    figures(a, composition)


def figures(a, composition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = c.OUT/'figures'
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    rows = [row for row in composition['pairs'] if row['same_q']]
    x = [row['zero_base_prediction_ns'] for row in rows]
    y = [row['full_tile_slope_ns'] for row in rows]
    ax.scatter(x, y, s=14, alpha=.6)
    high = max(max(x), max(y))*1.05
    ax.plot([0, high], [0, high], 'k--', linewidth=1)
    ax.set(xlabel='max(control, steady MC), ns/tile/SM', ylabel='Single-pass full grid slope, ns/tile/SM',
           title='Matched-residency composition test')
    fig.tight_layout()
    fig.savefig(folder/'composition.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [row for row in a['local_pairs'] if row.get('contrast_available') and row['R'] == 32 and row['q_condition'] == '4']
    ps = sorted({row['p'] for row in rows})
    for key, label in [('native_net_ns', 'BF16-friendly max minus fixture'), ('wide_net_ns', 'FP32 max minus scaled fixture')]:
        ax.plot(ps, [st.median(row[key] for row in rows if row['p'] == p) for p in ps], 'o-', label=label)
    ax.set(xscale='log', xlabel='p (R=32 held fixed)', ylabel='Paired increment, ns/tile/SM',
           title='Fixed-R local contrasts at q=4')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/'fixed_R.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    run()
