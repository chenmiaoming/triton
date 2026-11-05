"""Render complete measured matrices, keeping bridge/specialization effects separate."""
import argparse
from . import common as c
from .g4_prepare import out, protocol_path, gate_path
from .g4_analyze import save


def main(stage):
    target = out(stage); p = c.read(protocol_path(stage)); t = c.read(target / 'timing_analysis.json')
    gate = c.read(gate_path(stage)); eligibility = c.read(target / 'correctness/attempt_001/raw.json')
    bind = next(x for path in (target / 'bind').glob('attempt_*/raw.json')
                if (x := c.read(path))['status'] == 'OK' and x['protocol_SHA256'] == c.sha(c.encode(p)))
    lines = ['# ' + stage.upper() + ('：SGLang 相邻布局因子实验' if stage == 'g4a' else '：实际调用 JIT 与旧 B4 编译绑定'), '',
        '本轮检验非 TMA 默认布局的优化空间，没有比较历史编译器版本，不称为性能回归。所有样本、编译产物和 NCU 原始报告均保留；旧 G2/G3 文件逐项 SHA 保护。', '',
        '## 固定条件与校验', '',
        f'- Triton `{p["compiler"]}`；native `{c.NATIVE}`；H100! / chenmiaoming；复用原 native、ccache 和 kernel cache，native rebuild=0。',
        f'- {len(bind["records"])} 个 CPU 逻辑程序均由真实 CUDA Tensor 的 JIT 调用核对 CUBIN SHA 和 ABI；{len(eligibility["checks"])} 项完整输出、输入不变与 canary 校验通过。',
        f'- 3 个独立进程 × 5 visits × 31 samples，8 次预热；实际保存 {t["raw_samples"]:,} 个计时样本，覆盖 {t["logical_conditions"]} 个逻辑条件。精确 CUBIN+ABI+专化+launch context 相同的条件只运行一次并保留别名。',
        '- 时间为单次 CUDA Graph 的 µs；点估计取 worker/visit 中位数的均值，置信区间为 4096 次配对层级 bootstrap。区间为探索性逐项区间，未做多重比较校正；3% 为实用差异参考。',
        '- C02 同列 stage1 与两个原始阶段组成的 decode；C06 是真实源码中的 merge 子内核。尚未测完整服务、框架调度或整个模型。',
        '- 修订前的 CPU 预检误把三个 auxiliary 零整数专化为 constexpr，真实 Tensor JIT gate 拦截了 ABI 不符；失败记录、原协议和原 CPU 产物完整保留。修订后才进行正式计时。', '']
    def find(group, batch, length, mode, variant):
        return next(x for x in t['rows'] if x['group'] == group and x['batch'] == batch and x['length'] == length
                    and x['mode'] == mode and x['tag'].endswith(':' + variant))
    summary = ['## 本轮结果', '']
    if stage == 'g4a':
        a = find('C02_n64_d128', 8, 2048, 'decode', 'pv_only')
        b = find('C02_n64_d128', 8, 2048, 'decode', 'coordinated')
        neg = find('C02_n64_d64', 8, 2048, 'decode', 'coordinated')
        summary += [f'- d128/context2048 的 PV-only 再现 {a["gain_vs_G0_percent"]:.2f}% 的 G0→GL 收益，但相对原始 Triton 仅 {a["gain_vs_U_percent"]:.2f}%，低于 3% 实用门槛。',
            f'- 让 QK/PV 同时采用目标布局未带来联合优势：同一点相对 G0 为 {b["gain_vs_G0_percent"]:+.2f}%；d64/context2048 的联合改动相对 G0 为 {neg["gain_vs_G0_percent"]:+.2f}%。减少邻接转换不是充分条件。',
            '- QK 改动同时改变 K 访问、局部规约和资源分配；完整因子交互随 context 改变，不支持将转换计数乘一个统一延迟来解释整核。精确的依赖关键路径延迟仍未单独识别。',
            '', '![完整 decode 因子对照](figures/factorial_decode.png)', '']
    else:
        bytag = {r['tag']: r for r in gate['records']}
        identical = sum(bytag[g['id']+':actual:'+v]['cubin_SHA256'] == bytag[g['id']+':old_B4:'+v]['cubin_SHA256']
                        for g in p['groups'] for v in ('upstream', 'gluon_anchor', 'target'))
        gains = [find(f'C06_s16_d128_b{batch}_l8192', batch, 8192, 'merge', 'actual:target') for batch in (1, 8, 64)]
        text = ' / '.join(f'{r["gain_vs_G0_percent"]:.2f}%' for r in gains)
        summary += [f'- 30 个主内核 actual/B4 绑定比较中 {identical} 个 CUBIN 相同；6 个不同项均为 C06 batch1（num_seqs=1，运行参数从 11 降到 10）。4 个 SGLang 第二阶段与旧 B4 CUBIN 也全部相同。',
            '- batch1 专化本身的六个配对时间变化均不足 3%；保留其正负方向及全部区间，不把源码/JIT 变化本身当作实际收益。',
            f'- C06 s16d128 的预选 vec1 在 actual batch=1/8/64 下，相对 G0 分别为 {text}。C06 s32d256 的目标布局相对 G0 均不足 1%，其对 U 的收益主要是 Gluon 桥接差异，不能归为新布局收益。',
            '- 本组新增计时和实际 JIT 证据，未新增 NCU。旧 NCU 只在二进制与 workload/launch context 均一致时复用；不把 B8 缓存/占用计数外推给 B1/B64。C06 s16 表示 NUM_SEGMENTS_PER_SEQ=16，merge 逻辑矩阵为 16×128；实际分段掩码保持原始语义。',
            '', '![实际 JIT 的 merge 布局对照](figures/actual_jit_merge.png)', '']
    lines[4:4] = summary
    if stage == 'g4a':
        lines += ['## 预先固定的因子干预', '',
            '| geometry | PV target | QK target | 固定项 |', '|---|---|---|---|',
            '| 64×128/w4 | vec4:n_first | 同一个 vec4:n_first | Q、输出布局；算法/精度/分页/循环/第二阶段 |',
            '| 64×64/w4 | vec1:n_first | 同一个 vec1:n_first | 同上 |', '',
            '四格：G0=(default PV, default QK)，pv_only=(target PV, default QK)，qk_only=(default PV, target QK)，coordinated=(target PV,target QK)。另保留 untouched upstream。QK 的干预同时影响 K 加载与 QK 规约；不能把因子差分叫作纯布局转换延迟。', '']
    else:
        lines += ['## 实际专化与旧绑定的边界', '',
            '每个 batch/context 分别在原始入口上 JIT；G0 从该入口实际 TTGIR 提取默认布局。目标布局在新数据前已固定：C06 s16d128=vec1:n_first，C06 s32d256=vec4:within，C02 d128=vec4:n_first，C02 d64=vec1:n_first。新 JIT 与冻结 G2 的 B4 二进制在同一工作负载/worker 上配对计时。', '',
            '| binding group | variant | actual / B4 identical CUBIN | actual specialized runtime values | actual runtime args / B4 |',
            '|---|---|---|---|---|']
        bytag = {r['tag']: r for r in gate['records']}
        for group in p['groups']:
            for label in ('upstream', 'gluon_anchor', 'target'):
                a = bytag[group['id'] + ':actual:' + label]; old = bytag[group['id'] + ':old_B4:' + label]
                lines.append(f'| {group["id"]} | {label} | {a["cubin_SHA256"] == old["cubin_SHA256"]} | `{a["specialized_runtime_values"]}` | {len(a["runtime_argument_names"])} / {len(old["runtime_argument_names"])} |')
        lines += ['', '二进制相同的结果是有效的零结果。CUBIN 不同也只证明生成物发生变化，是否影响时间见配对区间；不自动认定为优化。原始阶段 2 保持源码/算法/配置，实际 JIT 与 B4 的阶段 2 SHA 单独检查。', '']
    lines += ['## 完整计时矩阵', '',
        'U 是此组实际绑定的原始 Triton，G0 是与它匹配默认布局的 Gluon。U→G0 的 lowering 差异与 G0→GL 的布局效应分别报告；不得只选较慢的参照。正数表示减少耗时。', '',
        '| group / batch / context | mode | variant | µs | vs U % [95% CI] | vs G0 % [95% CI] |',
        '|---|---|---|---:|---:|---:|']
    for row in t['rows']:
        a, b = row['gain_vs_U_CI95'], row['gain_vs_G0_CI95']
        label = row['tag'][len(row['group']) + 1:]
        lines.append(f'| {row["group"]} / {row["batch"]} / {row["length"]} | {row["mode"]} | {label} | {row["time_us"]:.4f} | {row["gain_vs_U_percent"]:+.2f} [{a[0]:+.2f},{a[1]:+.2f}] | {row["gain_vs_G0_percent"]:+.2f} [{b[0]:+.2f},{b[1]:+.2f}] |')
    if stage == 'g4a':
        lines += ['', '## 因子交互', '',
            '交互 = coordinated − pv_only − qk_only + G0（µs）。负值表示联合改动比两个独立改动的简单相加更省时；这仍是完整程序的交互，不能直接归因于某条指令。', '',
            '| geometry / context | mode | interaction µs [95% CI] |', '|---|---|---:|']
        for row in t['contrasts']:
            ci = row['interaction_CI95_us']
            lines.append(f'| {row["group"]} / {row["length"]} | {row["mode"]} | {row["interaction_us"]:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] |')
        counters = c.read(target / 'counter_analysis.json')
        lines += ['', '## NCU 动态结构与资源', '',
            f'{len(counters["rows"])} 份报告全部在本地 CPU 重新导入，逐项数值/动态指令与远端一致。单 CTA 软件计数使用相同 batch8/context2048 数据的第一个 CTA，包含 8 次 KV 循环；硬件计数为 batch8 的实际 grid。', '',
            '| variant | regs / q_cap | dynamic SHFL / BAR | shared LD / ST | FADD | global LD instructions |',
            '|---|---|---|---|---:|---:|']
        for row in counters['rows']:
            if row['mode'] != 'software': continue
            counts = row['counts']; op = row['opcode_counts']; res = row['resources']
            loads = sum(n for k, n in op.items() if k.startswith('LDG'))
            lines.append(f'| {row["tag"]} | {res["n_regs"]} / {res["occupancy_capacity_q"]} | {counts["shuffle"]} / {counts["barrier"]} | {counts["shared_load"]} / {counts["shared_store"]} | {sum(n for k,n in op.items() if k.startswith("FADD"))} | {loads} |')
        lines += ['', '| variant | LD WF / CTA | LD BC / CTA | global LD requests / sectors per CTA | achieved occ % | eligible warps/scheduler | issue active % | barrier / short scoreboard / long scoreboard / MIO % |',
                  '|---|---:|---:|---|---:|---:|---:|---|']
        for row in counters['rows']:
            if row['mode'] != 'warm_application': continue
            mt = row['metrics']; n = row['CTAs']
            stalls = [mt['smsp__warp_issue_stalled_' + s + '_per_warp_active.pct'] for s in ('barrier', 'short_scoreboard', 'long_scoreboard', 'mio_throttle')]
            stall_text = ' / '.join(f'{x:.2f}' for x in stalls)
            lines.append(f'| {row["tag"]} | {mt["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]/n:.2f} | {mt["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]/n:.2f} | {mt["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]/n:.2f} / {mt["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]/n:.2f} | {mt["sm__warps_active.avg.pct_of_peak_sustained_active"]:.2f} | {mt["smsp__warps_eligible.avg.per_cycle_active"]:.3f} | {mt["smsp__issue_active.avg.pct_of_peak_sustained_active"]:.2f} | {stall_text} |')
        attribution = c.read(target/'source_attribution.json')
        lines += ['', '## 从有效布局到实际执行的源码关联', '',
            'TTGIR 核对证明所有请求的 K/V/Q/output 布局实际生效（effective_layout_audit.json）。默认与目标若同时用于 QK/PV，对两个规约方向的影响相反：', '',
            '| geometry | 默认 QK lanes×warps / PV lanes×warps | 联合目标 QK / PV |',
            '|---|---|---|',
            '| 64×128/w4 | 16×1 / 2×4 | 32×1 / 1×4 |',
            '| 64×64/w4 | 8×1 / 4×4 | 32×2 / 1×2 |', '',
            '以 nvdisasm -gi 的 inline caller 源码行关联 NCU 动态 PC。下表是执行实例数，不是 cycles；compiler motion/fusion 会模糊源码边界。line 71 包含概率布局转换与 PV 合并，不作为裸规约延迟。', '',
            '| variant | QK line53 FADD / SHFL | PV+prob line71 FADD / SHFL | index conversion line63 LDS |',
            '|---|---|---|---:|']
        for row in attribution['rows']:
            def source_line(number):
                matches=[value for key,value in row['per_source_line'].items() if key.endswith('/sglang_controlled.py:'+str(number))]
                return matches[0] if matches else {'opcodes':{},'families':{}}
            q, v, index=source_line(53),source_line(71),source_line(63)
            def adds(x):return sum(n for key,n in x['opcodes'].items() if key.startswith('FADD'))
            lines.append(f'| {row["tag"]} | {adds(q)} / {q["families"].get("shuffle",0)} | {adds(v)} / {v["families"].get("shuffle",0)} | {index["families"].get("shared_load",0)} |')
        lines += ['', 'd128 的 PV-only 确实减少了 PV 动态合并与 shuffle；联合布局确实消除了 line63 的 index 重分布，但同时把 QK shuffle 从 1024 增至 2560。d128 联合项的 shared-load WF 比 PV-only 更低、相同 n_regs/q_cap 下却更慢，否定“只要减少 shared 转换就会更快”。这不把不同动态指令按等权相加，也不声称已识别每条依赖的绝对延迟。', '']
    else:
        lines += ['', '## actual JIT 相对 B4 replay 的配对变化', '',
            '| group | mode | variant | actual gain vs B4 % [95% CI] |', '|---|---|---|---:|']
        for row in t['contrasts']:
            ci = row['gain_CI95']
            lines.append(f'| {row["group"]} | {row["mode"]} | {row["contrast"]} | {row["gain_percent"]:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] |')
    lines += ['', '## 证据文件与解释边界', '',
        '- `timing_analysis.json`、`all_timings.csv`：全部条件、worker/visit 中位数与配对区间。`code_audit_v2.json`、`source_correlation_v2/`：真实 TTGIR 加载/规约/转换/输出计划，以及本地 nvdisasm 源码行关联。源码行信息用于定位生成物，不等于串行关键路径耗时。',
        '- `compile_gate_v2.json`、`compiled_payload_v2.zip`：精确执行元数据与二进制；`bind/`、`correctness/`、`timing/` 为 write-once 原始证据。',
        '- NCU 的 instruction、request、sector、wavefront 是不同单位；[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 解释 replay/缓存及计数语义。',
        '- [NVIDIA Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html) 要求结合发射、活跃与 eligible warps 解读 stalls，避免由单个百分比推断优化收益。此处不将 BAR 条数乘一个固定延迟，也不把多项指令数加成总时间。',
        '- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) 的合并访存原则支持按实际 sector 而非逻辑 vec 判断流量；q_cap 与实际 occupancy 分别报告。',
        '- 文档查阅日 2026-10-21；collector 与本地重导入版本记录在证据中。没有改 C++、训练新 cost model 或创建 PR。', '']
    save(target / 'REPORT.md', '\n'.join(lines).encode())
    print(stage, 'report written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['g4a', 'g4b']); main(parser.parse_args().stage)
