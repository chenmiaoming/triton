"""Descriptive C1 slopes, component trends and honest template limitations."""
import argparse
from collections import Counter, defaultdict
import math
import re
import statistics as st

from experiments.tma_reduction_layout.phase11.c1 import common as c


def regression(xs, ys):
    xm, ym = st.mean(xs), st.mean(ys)
    slope = math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.fsum((x - xm) ** 2 for x in xs)
    intercept = ym - slope * xm
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    total = math.fsum((y - ym) ** 2 for y in ys)
    return {'slope_us_per_iteration': slope, 'intercept_us': intercept,
            'residuals_us': residuals, 'max_abs_residual_us': max(map(abs, residuals)),
            'R_squared': 1 - math.fsum(r * r for r in residuals) / total if total else None}


def opcodes(text):
    ops = Counter()
    for line in text.splitlines():
        match = re.match(r'\s*(?:@!?%\w+\s+)?([\w.]+)\s+.*;', line)
        if match:
            ops[match[1]] += 1
    return ops


def templates():
    gate = c.read(c.OUT / 'artifact_gate.json')
    old = c.read(c.base.OUT / 'stage_b/artifact_gate.json')
    result = {}
    for n, w in ((64, 8), (128, 4)):
        for vec in (1, 2, 4, 8):
            tag = f'M32_N{n}_w{w}:vec{vec}:MR'
            now = c.OUT / 'artifacts' / gate['accepted_attempt'] / 'export' / tag.replace(':', '/')
            prior = c.base.OUT / 'stage_b/artifacts' / old['accepted_attempt'] / 'export' / f'M32_N{n}_w{w}' / 'bfloat16/max/host' / f'fixed_vec{vec}'
            a, b = opcodes((now / 'kernel.ptx').read_text()), opcodes((prior / 'kernel.ptx').read_text())
            amax = {k: v for k, v in a.items() if k.startswith('max.')}
            bmax = {k: v for k, v in b.items() if k.startswith('max.')}
            match = set(amax) == set(bmax)
            result[tag] = {'MR_max_PTX_opcode_sites': amax, 'full_kernel_max_PTX_opcode_sites': bmax,
                'max_opcode_kind_set_equal': match,
                'precision_template_mismatch': any('bf16' in k for k in bmax) and not any('bf16' in k for k in amax),
                'MR_PTX_SHA256': c.sha((now / 'kernel.ptx').read_bytes()),
                'full_kernel_PTX_SHA256': c.sha((prior / 'kernel.ptx').read_bytes()),
                'scope': 'static diagnostic; repeated PCs are validated independently by NCU',
                'full_output_conversion_in_MR': False, 'reduction_result_feedback_in_MR': False}
    return result


def derive():
    protocol = c.load_protocol()
    gate = c.read(c.OUT / 'artifact_gate.json')
    counters = c.read(c.OUT / 'counter_gate.json')
    assert counters['status'] == 'OK'
    worker = []
    for invocation in (1, 2, 3):
        raw = c.read(c.OUT / 'timing' / f'invocation_{invocation}' / 'raw.json')
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert raw['payload_SHA256'] == c.sha((c.OUT / 'frozen_payload.zip').read_bytes())
        assert raw['compilation'] == {'Triton_imported': False, 'subprocess_calls': 0, 'compile_calls': 0}
        assert raw['CUDA_return_codes'] == [0]
        assert len(raw['checks']) == 128 and all(row['correctness']['pass'] for row in raw['checks'])
        samples = defaultdict(list)
        for row in raw['samples']:
            meta = gate['metadata'][row['tag']]
            assert row['CUBIN_SHA256'] == meta['cubin_SHA256']
            assert row['parameters'][8] == row['K']
            assert len(row['samples_us']) == 5 and all(math.isfinite(x) and x > 0 for x in row['samples_us'])
            samples[row['tag'], row['grid'], row['K']].extend(row['samples_us'])
        assert len(samples) == 128 and all(len(v) == 20 for v in samples.values())
        fits = {}
        for tag in gate['metadata']:
            for grid in (1, 64 * raw['environment']['SM_count']):
                ys = [st.median(samples[tag, grid, k]) for k in protocol['K']]
                fits[tag, grid] = {**regression(protocol['K'], ys), 'medians_us_by_K': dict(zip(protocol['K'], ys)),
                    'adjacent_slopes_us_per_iteration': [(ys[i + 1] - ys[i]) / (protocol['K'][i + 1] - protocol['K'][i]) for i in range(3)]}
        worker.append({'raw': raw, 'fits': fits, 'samples': samples})
    sm_counts = {w['raw']['environment']['SM_count'] for w in worker}
    assert len(sm_counts) == 1, 'Report differing grids separately before aggregating slopes'
    grids = (1, 64 * next(iter(sm_counts)))
    rows = []
    for tag, meta in gate['metadata'].items():
        for grid in grids:
            fits = [w['fits'][tag, grid] for w in worker]
            slopes = [f['slope_us_per_iteration'] for f in fits]
            rows.append({'tag': tag, 'M': meta['M'], 'N': meta['N'], 'num_warps': meta['num_warps'],
                'vec': meta['vec'], 'family': meta['family'], 'grid': grid, 'worker_fits': fits,
                'median_slope_us_per_iteration': st.median(slopes), 'min_max_slope_us_per_iteration': [min(slopes), max(slopes)],
                'median_times_us_by_K': {k: st.median([f['medians_us_by_K'][k] for f in fits]) for k in protocol['K']},
                'median_adjacent_slopes_us_per_iteration': [st.median(f['adjacent_slopes_us_per_iteration'][i] for f in fits) for i in range(3)],
                'resource': gate['resources'][tag], 'counters': counters['binaries'][tag]})
    old = c.read(c.base.OUT / 'stage_b/analysis.json')
    trends = []
    for n, w in ((64, 8), (128, 4)):
        full = next(row for row in old['layout_results'] if row['case_id'] == f'M32_N{n}_w{w}:bfloat16:max:host' and row['fraction'] == 1)
        full_times = {vec: full['median_of_worker_medians_us'][f'M32_N{n}_w{w}:bfloat16:max:host:fixed_vec{vec}'] for vec in (1, 2, 4, 8)}
        for grid in grids:
            components, sums = {}, {}
            for vec in (1, 2, 4, 8):
                a, b = (next(row for row in rows if row['N'] == n and row['vec'] == vec and row['family'] == family and row['grid'] == grid)
                        for family in ('ML', 'MR'))
                per_worker = [x['slope_us_per_iteration'] + y['slope_us_per_iteration'] for x, y in zip(a['worker_fits'], b['worker_fits'])]
                components[vec] = {'ML': a['median_slope_us_per_iteration'], 'MR': b['median_slope_us_per_iteration'],
                                   'paired_worker_sum_slopes': per_worker,
                                   'median_paired_worker_sum_slope': st.median(per_worker)}
                sums[vec] = a['median_slope_us_per_iteration'] + b['median_slope_us_per_iteration']
            pairs = []
            for a, b in ((1, 2), (2, 4), (4, 8)):
                pairs.append({'from_vec': a, 'to_vec': b, 'delta_ML_us_per_iteration': components[b]['ML'] - components[a]['ML'],
                    'delta_MR_us_per_iteration': components[b]['MR'] - components[a]['MR'],
                    'delta_sum_us_per_iteration': sums[b] - sums[a],
                    'sum_relative_change_percent': 100 * (sums[b] / sums[a] - 1),
                    'full_kernel_relative_change_percent': 100 * (full_times[b] / full_times[a] - 1)})
            trends.append({'N': n, 'num_warps': w, 'grid': grid, 'components': components, 'sum_of_median_slopes_us_per_iteration': sums,
                'full_kernel_times_us': full_times, 'adjacent_layout_differences': pairs,
                'sum_spread_percent_of_min': 100 * (max(sums.values()) / min(sums.values()) - 1),
                'full_spread_percent_of_min': 100 * (max(full_times.values()) / min(full_times.values()) - 1),
                'vec2_to_vec4_to_vec8_monotone_sum_increase': sums[2] < sums[4] < sums[8],
                'sum_is_validated_end_to_end_model': False,
                'limitations': ['ML and MR have different resource/residency and auxiliary work',
                    'MR private recurrence changes widening/packed-max optimization in most layouts',
                    'output layout conversion is absent; TMA/setup are excluded by the slope',
                    'Stage B complete-kernel measurements are prior workers and a different grid; compare trends only']})
    return {'stage': 'C1', 'status': 'COMPLETE', 'scope': 'H100 feasibility only; no C2/C3/D or compiler selector',
        'logical_programs': 16, 'layouts': 8, 'formal_event_samples': 7680, 'worker_invocations': 3,
        'GPU_UUIDs': [w['raw']['environment']['gpu_uuid'] for w in worker], 'grids': grids,
        'rows': rows, 'trends': trends, 'templates': templates(),
        'NCU_reports': counters['report_count'], 'strict_K_counter_gate': counters['status'],
        'native_rebuild': False, 'score_model_used': False, 'PR_created': False,
        'uncertainty': 'three invocation slope ranges, descriptive; no independent-device or unique-winner claim'}


def render(result):
    lines = ['# Phase 11 C1 — H100 ML/MR feasibility', '',
        '8 个 N-first 布局、16 个程序；runtime K=1/16/64/256，共用冻结 CUBIN。',
        'H100，固定 upstream 20d9，host-created descriptor，BF16 输入。未重编译 Triton native。',
        '192 项编译阶段完整输出检查、3 次计时 worker、7,680 个单 kernel event 样本、128 份分离的 NCU 报告。',
        '全部原始数据、compiler artifacts、NCU report 与技术失败保留；未启动 C2/C3/D 或 PR。', '',
        '[组件斜率图](plots/component_slopes.png)、[与完整 kernel 的趋势对照](plots/trend_comparison.png)、[ML 时间曲线](plots/ML_time_curves.png)；同目录有可导出的 PDF。', '',
        '## 增量斜率', '',
        '每个 worker 对四个 K 的 20 样本中位数拟合 T=a+bK；下表为三次 worker 斜率的中位数。',
        '单位 ns/iteration，表示相应 grid 完整 fixture 的增量；包含 checksum/私有步进/转换/累加/循环。',
        '逐次斜率区间、相邻 ΔT/ΔK、残差与每个 K 的时间见 analysis.json。', '']
    for grid in result['grids']:
        lines += [f'### grid={grid}', '', '| 配置 | vec | ML | MR | ML+MR |', '|---|---:|---:|---:|---:|']
        for trend in result['trends']:
            if trend['grid'] != grid:
                continue
            for vec, values in trend['components'].items():
                lines.append(f"| 32×{trend['N']}/w{trend['num_warps']} | {vec} | {values['ML']*1000:.3f} | {values['MR']*1000:.3f} | {trend['sum_of_median_slopes_us_per_iteration'][vec]*1000:.3f} |")
        lines += ['']
    lines += ['### 相邻 K 区间的增量斜率', '',
        '下面每个值先在 worker 内计算 ΔT/ΔK，再取三个 worker 的中位数，单位 ns/iteration。',
        '这些区间斜率保留非线性；四点 OLS 只是此 K 区间的描述性汇总。', '']
    for grid in result['grids']:
        lines += [f'grid={grid}', '', '| 配置 | vec | ML 1→16 | ML 16→64 | ML 64→256 | MR 1→16 | MR 16→64 | MR 64→256 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for n,w in ((64,8),(128,4)):
            for vec in (1,2,4,8):
                a,b = (next(r for r in result['rows'] if r['N']==n and r['vec']==vec and r['family']==f and r['grid']==grid) for f in ('ML','MR'))
                values = a['median_adjacent_slopes_us_per_iteration'] + b['median_adjacent_slopes_us_per_iteration']
                lines.append(f'| 32×{n}/w{w} | {vec} | ' + ' | '.join(f'{x*1000:.3f}' for x in values) + ' |')
        lines += ['']
    lines += ['## NCU：实际动态执行', '',
        '保留每个 PC 的原始计数，统一为 function offset；按同一逻辑 opcode 的展开/尾部副本合计，严格核对 K=1/16/64/256。',
        '编译器把部分循环展开四份，K=1 走尾部；单个展开副本的 PC 不会单独执行 K 次。该解释修订在正式 event 计时前冻结为 revision 4。',
        '初始化和最终写出的 K 不变 PC 单独列出，机器循环范围内不能混入非零 K 不变的目标 PC。',
        '下表是每 CTA、每次 body 的动态 warp 指令数。Wavefront 为硬件计数的 (K256−K1)/255，含固定项的原值全部保留。',
        '指令数是 NCU software/source 的 warp 计数，不等于线程访问数或硬件内存 requests；profiler duration 不进入 event 统计。',
        '宽访问可能天然需要多个 wavefront，不能把全部 wavefront 当 bank conflict。', '',
        '| 配置 | vec | ML LDS | ML load WF | MR LDS | MR STS | MR SHFL | MR BAR.SYNC | MR load WF |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    by_tag = {row['tag']: row for row in result['rows'] if row['grid'] == 1}
    for n, w in ((64, 8), (128, 4)):
        for vec in (1, 2, 4, 8):
            a, b = (by_tag[f'M32_N{n}_w{w}:vec{vec}:{family}']['counters'] for family in ('ML', 'MR'))
            wf = 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'
            af, bf = a['families'], b['families']
            count = lambda name: bf[name]['loop_per_iteration_warp_instructions']
            lines.append(f"| 32×{n}/w{w} | {vec} | {af['shared_load']['loop_per_iteration_warp_instructions']} | {a['hardware_delta_per_iteration'][wf]:.3f} | {count('shared_load')} | {count('shared_store')} | {count('shuffle')} | {count('barrier')} | {b['hardware_delta_per_iteration'][wf]:.3f} |")
    lines += ['', '## ML+MR 与完整 kernel 的趋势对照', '',
        '完整 kernel 只引用阶段 B 的 H100/host/BF16 数据。不同 grid、不同 worker 的绝对时间不直接相加或相除。',
        '下面对照各自条件内相邻布局的相对变化；它检验加法假设，不拟合权重。', '',
        '| 配置 / grid | 布局变化 | ΔML ns | ΔMR ns | 和的变化 | 原完整 kernel 变化 |',
        '|---|---|---:|---:|---:|---:|']
    for trend in result['trends']:
        for row in trend['adjacent_layout_differences']:
            if row['from_vec'] == 1:
                continue
            lines.append(f"| 32×{trend['N']}/w{trend['num_warps']} / {trend['grid']} | vec{row['from_vec']}→vec{row['to_vec']} | {row['delta_ML_us_per_iteration']*1000:.3f} | {row['delta_MR_us_per_iteration']*1000:.3f} | {row['sum_relative_change_percent']:+.2f}% | {row['full_kernel_relative_change_percent']:+.2f}% |")
    mismatch = sum(row['precision_template_mismatch'] for row in result['templates'].values())
    lines += ['', '观察结论：32×64 的 vec2→4→8 中 MR 增长、ML 增量较小，组件和能够复现退化方向，但幅度与完整 kernel 不一致。',
        '32×128/grid8448 的组件和跨度约 23.2%，完整 kernel 约 2.23%；当前两项相加不能解释它的平坦表现。',
        '', '## 解释边界与阶段停点', '',
        f'MR 中 {mismatch}/8 个布局的线程内 max 精度模板与完整 kernel 不同：私有更新复用 FP32 值，使 BF16 packed max 优化失效。',
        '这次 MR 可提供实际通信子图与包含私有步进的有效斜率；线程内 max/widening 的纯参数尚未分离。',
        'ML 的 T(K) 在小 K 区间并非线性：例如 32×128/vec8/grid8448 的 T1/T16/T64/T256 为约 113.17/101.89/248.21/942.38 µs。',
        '因此 OLS 斜率不是在全部 K 上成立的固定单次加载成本。ML 最后写完整输入 tile，输出字节量大于原 max kernel；这项固定工作与 body 的吞吐/重叠关系仍需辨识。',
        '原完整 kernel 的输出布局转换也未包含在 MR 中。ML 双缓冲与 MR 额外活跃值改变资源；每个 artifact 的寄存器/SMEM/可驻留 CTA 数保留。',
        '因此 ML+MR 目前是待检验的组件趋势，尚不是可部署的 layout cost model；报告不以其绝对和预测端到端时间。',
        '后续建议先修正 MR 的 packed-max 模板/私有更新扰动，以及 ML checksum 输出与资源边界，再讨论正式校准；本次不执行这些新变体。',
        '停在 C1，等待用户审阅后决定是否修订探针或进入 C2。', '',
        '## 证据与复现', '',
        '- `preregistration/`：执行前冻结协议、逐次技术修订、worker source snapshot、旧草案快照与历史证据 manifest。',
        '- `artifacts/`：初次 gl.range API 失败及成功的 16 个 artifacts、完整输出检查、资源和缓存记录。',
        '- `profiles/`：初次工具容器缺 NumPy 的失败，以及 128 份原始 report、collect/import/source 与 CUBIN 绑定。',
        '- `timing/`：三次独立 worker 的全部 event 样本、预热正确性、参数与 Driver UUID。',
        '- `artifact_gate.json`、`counter_gate.json`、`analysis.json`、`validation.json`：派生核验与完整分析。',
        '', '```sh',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c1.analyze artifacts --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c1.analyze counters --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c1.report --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c1.validate --committed', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    c.protect_prior()
    result = derive()
    if args.validate:
        assert c.read(c.OUT / 'analysis.json') == __import__('json').loads(c.encode(result))
        assert (c.OUT / 'summary.md').read_text() == render(result)
    else:
        (c.OUT / 'analysis.json').write_bytes(c.encode(result))
        (c.OUT / 'summary.md').write_text(render(result))
    print('C1 analysis PASS: 8 layouts, 16 programs, 7,680 event samples, 128 NCU reports')


if __name__ == '__main__':
    main()
