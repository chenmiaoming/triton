"""C1.5 steady-range slopes, K1 references and template/counter evidence."""
import argparse
from collections import defaultdict
import json
import math
import statistics as st

from experiments.tma_reduction_layout.phase11.c1.report import regression
from experiments.tma_reduction_layout.phase11.c15 import common as c


def dataset(root, protocol, metadata):
    fits = defaultdict(list)
    uuids, sm_counts = [], set()
    for invocation in (1, 2, 3):
        raw = c.read(root / 'timing' / f'invocation_{invocation}' / 'raw.json')
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert raw['payload_SHA256'] == c.sha((root / 'frozen_payload.zip').read_bytes())
        assert raw['compilation'] == {'Triton_imported': False, 'subprocess_calls': 0, 'compile_calls': 0}
        assert raw['CUDA_return_codes'] == [0]
        assert len(raw['checks']) == 128 and all(r['correctness']['pass'] for r in raw['checks'])
        uuids.append(raw['environment']['gpu_uuid'])
        sm_counts.add(raw['environment']['SM_count'])
        samples = defaultdict(list)
        for row in raw['samples']:
            assert row['CUBIN_SHA256'] == metadata[row['tag']]['cubin_SHA256']
            assert row['parameters'][8] == row['K']
            assert len(row['samples_us']) == 5 and all(math.isfinite(x) and x > 0 for x in row['samples_us'])
            samples[row['tag'], row['grid'], row['K']].extend(row['samples_us'])
        assert len(samples) == 128 and all(len(v) == 20 for v in samples.values())
        for tag in metadata:
            for grid in (1, 64 * raw['environment']['SM_count']):
                medians = {k: st.median(samples[tag, grid, k]) for k in (1, 16, 64, 256)}
                fit = regression([16, 64, 256], [medians[k] for k in (16, 64, 256)])
                fit.update({'invocation': invocation, 'medians_us_by_K': medians,
                    'fit_K': [16, 64, 256], 'T1_us': medians[1],
                    'T1_minus_one_slope_us': medians[1] - fit['slope_us_per_iteration'],
                    'adjacent_slopes_us_per_iteration': [(medians[64]-medians[16])/48,
                                                         (medians[256]-medians[64])/192]})
                fits[tag, grid].append(fit)
    assert len(sm_counts) == 1
    return fits, uuids, (1, 64 * next(iter(sm_counts)))


def summarize(fits):
    slopes = [f['slope_us_per_iteration'] for f in fits]
    adjacent = [st.median(f['adjacent_slopes_us_per_iteration'][i] for f in fits) for i in (0, 1)]
    return {'worker_fits': fits, 'median_slope_us_per_iteration': st.median(slopes),
        'slope_range_us_per_iteration': [min(slopes), max(slopes)],
        'median_T1_us': st.median(f['T1_us'] for f in fits),
        'median_T1_minus_one_slope_us': st.median(f['T1_minus_one_slope_us'] for f in fits),
        'median_steady_fit_intercept_us': st.median(f['intercept_us'] for f in fits),
        'median_adjacent_slopes_us_per_iteration': adjacent,
        'adjacent_slope_relative_gap': abs(adjacent[1]-adjacent[0])/max(abs(adjacent[0]),abs(adjacent[1]),1e-30),
        'median_times_us_by_K': {k: st.median(f['medians_us_by_K'][k] for f in fits) for k in (1,16,64,256)}}


def derive():
    protocol = c.load_protocol()
    gate = c.read(c.OUT / 'artifact_gate.json')
    counter = c.read(c.OUT / 'counter_gate.json')
    assert counter['status'] == 'OK'
    now, uuids, grids = dataset(c.OUT, protocol, gate['metadata'])
    prior_meta = c.read(c.prior.OUT / 'artifact_gate.json')['metadata']
    before, before_uuids, before_grids = dataset(c.prior.OUT, c.prior.load_protocol(), prior_meta)
    assert grids == before_grids
    prior_resources = c.read(c.prior.OUT / 'artifact_gate.json')['resources']
    rows = []
    for tag, meta in gate['metadata'].items():
        for grid in grids:
            rows.append({'tag': tag, 'N': meta['N'], 'num_warps': meta['num_warps'], 'vec': meta['vec'],
                'family': meta['family'], 'grid': grid, **summarize(now[tag, grid]),
                'C1_refit_same_K': summarize(before[tag, grid]),
                'resources': gate['resources'][tag], 'C1_resources': prior_resources[tag],
                'counter': counter['binaries'][tag]})
    trends = []
    full = c.read(c.base.OUT / 'stage_b/analysis.json')
    for n,w in ((64,8),(128,4)):
        original = next(r for r in full['layout_results'] if r['case_id']==f'M32_N{n}_w{w}:bfloat16:max:host' and r['fraction']==1)
        actual = {v:original['median_of_worker_medians_us'][f'M32_N{n}_w{w}:bfloat16:max:host:fixed_vec{v}'] for v in (1,2,4,8)}
        for grid in grids:
            sums, prior_sums = {}, {}
            for vec in (1,2,4,8):
                components = [next(r for r in rows if r['N']==n and r['vec']==vec and r['grid']==grid and r['family']==f) for f in ('ML','MR')]
                sums[vec] = sum(r['median_slope_us_per_iteration'] for r in components)
                prior_sums[vec] = sum(r['C1_refit_same_K']['median_slope_us_per_iteration'] for r in components)
            trend = {'N':n,'num_warps':w,'grid':grid,'component_sum_us_per_iteration':sums,
                     'C1_refit_component_sum_us_per_iteration':prior_sums,'StageB_full_times_us':actual,
                     'component_span_percent':100*(max(sums.values())/min(sums.values())-1),
                     'full_span_percent':100*(max(actual.values())/min(actual.values())-1)}
            if grid > 1:
                factor = (8192//32)*(8192//n)/grid
                scaled = {v:factor*sums[v] for v in sums}
                predicted = {v:45+scaled[v] for v in sums}
                trend['full_grid_hypothesis'] = {'factor':factor, 'component_scaled_us':scaled,
                    'assumed_HBM_base_us':45,'predicted_full_us':predicted,
                    'delta_vec8_minus_vec1_us':scaled[8]-scaled[1],
                    'predicted_time_reduction_percent':100*(predicted[8]-predicted[1])/predicted[8],
                    'observed_time_reduction_percent':100*(actual[8]-actual[1])/actual[8],
                    'assumptions_measured':False}
            trends.append(trend)
    return {'stage':'C1.5','status':'COMPLETE','scope':'Eight layouts; BF16 MR and steady-K slope correction only; stop before C2',
        'new_MR_programs':8,'reused_ML_CUBINs':8,'new_MR_full_output_checks':96,'reused_ML_full_output_checks':96,
        'new_event_samples':7680,'worker_invocations':3,'new_MR_NCU_reports':64,'reused_ML_NCU_reports':64,
        'GPU_UUIDs':uuids,'C1_GPU_UUIDs':before_uuids,'grids':grids,'fit_K':[16,64,256],
        'rows':rows,'trends':trends,'MR_template_audit':gate['MR_template_audit'],
        'C1_full_grid_arithmetic_audit':c.read(c.OUT/'preregistration/C1_full_grid_hypothesis_audit.json'),
        'native_rebuild':False,'PR_created':False,'C2_started':False,
        'pure_HBM_latency_measured':False,'slope_excludes_K1':True,
        'steady_range_does_not_imply_linearity':True}


def render(result):
    lines = ['# Phase 11 C1.5 — BF16 MR / steady-range slope correction','',
        '范围：H100，固定 upstream 20d9 和 host-created descriptor，当前两种几何的 N-first vec1/2/4/8。',
        '新编译 8 个 MR，ML 的 8 个 CUBIN 与 C1 逐字节相同；native/ccache 未变。',
        '96 项新 MR 完整输出检查；96 项 ML 检查复用 C1。新一轮三个 worker 共 7,680 个单 kernel event 样本。',
        '64 份新 MR NCU 报告，64 份原 ML 报告复用并按 CUBIN SHA 绑定。全部原始证据与 C1 保持不变。','',
        '## BF16 模板验证','',
        '每轮先用 BF16 状态、BF16 step/threshold/wrap 做私有更新，再 single-use widening 直接供给 max，归约结果仅寄存器累加。',
        'GLIR 核对完整输入 widening 的唯一消费者；PTX 的 max opcode 种类和静态数量逐个与阶段 B 原完整 kernel 比较。','',
        '| 配置 | vec | C1.5 max PTX 静态站点 | 与原完整 kernel 一致 |', '|---|---:|---|---|']
    for tag,row in sorted(result['MR_template_audit'].items()):
        geo,vec,_=tag.split(':')
        ops=', '.join(f'{op} ×{count}' for op,count in row['current_max_PTX_sites'].items())
        lines.append(f'| {geo} | {vec} | {ops} | 是 |')
    lines += ['', '7 个需要线程内 BF16 combine 的布局恢复 max.bf16/max.bf16x2；32×64/vec8 的原始模板没有线程内 BF16 combine，保持 max.f32 通信树。',
        '静态站点不等于硬件执行次数；NCU 的动态 opcode 组仍严格随 K 增长，保留展开/尾部的逐 PC 原值。','',
        '## 斜率：统一使用 K=16/64/256','',
        '每个 worker 先取每个 K 的 20 样本中位数，对三个点拟合 T=a+bK；下表为三个 worker 的斜率中位数。',
        '单位 ns/iteration，表示相应 grid 的完整 fixture 增量，仍包含私有更新、checksum、累加和循环工作。','']
    for grid in result['grids']:
        lines += [f'### grid={grid}','', '| 配置 | vec | ML | MR | ML+MR | C1 同区间重拟合的和 |','|---|---:|---:|---:|---:|---:|']
        for t in result['trends']:
            if t['grid']!=grid: continue
            for v in (1,2,4,8):
                a,b=(next(r for r in result['rows'] if r['N']==t['N'] and r['vec']==v and r['grid']==grid and r['family']==f) for f in ('ML','MR'))
                lines.append(f"| 32×{t['N']}/w{t['num_warps']} | {v} | {a['median_slope_us_per_iteration']*1000:.3f} | {b['median_slope_us_per_iteration']*1000:.3f} | {t['component_sum_us_per_iteration'][v]*1000:.3f} | {t['C1_refit_component_sum_us_per_iteration'][v]*1000:.3f} |")
        lines += ['']
    lines += ['## K1 固定开销参考与区间一致性','',
        'K=1 不参加斜率拟合。T1 包含一次 body；T1−b 是固定开销参考，另列稳态三点拟合的截距 a。',
        '这些值都包含 fixture 的 TMA 初始化、最终输出及吞吐/重叠影响，尚未单独测量纯 HBM/TMA 延迟。','',
        '| 配置 | vec | ML b16→64 / b64→256 ns | MR b16→64 / b64→256 ns | ML T1 / T1−b / a µs | MR T1 / T1−b / a µs |',
        '|---|---:|---:|---:|---:|---:|']
    grid=max(result['grids'])
    for n,w in ((64,8),(128,4)):
        for v in (1,2,4,8):
            a,b=(next(r for r in result['rows'] if r['N']==n and r['vec']==v and r['grid']==grid and r['family']==f) for f in ('ML','MR'))
            fmt=lambda r:' / '.join(f'{x*1000:.2f}' for x in r['median_adjacent_slopes_us_per_iteration'])
            fixed=lambda r:' / '.join(f'{r[k]:.3f}' for k in ('median_T1_us','median_T1_minus_one_slope_us','median_steady_fit_intercept_us'))
            lines.append(f'| 32×{n}/w{w} | {v} | {fmt(a)} | {fmt(b)} | {fixed(a)} | {fixed(b)} |')
    lines += ['', '保留两段增量斜率、三个 worker 的范围和拟合残差；排除 K1 是统一的分析修正，不自动证明 K>=16 已完全进入稳态。',
        '', '## NCU 与资源','', '| 配置 | vec | MR load / store / shuffle / barrier（动态 warp 指令/轮/CTA） | MR load WF/轮/CTA | MR 寄存器 C1→C1.5 |',
        '|---|---:|---:|---:|---:|']
    for n,w in ((64,8),(128,4)):
        for v in (1,2,4,8):
            r=next(r for r in result['rows'] if r['N']==n and r['vec']==v and r['grid']==1 and r['family']=='MR')
            q=r['counter']; f=q['families']
            counts=' / '.join(str(f[k]['loop_per_iteration_warp_instructions']) for k in ('shared_load','shared_store','shuffle','barrier'))
            wf=q['hardware_delta_per_iteration']['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']
            lines.append(f"| 32×{n}/w{w} | {v} | {counts} | {wf:.3f} | {r['C1_resources']['num_regs']}→{r['resources']['num_regs']} |")
    lines += ['', 'MR shared scratch 来自规约 lowering，无结果反馈到输入的额外广播。所有程序无 spill。',
        'BF16 SASS 实际包含 HMNMX2 和 VHMNMX；计数解析覆盖两者，原始 source/report 的所有 PC 原值均保留。',
        '例如 32×128/vec1 在 PTX 有 31 个 max.bf16 静态站点，NCU body 为 24 个 HMNMX2 和 40 个 VHMNMX 动态 warp 指令/轮/CTA；两种层级的数值不直接相等。',
        '修正同时改变私有更新精度、widening、寄存器活跃值和可驻留 CTA 数；这些成本仍应在后续模型中明确处理。','',
        '## 全图缩放与 45 µs 底座假设的算术核对','',
        '用户提出的 CTA 倍率正确：32768/8448=3.8788，16384/8448=1.9394。',
        '以旧 C1 四点 OLS 为输入，线性放大后的 vec8−vec1 差值为 58.4568 和 3.9604 µs。',
        '但加上假设的 45 µs 底座后，32×128 为 66.7904/70.7509 µs，时间缩短为 5.5977%，并非小于 3%；阶段 B 实测差值为 1.2480 µs。',
        '下面对新的 C1.5 稳态区间斜率做相同算术；CTA 线性缩放和固定底座的可加性仍是待验证假设。','',
        '| 配置 | 放大后的 vec8−vec1 µs | +45 后 vec1 / vec8 µs | 预测时间缩短 | 阶段 B 实测时间缩短 |',
        '|---|---:|---:|---:|---:|']
    for t in result['trends']:
        if t['grid']!=grid: continue
        h=t['full_grid_hypothesis']; p=h['predicted_full_us']
        lines.append(f"| 32×{t['N']}/w{t['num_warps']} | {h['delta_vec8_minus_vec1_us']:.3f} | {p[1]:.3f} / {p[8]:.3f} | {h['predicted_time_reduction_percent']:.2f}% | {h['observed_time_reduction_percent']:.2f}% |")
    lines += ['', '时间缩短以 vec8 为分母；slowdown 以 vec1 为分母，二者不混用。45 µs 没有被当作实测参数或为拟合而调整。',
        '', '## 证据与停点','',
        '`preregistration/` 保存冻结协议、source snapshot、ML 复用映射和运行前的算术核对；',
        '`artifacts/`、`profiles/`、`timing/` 保存新原始数据，C1 的原始证据不修改。',
        '本阶段只完成 C1.5 修正与验证，停在 C2 之前。完整残差、K1 参考与原 C1 同区间重拟合在 analysis.json。','',
        '```sh',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c15.analyze artifacts --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c15.analyze counters --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c15.report --validate',
        '.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.c15.validate --committed','```','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    c.protect_prior()
    result=derive()
    if args.validate:
        assert c.read(c.OUT/'analysis.json')==json.loads(c.encode(result))
        assert (c.OUT/'summary.md').read_text()==render(result)
    else:
        (c.OUT/'analysis.json').write_bytes(c.encode(result))
        (c.OUT/'summary.md').write_text(render(result))
    print('C1.5 analysis PASS: packed-max restored; slopes fit K16/64/256 only')


if __name__=='__main__':
    main()
