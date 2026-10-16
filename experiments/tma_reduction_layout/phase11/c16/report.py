"""Frozen predictions, independent transfer controls and held-out checks."""
import argparse
from collections import defaultdict
import json
import math
import statistics as st
from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.c1.report import regression


def submission_analysis():
    protocol=c.read(c.OUT/'preregistration/submission_protocol.json')
    raw=c.read(c.OUT/'submission/attempt_1/raw.json')
    assert raw['status']=='OK' and len(raw['checks'])==20 and all(x['pass'] for x in raw['checks'])
    assert raw['diagnostic_SHA256']==c.sha(c.encode(protocol))
    assert raw['payload_SHA256']==c.sha((c.OUT/'frozen_payload.zip').read_bytes())
    assert raw['Triton_imported'] is False and raw['subprocess_calls']==0
    assert raw['CUDA_return_codes']==[0] and len(raw['graphs'])==10
    assert raw['CUDA_call_counts']['cuGraphLaunch']==650
    assert raw['CUDA_call_counts']['cuLaunchKernel']==680
    for graph in raw['graphs'].values():
        assert sorted(graph['node_types'])==[0,7,7]
        assert graph['kernel_nodes']==1 and graph['external_event_record_nodes']==2
    cells=defaultdict(list)
    for row in raw['samples']:
        assert len(row['samples_us'])==5
        assert row['CUBIN_SHA256']==raw['graphs'][row['tag']]['CUBIN_SHA256']
        cells[row['tag'],row['cache'],row['method']].extend(row['samples_us'])
    assert len(cells)==40 and all(len(x)==20 for x in cells.values())
    rows=[]
    for tag in sorted(raw['graphs']):
        for mode in ('warm','cold'):
            direct=st.median(cells[tag,mode,'direct'])
            graph=st.median(cells[tag,mode,'graph'])
            rows.append({'tag':tag,'cache':mode,'direct_us':direct,'graph_us':graph,
                         'direct_minus_graph_us':direct-graph,
                         'direct_range_us':[min(cells[tag,mode,'direct']),max(cells[tag,mode,'direct'])],
                         'graph_range_us':[min(cells[tag,mode,'graph']),max(cells[tag,mode,'graph'])]})
    return {'event_samples':800,'worker_count':1,'environment':raw['environment'],'rows':rows,
            'graph_node_topologies_verified':10,'one_kernel_per_graph':True,
            'CUDA_call_counts':raw['CUDA_call_counts'],
            'no_model_refit':True,'primary_measurement_method_unchanged':True}


def derive():
    protocol = c.load_protocol()
    gate = c.read(c.OUT / 'artifact_gate.json')
    counters = c.read(c.OUT / 'counter_gate.json')
    assert counters['status'] == 'OK'
    workers, bins, bindings = [], defaultdict(list), []
    for invocation in (1, 2, 3):
        raw = c.read(c.OUT / 'timing' / f'invocation_{invocation}' / 'raw.json')
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
        assert raw['payload_SHA256'] == c.sha((c.OUT / 'frozen_payload.zip').read_bytes())
        assert raw['compilation'] == {'Triton_imported': False, 'subprocess_calls': 0}
        assert raw['CUDA_return_codes'] == [0]
        assert len(raw['checks']) == 78 and all(x['pass'] for x in raw['checks'])
        samples = defaultdict(list)
        for row in raw['samples']:
            assert row['CUBIN_SHA256'] == gate['metadata'][row['tag']]['cubin_SHA256']
            assert len(row['samples_us']) == 5
            assert all(math.isfinite(x) and x > 0 for x in row['samples_us'])
            samples[row['tag'], row['grid'], row['K'], row['cache']].extend(row['samples_us'])
        assert len(samples) == 108 and all(len(v) == 20 for v in samples.values())
        medians = {key: st.median(value) for key, value in samples.items()}
        for key, value in medians.items():
            bins[key].append(value)
        slopes, costs = {}, {}
        for n, w in ((64, 8), (128, 4)):
            for vec in (1, 2, 4, 8):
                sums = 0
                for family in ('ML', 'MR'):
                    tag = f'M32_N{n}_w{w}:vec{vec}:{family}'
                    fit = regression(protocol['K'], [medians[tag, 8448, k, 'warm'] for k in protocol['K']])
                    slopes[tag] = fit
                    sums += fit['slope_us_per_iteration']
                costs[n, vec] = sums * (256 * (8192 // n) / 8448)
        secondary = []
        for mode in ('warm', 'cold'):
            cal = medians['M32_N64_w8:vec4:FULL', 32768, 0, mode]
            offset = cal - costs[64, 4]
            for n, w in ((64, 8), (128, 4)):
                grid = 256 * (8192 // n)
                control = medians[f'M32_N{n}_w{w}:control:TMA', grid, 0, mode]
                for vec in (1, 2, 4, 8):
                    actual = medians[f'M32_N{n}_w{w}:vec{vec}:FULL', grid, 0, mode]
                    predicted = max(control, offset + costs[n, vec])
                    secondary.append({'N': n, 'vec': vec, 'cache': mode, 'control_us': control,
                        'calibrated_offset_us': offset, 'same_worker_C_onchip_us': costs[n, vec],
                        'predicted_us': predicted, 'actual_us': actual,
                        'error_percent': 100 * (predicted - actual) / actual,
                        'held_out': not (n == 64 and vec == 4)})
        workers.append({'invocation': invocation, 'environment': raw['environment'],
                        'micro_fits': slopes, 'secondary_predictions': secondary})
        for tag, binding in raw['binaries'].items():
            assert binding['CUBIN_SHA256'] == binding['loaded_CUBIN_SHA256'] == gate['metadata'][tag]['cubin_SHA256']
            assert binding['sha_guards'] == binding['launches'] > 0
        bindings.append({'invocation': invocation,
                         'launches': sum(x['launches'] for x in raw['binaries'].values())})
    rows = [{'tag': key[0], 'grid': key[1], 'K': key[2], 'cache': key[3],
             'median_us': st.median(value), 'worker_medians_us': value,
             'worker_range_us': [min(value), max(value)]} for key, value in sorted(bins.items())]
    primary, controls = [], []
    for frozen in protocol['analysis']['frozen_predictions']:
        n, vec = frozen['N'], frozen['vec']
        w, grid = (8 if n == 64 else 4), 256 * (8192 // n)
        for mode in ('warm', 'cold'):
            row = next(x for x in rows if x['tag'] == f'M32_N{n}_w{w}:vec{vec}:FULL' and x['grid'] == grid and x['cache'] == mode)
            actual = row['median_us']
            primary.append({**frozen, 'cache': mode, 'actual_us': actual,
                            'error_us': frozen['predicted_us'] - actual,
                            'error_percent': 100 * (frozen['predicted_us'] - actual) / actual,
                            'worker_medians_us': row['worker_medians_us']})
    for n, w in ((64, 8), (128, 4)):
        for divisor in (4, 2, 1):
            grid = 256 * (8192 // n) // divisor
            for mode in ('warm', 'cold'):
                control = next(x for x in rows if x['tag'] == f'M32_N{n}_w{w}:control:TMA' and x['grid'] == grid and x['cache'] == mode)
                for vec in (1, 2, 4, 8):
                    full = next(x for x in rows if x['tag'] == f'M32_N{n}_w{w}:vec{vec}:FULL' and x['grid'] == grid and x['cache'] == mode)
                    controls.append({'N': n, 'vec': vec, 'divisor': divisor, 'cache': mode,
                                     'control_us': control['median_us'], 'full_us': full['median_us'],
                                     'full_minus_control_us': full['median_us'] - control['median_us']})
    return {'stage': 'C1.6', 'status': 'COMPLETE', 'event_samples': 6480,
        'submission_diagnostic':submission_analysis(), 'total_event_samples_including_diagnostic':7280,
        'raw_worker_checks': 234, 'new_control_correctness_checks': 4,
        'new_programs': 2, 'exact_reused_programs': 24, 'profiles': 20,
        'protocol_SHA256': c.sha(c.encode(protocol)), 'workers': workers, 'cells': rows,
        'primary_frozen_predictions': primary, 'transfer_controls': controls,
        'counter_rows': counters['rows'], 'launches': bindings,
        'primary_warm_max_abs_error_percent': max(abs(r['error_percent']) for r in primary if r['cache'] == 'warm'),
        'primary_cold_max_abs_error_percent': max(abs(r['error_percent']) for r in primary if r['cache'] == 'cold'),
        'secondary_warm_heldout_abs_error_percent': [abs(r['error_percent']) for w in workers
            for r in w['secondary_predictions'] if r['held_out'] and r['cache'] == 'warm'],
        'offset_physically_isolated': False, 'native_rebuild': False, 'C2_started': False, 'PR_created': False}


def render(result):
    lines = ['# C1.6 — overlap/HBM validation', '',
        '原 descriptor、pinned upstream/native；24 个原 CUBIN 逐字节复用，只编译两个 TMA 搬运对照。',
        '三个新 worker，同轮测完整 kernel、对照与 K=16/64/256 微基准；6480 个单 kernel event 样本。',
        '20 份新 NCU 硬件报告采用 application replay，每次重建 warm/cold 状态；其 duration 不进入正式计时。',
        '原始数据、失败编译、源码修订和 compiler artifacts 均保留。旧证据不修改。', '',
        '## 冻结参数的复现检查', '',
        '`max(platform,26.6+C_onchip)`：沿用先前 C1.5 斜率，platform=61.4/56.1 us，未按新结果调整。',
        '这些布局已被先前观察过；这是新 cohort 的复现检查，不是 unseen-shape 泛化。单位 us。', '',
        '| 配置 | vec | 冻结预测 | 新 warm | warm 误差 % | 新 cold | cold 误差 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,w in ((64,8),(128,4)):
        for v in (1,2,4,8):
            a,b=(next(x for x in result['primary_frozen_predictions'] if x['N']==n and x['vec']==v and x['cache']==mode) for mode in ('warm','cold'))
            lines.append(f"| 32×{n}/w{w} | {v} | {a['predicted_us']:.3f} | {a['actual_us']:.3f} | {a['error_percent']:+.2f} | {b['actual_us']:.3f} | {b['error_percent']:+.2f} |")
    lines += ['', '## 独立 TMA 搬运对照', '',
        '搬运原完整 tile，等待完成，仅从原 shared descriptor gather 首行并输出 BN 个 FP32。',
        '逻辑输入/输出字节与完整 kernel 相同，寄存器和驻留可能变化；它不是纯 HBM latency。', '',
        '| 配置 | vec | warm 对照 / 完整 us | cold 对照 / 完整 us |', '|---|---:|---:|---:|']
    for n,w in ((64,8),(128,4)):
        for v in (1,2,4,8):
            a,b=(next(x for x in result['transfer_controls'] if x['N']==n and x['vec']==v and x['divisor']==1 and x['cache']==mode) for mode in ('warm','cold'))
            lines.append(f"| 32×{n}/w{w} | {v} | {a['control_us']:.3f} / {a['full_us']:.3f} | {b['control_us']:.3f} / {b['full_us']:.3f} |")
    lines += ['', '## NCU DRAM / L2 证据', '',
        'NCU 在独立 profiling worker 上测量，以下为真实 counter 值，不由 event 时间推算。',
        'DRAM bytes 是观测流量，逻辑数据量为输入 128 MiB、输出 8 MiB；写回可受 cache 状态影响。', '',
        '| 配置 / 类型 | cache | DRAM 读 MiB | DRAM 写 MiB | DRAM sustained-peak % | L2 hit % | SM % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result['counter_rows']:
        m=r['metrics']
        lines.append(f"| {r['tag']} | {r['cache']} | {m['dram__bytes_read.sum']/2**20:.3f} | {m['dram__bytes_write.sum']/2**20:.3f} | {m['dram__throughput.avg.pct_of_peak_sustained_elapsed']:.2f} | {m['lts__t_sector_hit_rate.pct']:.2f} | {m['sm__throughput.avg.pct_of_peak_sustained_elapsed']:.2f} |")
    lines += ['', 'warm/cold 的实际 DRAM 读量均约 128 MiB，而 cold 写量从约 10.4–10.9 MiB 增加到 27.1–27.9 MiB。',
        '因此清缓存干预改变了计数窗口内写回流量，不能称为仅改变目标输入读命中率；不将该写入量全归因于 kernel 的逻辑输出。',
        '', '## 单 kernel Graph 提交间隙校验', '',
        '另一个 worker 对相同 10 个二进制进行 800 个配对样本。每个 Graph 经 API 核对：一个 kernel 节点、两个外部 event record 节点。',
        '事件在 Graph 内部记录；无批量除次数，不改变主实验的计时方法或模型参数。单位 us。', '',
        '| 二进制 | cache | direct event | Graph 内 event | direct−Graph |', '|---|---|---:|---:|---:|']
    for row in result['submission_diagnostic']['rows']:
        lines.append(f"| {row['tag']} | {row['cache']} | {row['direct_us']:.3f} | {row['graph_us']:.3f} | {row['direct_minus_graph_us']:.3f} |")
    lines += ['', '配对差异包含主机提交间隙和 Graph 调度差异，不能将其全部解释成 CPU 指令耗时。',
        '其量级可用于检查 26.6 us 固定项是否主要来自直接 event 的提交方式。',
        '', '## 独立平台 + 单点校准后的留出检查', '',
        '每个 worker/cache 模式分别用 N64 vec4 一个点校准 offset；C_onchip 来自同 worker 微基准，',
        'platform 来自 TMA 对照。其余 7 个完整 kernel 点均不参与校准。offset 是经验残差，不能解释为独立硬件延迟。', '',
        '| worker | cache | 校准 offset us | 7 个留出点最大绝对误差 % |', '|---:|---|---:|---:|']
    for w in result['workers']:
        for mode in ('warm','cold'):
            rows=[r for r in w['secondary_predictions'] if r['cache']==mode]
            lines.append(f"| {w['invocation']} | {mode} | {rows[0]['calibrated_offset_us']:.3f} | {max(abs(r['error_percent']) for r in rows if r['held_out']):.2f} |")
    lines += ['', '## 缩放诊断与可识别性', '',
        '半图/四分之一图用于检查缓存和 CTA 缩放；未事后指定固定项的缩放规律。',
        '所有逐 worker 斜率、资源绑定、counter 与缩放数据均保存在 analysis.json。',
        'HBM 饱和必须结合实际流量、带宽利用率和对照判断；平坦时间本身不构成充分证据。',
        'ML/MR 仍包括私有更新、累加与循环成本，可加性、驻留和完整 kernel 的重叠方式仍需区分。',
        '当前只验证 C1.6，未开始 C2、未更改 compiler、未创建 PR。', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    args=p.parse_args()
    c.protect_prior()
    result=derive()
    if args.validate:
        assert c.read(c.OUT/'analysis.json')==json.loads(c.encode(result))
        assert (c.OUT/'summary.md').read_text()==render(result)
    else:
        (c.OUT/'analysis.json').write_bytes(c.encode(result))
        (c.OUT/'summary.md').write_text(render(result))
    print('C1.6 report derived from 6480 event samples and 20 NCU reports')


if __name__=='__main__':
    main()
