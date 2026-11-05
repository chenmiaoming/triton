"""Render the completed N5 evidence, retaining positive and negative outcomes.

Writing is allowed only after all six hardware profiles and both independent
formal timing comparisons are available.  The renderer never chooses a new
candidate or changes a frozen experimental artifact.
"""
import argparse
from pathlib import Path

from . import common as c
from .n5_profile_prepare import PAYLOAD_SHA256


def storage_bytes(metadata):
    """Physical data extents from actual fixture views; allocation guards excluded."""
    groups = {}
    for name, view in metadata['fixture']['views'].items():
        assert all(stride>0 for stride in view['stride'])
        size = (view['storage_offset']+1+sum((n-1)*s for n,s in zip(view['shape'],view['stride']))) * view['element_size']
        group = groups.setdefault(view['group'],dict(bytes=0,names=[]))
        group['bytes'] = max(group['bytes'],size)
        group['names'].append(name)
    return dict(total=sum(group['bytes'] for group in groups.values()),groups=groups)


def load_inputs(root=c.OUT):
    root = Path(root)
    files = dict(
        binding_protocol=root/'stage_n5/revision_002/preregistration/protocol.json',
        binding=root/'stage_n5/revision_002/binding/attempt_001/export/result.json',
        binding_validation=root/'stage_n5/revision_002/BINDING_VALIDATION.json',
        timing_protocol=root/'stage_n5_timing/preregistration/protocol.json',
        timing=root/'stage_n5_timing/analysis/analysis.json',
        profiles_protocol=root/'stage_n5_profiles/preregistration/protocol.json',
        profiles=root/'stage_n5_profiles/analysis/summary.json')
    values = {name:c.read(path) for name,path in files.items()}
    assert values['binding']['status']==values['binding_validation']['status']=='OK'
    assert values['binding']['counts']==dict(cases=3,programs=16,full_correctness_checks=48,passing_checks=48)
    assert values['binding']['native_rebuilds']==0 and not values['binding']['formal_timing']
    assert values['binding']['ccache_before']==values['binding']['ccache_after']
    assert values['timing']['status']==values['profiles']['status']=='OK'
    assert values['timing']['payload_SHA256']==values['profiles']['payload_SHA256']==PAYLOAD_SHA256
    assert values['timing_protocol']['payload_SHA256']==values['profiles_protocol']['payload_SHA256']==PAYLOAD_SHA256
    assert values['timing']['protocol_SHA256']==c.sha(c.encode(values['timing_protocol']))
    assert values['profiles']['protocol_SHA256']==c.sha(c.encode(values['profiles_protocol']))
    assert values['profiles']['report_count']==len(values['profiles']['rows'])==6
    assert values['profiles']['InstructionStats_collected'] is False
    assert values['profiles']['baseline_software_crosscheck']['status']=='NOT_COLLECTED'
    assert all(row['mode']=='warm_application' for row in values['profiles']['rows'])
    assert len(values['timing']['baseline_equivalence'])==2
    assert all(row['classification']=='practically_equivalent' for row in values['timing']['baseline_equivalence'])
    values['input_files'] = {name:dict(path=str(path.relative_to(root)),SHA256=c.sha(path.read_bytes()))
                            for name,path in files.items()}
    return values


def _interval(values):
    return '['+', '.join(f'{value:+.4f}%' for value in values)+']'


def render(data):
    binding, bp = data['binding'], data['binding_protocol']
    timing, tp = data['timing'], data['timing_protocol']
    profiles = data['profiles']
    cases = [case for case in binding['cases'] if case['followup_role']=='footprint']
    index = {(row['case'],row['variant']):row for row in timing['rows']}
    baseline = {row['case']:row for row in timing['baseline_equivalence']}
    candidates = {case['id']:'GL_'+case['selected_variant'] for case in cases}
    bytes_by_case = {record['case']:storage_bytes(record['metadata']) for record in binding['records']
                     if record['case'] in candidates}
    assert bytes_by_case['N01_m4096_n8192']['total']==64*2**20
    assert bytes_by_case['N02_k8192_p1024_g128']['total']==161.25*2**20
    n01 = index['N01_m4096_n8192',candidates['N01_m4096_n8192']]
    n02 = index['N02_k8192_p1024_g128',candidates['N02_k8192_p1024_g128']]
    assert n01['gain_vs_U_CI95'][0]>3
    assert -3<n02['gain_vs_U_CI95'][0]<n02['gain_vs_U_CI95'][1]<0
    lines = ['# Stage N5 — footprint 与尾块/步幅跟进','',
        f"固定候选在新 footprint 上表现不同：N01 scale swizzle 相对 U 提高 **{n01['gain_vs_U_percent']:.4f}%**"
        f"（95% CI {_interval(n01['gain_vs_U_CI95'])}）；AWQ dequant 候选则慢 **{-n02['gain_vs_U_percent']:.4f}%**"
        f"（收益 CI {_interval(n02['gain_vs_U_CI95'])}）。两项 U/G0 基线均实测等价。"
        'AWQ 的方向为小幅变慢，但整个区间仍在预注册 ±3% 实用等价带内；不能称为 >3% 的实用退化。','',
        '这轮是已知 kernel 的条件跟进。两个大 case 各只有一个事先绑定的候选：'
        '`N01 coordinated_fully_coalesced` 与 `N02 packed_compute_vec1`；未根据本轮计时或计数重新选择、修改候选。'
        '编译器版本固定，因此这里讨论布局选择与尺寸的关系，不把它称为版本升级引起的性能回归。','',
        '## 1. 程序、footprint 与正确性','',
        '| Case | 数据 storage 体积 | 固定候选 | 正确性覆盖 |',
        '| --- | --- | --- | --- |',
        '| N01_m4096_n8192 | uint8 input 32 MiB + output 32 MiB = **64 MiB** | coordinated_fully_coalesced | U/G0/GL × 3 patterns，9/9 |',
        '| N02_k8192_p1024_g128 | packed int32 weight 32 MiB + FP16 scale 1 MiB + zero 0.25 MiB + FP16 output 128 MiB = **161.25 MiB** | packed_compute_vec1 | U/G0/GL × 3 patterns，9/9 |',
        '| N01_tail_stride_m131_n7 | input logical 131×7，stride=(19,2)；完整 padded output=256×8 | 原八个声明布局全部保留，包括默认 alias | U/G0/8 GL × 3 patterns，30/30 |','',
        '以上体积来自真实绑定 fixture 的 shape/stride/element size 对应的数据 storage，不含每个 allocation 的 256B 前后 guard。'
        '大 case 的数据 tensor 均为完整连续 storage；尾块输入覆盖 2483B 物理跨度，其中 917B 为逻辑元素、1566B 为只读间隙。'
        '输出的 2048 个 uint8 位置全部验证，包含 1131 个由 padding 产生的零值位置。','',
        '**总计 16 个逻辑程序、48/48 个完整检查通过**，覆盖 finite/structured/edge、独立 oracle、guard、'
        '只读输入及存储间隙。三个 case 的真实 Tensor U/G0 JIT 均通过 allocated ELF、运行 ABI/launch 与资源一致性门。'
        '尾块仅用于正确性检查，不进入计时或这六份 NCU 报告。','',
        f"编译器 commit：`{bp['compiler']}`；native SHA256：`{binding['native_SHA256']}`。"
        '复用现有编译镜像及 kernel/compiler 缓存；绑定阶段 ccache 前后统计相同，native rebuild=0。'
        '正式计时与 NCU 阶段均加载冻结 CUBIN，不执行 JIT。','',
        '## 2. 无 profiler 的正式计时','',
        f"{tp['timing']['workers']} 个独立 worker，每个 {tp['timing']['visits']} visits、"
        f"每个 visit {tp['timing']['samples']} 个原始样本，共 {timing['audit']['raw_sample_count']} 个样本。"
        '每轮随机交错测量各物理程序；单 kernel CUDA Graph 配两个外部计时事件。'
        '点估计为 worker×visit 中位数的算术平均；4096 次配对 worker/visit 层次 bootstrap，U/G0/GL 共享抽样。'
        '区间为逐点 95% percentile CI，未作多重比较校正。','',
        '| Case | U (µs) | G0 (µs) | 固定 GL (µs) | GL vs U 收益与 CI | GL vs G0 收益与 CI |',
        '| --- | --- | --- | --- | --- | --- |']
    for case in cases:
        name = case['id']
        candidate = index[name,candidates[name]]
        lines.append(f"| {name} | {index[name,'U']['time_us']:.4f} | {index[name,'G0']['time_us']:.4f} | {candidate['time_us']:.4f} | {candidate['gain_vs_U_percent']:+.4f}% {_interval(candidate['gain_vs_U_CI95'])} | {candidate['gain_vs_G0_percent']:+.4f}% {_interval(candidate['gain_vs_G0_CI95'])} |")
    lines += ['', '| Case | G0 vs U 收益 | 95% CI | 基线结论 |',
        '| --- | --- | --- | --- |']
    for case in cases:
        value = baseline[case['id']]
        lines.append(f"| {case['id']} | {value['gain_percent']:+.4f}% | {_interval(value['CI95'])} | ±3% 内实用等价，U/G0 独立物理测量 |")
    lines += ['', 'N01 在这一较大 footprint 上仍有超过 3% 的收益。AWQ 的相同候选没有获得实用收益，且观察到约 2% 的小幅反向。'
        '因此，减少某条转换路径或增加资源容量都不能单独当作端到端收益判据。此处没有重扫其他 AWQ 布局来替换该结果。','',
        '## 3. 当前 footprint 的 NCU 硬件观测','',
        '六份报告均为完整 grid、`application replay / cache none / clock none`，八次同程序 warmup 后采集一个 kernel。'
        '每份报告独立加载一个冻结二进制，全部 replay 的 loaded SHA、完整 oracle/guard、原始参数、GPU UUID 和 CUDA q 已验证。'
        '**N5 没有采集 InstructionStats**，不生成 CTA0 软件指令表，也没有复用小尺寸 N3 的全图缓存百分比。'
        '以下只使用本轮大 footprint 的真实计数。','',
        '| Case / Variant | CUDA REG / capacity q | L2 hit (%) | DRAM read / write (MiB) | DRAM % peak | active warps % peak |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in profiles['rows']:
        metric = lambda name:row['selected_metrics'][name]['value']
        lines.append(f"| {row['case']} / {row['variant']} | {row['resources']['n_regs']} / {row['resources']['occupancy_capacity_q']} | {metric('lts__t_sector_hit_rate.pct'):.4f} | {metric('dram__bytes_read.sum')/2**20:.4f} / {metric('dram__bytes_write.sum')/2**20:.4f} | {metric('dram__throughput.avg.pct_of_peak_sustained_elapsed'):.4f} | {metric('sm__warps_active.avg.pct_of_peak_sustained_active'):.4f} |")
    lines += ['', '| Case / Variant | Global load requests / sectors per CTA | Global store requests / sectors per CTA | Shared load / store wavefronts per CTA |',
        '| --- | --- | --- | --- |']
    for row in profiles['rows']:
        per = lambda name:row['selected_additive_counts_per_CTA'][name]['value']
        lines.append(f"| {row['case']} / {row['variant']} | {per('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'):.4f} / {per('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'):.4f} | {per('l1tex__t_requests_pipe_lsu_mem_global_op_st.sum'):.4f} / {per('l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'):.4f} | {per('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'):.4f} / {per('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'):.4f} |")
    lines += ['', '`capacity q` 来自 CUDA occupancy API，是给定函数与 block/shared 配置的最大 CTA 容量；'
        '它不是实际运行时始终驻留的 CTA 数。active-warps 百分比是另一项实测指标，不能与 q 混写。'
        '只对 additive 流量/指令 sum 按真实 CTA 数归一；百分比及比值保持原单位。','',
        '## 4. 正向与反向结果的条件解释','']
    for case in cases:
        name = case['id']
        candidate = next(row for row in profiles['rows'] if row['case']==name and row['variant']==candidates[name])
        reference = next(row for row in profiles['rows'] if row['case']==name and row['variant']=='U')
        raw = lambda row,key:row['selected_metrics'][key]['value']
        per = lambda row,key:row['selected_additive_counts_per_CTA'][key]['value']
        measured = index[name,candidates[name]]
        lines += [f"### {name}",'',
            f"冻结候选的正式收益为 {measured['gain_vs_U_percent']:+.4f}% {_interval(measured['gain_vs_U_CI95'])}。"
            f"本轮 U→GL 的全局 store sectors/CTA 为 {per(reference,'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'):.4f}→{per(candidate,'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'):.4f}；"
            f"global load requests/CTA 为 {per(reference,'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'):.4f}→{per(candidate,'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'):.4f}；"
            f"shared load wavefronts/CTA 为 {per(reference,'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'):.4f}→{per(candidate,'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'):.4f}。"
            f"REG / capacity q 从 {reference['resources']['n_regs']} / {reference['resources']['occupancy_capacity_q']} 变为 {candidate['resources']['n_regs']} / {candidate['resources']['occupancy_capacity_q']}。",'',
            f"大 footprint 下测得的 L2 hit 为 {raw(reference,'lts__t_sector_hit_rate.pct'):.4f}%→{raw(candidate,'lts__t_sector_hit_rate.pct'):.4f}%；"
            f"DRAM read/write 为 {raw(reference,'dram__bytes_read.sum')/2**20:.4f}/{raw(reference,'dram__bytes_write.sum')/2**20:.4f} MiB→"
            f"{raw(candidate,'dram__bytes_read.sum')/2**20:.4f}/{raw(candidate,'dram__bytes_write.sum')/2**20:.4f} MiB。"
            '这些是当前二进制与当前运行条件的观测，不能从 storage 大小预先推出。','']
        if case['family']=='N01':
            lines += ['该对照同时协调输入与输出所有权，保留原始字节置换。请求分组的变化与本轮实用收益可以并列观察；'
                '本轮没有独立扫描加载、写回和缓存并发，因而不把全部收益唯一归因于某一项计数。','']
        else:
            lines += ['该对照保留原始 AWQ 解包、零点、scale 和 FP16 输出语义。即使通信或资源指标改善，也必须保留这次小幅变慢的正式结果；'
                '加载指令的宽度/数量、请求、展开计算与并发掩盖会共同决定净时间，本轮没有分量计时来唯一分解它们。','']
    lines += ['L2 hit 是缓存层汇总比例，不能据此证明全部 tensor 的驻留状态；DRAM 读写也受缓存及写策略影响。'
        '这轮没有独立测定纯 HBM 带宽上界、流量下限和重叠边界，因此不宣称“超过 L2 容量就处于纯 HBM roof”。'
        'NCU 下的时间也不替代第 2 节无 profiler 的正式计时。','',
        '## 5. 失败与派生修订均保留','',
        '- `revision_001`：两个大 case 已通过 18 个完整检查；尾块 U/G0 的编译一致性也通过。'
        '旧 adapter 跳过与新默认相同的输入布局，随后触发固定八个候选的枚举断言，尾块执行前终止。'
        '完整 raw export 保留。`revision_002` 仅在独立 N5 协议中绑定原先八个候选定义并保留 alias，未修改冻结 adapter 或内核。',
        '- 首份派生 `BINDING_REPORT.md` 误用展开宽度填写 AWQ 输入 stride；保留原文件，'
        '`BINDING_REPORT_v2.md` 按真实 fixture 元数据改为 packed qweight shape=`8192×1024`、stride=`(1024,1)`。原始实验与验证未变。',
        '- 正式表同时保留 N01 的收益与 AWQ 的小幅反向，不用新的候选或新的指标替换不利结果。','',
        '## 6. 证据入口与停点','',
        '- [真实 JIT 绑定、48 项检查和资源门](revision_002/BINDING_REPORT_v2.md)',
        '- [绑定 CPU 最终校验](revision_002/BINDING_VALIDATION.json)',
        '- [正式计时分析及所有配对区间](../stage_n5_timing/analysis/analysis.json)',
        '- [六份硬件报告的独立数值分析](../stage_n5_profiles/analysis/summary.json)',
        '- [NCU 口径与完整指标长表](../stage_n5_profiles/analysis/REPORT.md)', '',
        '本轮补齐的是当前 kernel、当前设备和两个新 footprint 下的证据，以及一项尾块/步幅正确性压力测试。'
        '它不证明所有模型尺寸或其他架构上的收益，也没有启动新的布局调优或编译器修改。','',
        '| 输入 | SHA256 |','| --- | --- |']
    for name,value in data['input_files'].items():
        lines.append(f"| {name} | `{value['SHA256']}` |")
    return '\n'.join(lines)+'\n'


def run(root=c.OUT):
    root = Path(root)
    target = root/'stage_n5/REPORT.md'
    assert not target.exists(), target
    data = load_inputs(root)  # Raises before writing until all profiles are ready.
    text = render(data)
    c.write_once(target,text.encode())
    print('N5 completed report written once:',target)
    return text


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=c.OUT)
    run(parser.parse_args().root)
