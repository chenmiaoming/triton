"""Render the academic N1 baseline report after immutable analysis is ready."""
import argparse
from collections import Counter
import os
from pathlib import Path

from . import common as c
from .validate import _path


SUBTYPES = [
    ('N01', '量化 scale swizzle', 'torchao 的可选 Triton scale 重排；不包含 scale 计算或量化归约。'),
    ('N02', 'AWQ 反量化', 'vLLM 可选 Triton AWQ 解包/反量化；不是 AWQ GEMM，也不代表默认后端。'),
    ('N03/FA', '普通 RoPE', 'FlashAttention interleaved、非原地、完整 rotary dimension；本轮不涵盖 varlen/partial rotary 等分支。'),
    ('N03/MRoPE', '三轴 MRoPE', 'SGLang CUDA NeoX-style、BF16 cache、Q/K 原地更新；未覆盖 GLM/interleaved 的其他分支。'),
    ('N04', '固定 scale 的 FP8 QKV pack', '提取自 SGLang SM120 attention preparation；仅迁移独立 pack 到 H100，不是该后端在 H100 的生产默认。'),
    ('N05', 'KDA forward gate', '固定 A/bias，未启用 beta/lower-bound 分支；不包含 chunk-cumsum、反向或完整 KDA 流水。'),
    ('N06', 'KV cache scatter', 'SGLang CUDA HND 物理布局，唯一 slot 映射；未启用 SWA/scale，也不包含 attention。'),
]


def _link(path, report_directory):
    path=_path(path)
    return os.path.relpath(path,report_directory)


def _shape(case):
    family=case['family']
    if family=='N01':return f'{case["rows"]}×{case["cols"]}'
    if family=='N02':return f'packed {case["rows"]}×{case["packed_cols"]} → {case["rows"]}×{case["cols"]}'
    if family=='N03':
        if case['operator']=='flash_rope':return '×'.join(str(case[x]) for x in ('B','S','H','D'))
        return f'T={case["T"]}, QH/KH={case["QH"]}/{case["KH"]}, D={case["D"]}'
    if family=='N04':return f'H={case["H"]}, S={case["S"]}, D={case["D"]}'
    if family=='N05':return f'T={case["T"]}, H={case["H"]}, D={case["D"]}, BT={case["BT"]}'
    return f'T={case["T"]}, H={case["H"]}, D={case["D"]}, block={case["block_size"]}'


def _source_rows():
    sources=c.SOURCE/'sources';rows=[]
    for entry in c.read(sources/'quant_provenance.json')['records']:
        rows.append((entry['function'],entry['url'],entry['commit'],entry['Git_blob_SHA1']))
    for entry in c.read(sources/'rotary_provenance.json'):
        rows.append((entry['function'],entry['url'],entry['commit'],entry['upstream_git_blob_SHA1']))
    pack=c.read(sources/'pack_gate_provenance.json')
    for entry in pack['sources']:
        rows.append((entry['local'],entry['upstream_url'],entry['ref'],entry['github_blob_SHA']))
    entry=c.read(sources/'cache_provenance.json')
    rows.append(('reshape_and_cache_flash',entry['url'],entry['commit'],entry['Git_blob_SHA1']))
    return rows


def render(manifest_path, validation_path=None, report_directory=None):
    """Read completed evidence; never infer or change an acceptance decision."""
    manifest_path=_path(manifest_path);manifest=c.read(manifest_path)
    report_directory=Path(report_directory or c.OUT/'stage_n1')
    protocol_path=_path(manifest['baseline_protocol']);protocol=c.read(protocol_path)
    analysis_path=_path(manifest['baseline_analysis']);analysis=c.read(analysis_path)
    assert analysis['status']=='OK' and analysis['protocol_SHA256']==c.sha(c.encode(protocol))
    assert analysis['payload_SHA256']==protocol['payload_SHA256']
    assert all(row['variant'] in {'U','G0'} for row in analysis['rows'])
    assert len(protocol['cases'])==14 and len({x['family'] for x in protocol['cases']})==6
    assert len(manifest['accepted'])==14
    actual=manifest['actual_G0']
    actual_dir=c.OUT/'stage_n1'/f'revision_{actual["revision"]:03}'/'g0bind'/f'attempt_{actual.get("attempt",1):03}'/'export'
    actual_result=c.read(actual_dir/'result.json')
    assert actual_result['status']=='OK' and len(actual_result['records'])==14
    assert all(row['allocated_sections_equal'] and row['ABI_and_launch_equal'] for row in actual_result['records'])
    raw_paths=[_path(path) for path in manifest['baseline_raw']]
    raws=[c.read(path) for path in raw_paths]
    correctness=[raw for raw in raws if raw['invocation']==0]
    timing=[raw for raw in raws if raw['invocation']>0]
    assert len(correctness)==1 and len(timing)==protocol['timing']['workers']
    assert all(raw['status']=='OK' for raw in raws)
    assert len(actual_result['checks'])==42 and all(row['pass'] for row in actual_result['checks'])
    assert all(not raw['Triton_imported'] and raw['compile_calls']==0 and
               not raw['profiler_attached'] for raw in raws)
    if validation_path:
        validation_path=_path(validation_path);validation=c.read(validation_path)
        assert validation['status']=='OK' and not validation['errors']
        assert not validation['pending_final_gates']
    configurations=c.read(c.OUT/'stage_n1/revision_003/autotune/attempt_001/export/result.json')
    assert configurations['status']=='OK'
    selected=configurations['selected_cases']
    audit=analysis['audit'];cfg=protocol['timing']
    classifications=Counter(row['classification'] for row in analysis['baseline_equivalence'])
    initial_analysis_path=c.OUT/'stage_n1_baseline_r2/analysis/analysis.json'
    initial_analysis=c.read(initial_analysis_path)
    initial_classes=Counter(row['classification'] for row in initial_analysis['baseline_equivalence'])
    revision_status=[];cache_rows=[]
    for path in sorted((c.OUT/'stage_n1').glob('revision_*/*/attempt_*/export/result.json')):
        result=c.read(path)
        revision_status.append((str(path.relative_to(c.OUT/'stage_n1')),result['status']))
        if 'native_rebuilds' in result:assert result['native_rebuilds']==0
        if 'ccache_before' in result and 'ccache_after' in result:
            cache_rows.append(result['ccache_before']==result['ccache_after'])
    lines=['# Stage N1：无规约算子的 U / G0 基线对齐','',
        '本阶段建立可复核的实验基线：六个算子族、七个计算子类、14 个固定 case。'
        'U 为冻结的公开 Triton 程序；G0 按 U 的实际编译结果重建。'
        '本阶段仅比较 U/G0，不据此判断候选布局收益或跨算子的共同性能机制。','',
        '## 1. 研究范围与公开源码','',
        '| 族/子类 | 独立 kernel | 本轮适用范围 |','|---|---|---|']
    lines += [f'| {family} | {name} | {scope} |' for family,name,scope in SUBTYPES]
    lines += ['', '所有被测 kernel 的源码与生成 IR 均检查了 reduction、scan、dot 和原子累积。'
              '逐元素 maximum、三轴系数逐元素相加及位解包不属于轴规约。'
              '完整公开文件可能还包含其他未执行的 kernel；检查范围为所选函数及其本地 helper 闭包。','',
              '| 函数/依赖 | 冻结 commit | 原始 Git blob |','|---|---|---|']
    for name,url,commit,blob in _source_rows():
        lines.append(f'| [{name}]({url}) | `{commit[:12]}` | `{blob}` |')
    lines += ['', '原始完整源码、独立提取文件与 AST/Git-blob 指纹分别保留。'
              'N04 的 `_pack_qkv` 与 helper 保持原始函数体/签名；N05 保留 NVIDIA softplus 与 '
              '`FLA_USE_FAST_OPS=0` 的 exp 语义，仅把原来的宿主 autotune/heuristic 包装拆开记录。'
              'N06 的绑定文件与原始完整文件只差末尾换行，kernel 和 wrapper AST 相同。','',
              '## 2. 配置、编译与正确性门','',
              f'- 编译器 commit：`{protocol["compiler"]}`；native SHA256：`{protocol["native_SHA256"]}`。',
              '- 原始 U 使用实际 Torch Tensor JIT 绑定，保留真实 dtype、stride、storage alias、constexpr 与运行时整数的 specialization。',
              '- revision 008：14/14 个 CPU G0 与 U 的全部 allocated ELF section、运行 ABI/launch 及资源使用相同。'
              '比较不只覆盖 PTX/SASS 计数，也不要求调试节或非 allocated 节的文件字节相同。',
              f'- revision {actual["revision"]:03}：实际 Tensor JIT 的 G0、CPU G0 与 U 三方 allocated ELF/ABI/资源检查全部通过；'
              f'G0 共 {len(actual_result["checks"])} 项 finite/structured/edge 正确性检查通过。',
              f'- 独立冻结二进制 driver 检查：{analysis["audit"]["logical_programs"]} 个 U/G0 程序，'
              f'{len(correctness[0]["checks"])} 项完整输出/只读与未激活区域/guard 检查通过。'
              f'三个计时 worker 另各有 {", ".join(str(len(raw["checks"])) for raw in timing)} 项有限值检查；'
              '交错采样协议下，这些检查位于首个 visit 的 warmup 后、正式采样轮之前。',
              '- MRoPE 的 q/k 是原地输出。每次 warmup 与每个计时样本之前都恢复原始物理存储；'
              '恢复操作位于 graph 的两个计时事件之外。该 kernel 时间不含恢复/宿主搬运成本。','',
              '| Case | 逻辑形状/参数 | Warps | Stages |','|---|---|---:|---:|']
    for case in protocol['cases']:
        lines.append(f'| `{case["id"]}` | {_shape(case)} | {case["W"]} | {case["stages"]} |')
    lines += ['', 'N05 先只用 U 遍历原公开的 `BT∈{32,64,128} × W∈{4,8,16,32} × stages∈{2,3}`，'
              '即每个 case 24 个配置。每配置 3 个 visit、每 visit 9 个 graph 样本，按 visit 中位数的均值选取。'
              '这是保留原配置菜单的独立 U-only 选择，不冒充完整 FLA autotuner/cache 实现，也未利用 Gluon 布局速度选参。']
    for case in selected:
        lines.append(f'- `{case["id"]}` 选定：`BT={case["BT"]}, W={case["W"]}, stages={case["stages"]}`。')
    lines += ['', '## 3. 同轮 U/G0 计时','',
              f'正式估计量为 {cfg["workers"]} 个 worker × 每 worker {cfg["visits"]} 个 visit；'
              f'每 visit {cfg["samples"]} 个原始样本先取中位数，再对 worker/visit 中位数取算术平均。'
              f'共 {audit["raw_sample_count"]:,} 个正式原始样本。'
              f'每次 graph 仅一个 kernel 与两个外部计时事件，未附加 profiler。',
              '', '采样方案修订：初轮采用逐程序成块采样，观察到部分 visit 间漂移及时间离群；'
              '查看全部 U/G0 控制后，对全部 14 个 case 统一改为同一 case 内逐 round 随机交错 U/G0，'
              '并统一增加为 10 个 visit。每 round 每个物理程序各获得一个观测，完整顺序随原始记录保留。'
              'kernel、单 kernel graph、输入与冻结二进制保持相同；原轮未删除、未与新轮混合。'
              '这是查看初轮控制后的自适应采样修订，不是初轮预注册方案的确认性结果。','',
              '| 采样轮 | 顺序与 visit 数 | 原始样本数 | 等价 / 不确定 |','|---|---|---:|---|',
              f'| [r2 初轮]({_link(initial_analysis_path,report_directory)}) | 逐程序成块，5 visits | '
              f'{initial_analysis["audit"]["raw_sample_count"]:,} | '
              f'{initial_classes["practically_equivalent"]} / {initial_classes["inconclusive"]} |',
              f'| 当前冻结轮 | `{cfg.get("sample_order","blocked")}`，{cfg["visits"]} visits | '
              f'{audit["raw_sample_count"]:,} | {classifications["practically_equivalent"]} / '
              f'{classifications["inconclusive"]} |',
              '', '新轮的统计稳定性与更均衡的时间采样一致；没有用 profiler 对初轮离群的硬件原因进行归因，'
              '也不把这一采样比较解释为某条 GPU 指令或布局机制的因果实验。',
              '', 'U/G0 在每个 worker 独立测量。4096 次配对分层 bootstrap 共用 worker/visit 抽样索引；'
              '95% CI 完全落在 ±3% 内才分类为 practical equivalence，其余分类按观察值保留。'
              '机器代码等价门与带噪声的计时分类分别报告；完整保留每轮数据及采样方案修订。','',
              '| Case | U (µs) | G0 (µs) | G0 相对 U 收益 (%) | 配对 95% CI (%) | 分类 |',
              '|---|---:|---:|---:|---|---|']
    for row in analysis['baseline_equivalence']:
        low,high=row['CI95']
        lines.append(f'| `{row["case"]}` | {row["U_time_us"]:.3f} | {row["G0_time_us"]:.3f} | '
                     f'{row["gain_percent"]:+.2f} | [{low:+.2f}, {high:+.2f}] | {row["classification"]} |')
    lines += ['', '分类计数：'+'；'.join(f'`{key}`={value}' for key,value in sorted(classifications.items()))+'。','',
              '| 计时 worker | GPU UUID | GPU / SM 数 |','|---|---|---|']
    for raw in timing:
        env=raw['environment']
        lines.append(f'| {raw["invocation"]} | `{env["gpu_uuid"]}` | {env["name"]} / {env["SM_count"]} |')
    lines += ['', '## 4. 缓存、失败记录与证据边界','',
              '- 各 worker 的 native SHA 固定，记录的 native rebuild 数均为 0；复用既有 Triton image、'
              '内核编译缓存与 Modal cache volume，没有重新构建 Triton C++。'
              f'保留的 {len(cache_rows)} 组 ccache 前后快照中，{sum(cache_rows)} 组计数未变；'
              '这些快照用于检查 native 构建活动，不据此推导未记录的 Triton kernel-cache 命中率。',
              '- revision 001 的冻结依赖缺失、首个 baseline worker 导入时读取本地主机协议路径的失败均原样保留。'
              '两者未形成正式计时样本；后者改用新增的 `stage_n1_baseline_r2`，不覆盖原协议。',
              '- 各次 G0 编译失败或不匹配也保留在独立 revision。最终接受 revision 008 与实际绑定 revision 009；'
              '不要求历史失败尝试变成通过。',
              '- H100 上的独立 kernel 证据不能外推为 SM120/B200、完整 LLM 或框架默认路径的收益。'
              '本阶段尚未证明哪些候选布局更快，也未证明存在版本性能回归。后续布局比较需要同轮 U/G0 与完整候选真值。','',
              '## 5. 可复核产物','',
              f'- [接受清单]({_link(manifest_path,report_directory)})，SHA256：`{c.sha(manifest_path.read_bytes())}`。',
              f'- [正式协议]({_link(protocol_path,report_directory)})，SHA256：`{analysis["protocol_SHA256"]}`。',
              f'- [计时分析]({_link(analysis_path,report_directory)})；全部原始记录由其输入 SHA 列表与接受清单关联。',
              f'- [实际 G0 绑定]({_link(actual_dir/"result.json",report_directory)})；保留三方完整 IR/PTX/SASS/CUBIN、资源与检查记录。',
              f'- [源码 provenance]({_link(c.SOURCE/"sources",report_directory)})；各 revision 保留协议与 source snapshot ZIP。']
    if validation_path:
        lines.append(f'- [最终 CPU validator]({_link(validation_path,report_directory)})：`OK`。')
    lines += ['', '历史尝试索引（状态不作改写）：','', '| 记录 | 状态 |','|---|---|']
    for path,status in revision_status:
        lines.append(f'| `{path}` | {status} |')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=c.OUT/'stage_n1/accepted_manifest.json')
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--output',type=Path,default=c.OUT/'stage_n1/REPORT.md')
    args=parser.parse_args()
    content=render(args.manifest,args.validation,args.output.parent)
    c.write_once(args.output,content.encode())
    print(args.output)


if __name__=='__main__':
    main()
