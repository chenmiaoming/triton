"""Derive a timing-blind review from frozen N2 CPU compiler artifacts only."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from . import common as c


SASS_INSTRUCTION = re.compile(
    r'/\*[0-9a-f]+\*/\s+(?:@!?(?:UP|P)\d+\s+)?([A-Z][A-Z0-9_.]+)\b')


def subtype(case):
    return 'N03_' + case['operator'] if case['family'] == 'N03' else case['family']


def summarize(program, root):
    folder = root / program['artifact_directory']
    for name, digest in program['artifact_SHA256'].items():
        assert c.sha((folder / name).read_bytes()) == digest, (folder, name)
    sass = (folder / 'kernel.sass').read_text()
    ir = (folder / 'kernel.ttgir').read_text()
    opcodes = Counter(SASS_INSTRUCTION.findall(sass))
    assert opcodes, folder
    total = lambda prefixes: sum(n for op, n in opcodes.items() if op.startswith(prefixes))
    converts = [line.strip().split(' loc(')[0] for line in ir.splitlines()
                if 'ttg.convert_layout' in line]
    identity = sum(line.split(' : ', 1)[1].split(' -> ', 1)[0]
                   == line.split(' -> ', 1)[1] for line in converts)
    regs, warps = program['resources']['REG'], program['num_warps']
    # Deliberately loose architectural ceiling, not CUDA occupancy.  Allocation
    # rounding, shared-memory carveout and device/runtime details are omitted.
    ideal_q = min(32, 64 // warps, 65536 // (regs * warps * 32))
    return dict(geometry=program['geometry'], subtype=subtype(program['case']),
        variant=program['variant'], case=program['case'],
        artifact_directory=program['artifact_directory'],
        artifact_SHA256=program['artifact_SHA256'], cubin_SHA256=program['cubin_SHA256'],
        resources=program['resources'], dynamic_smem_bytes=program['dynamic_smem_bytes'],
        ideal_register_warp_q_ceiling=ideal_q, CUDA_occupancy_q=None,
        static_SASS_sites=dict(global_load=total(('LDG',)), global_store=total(('STG',)),
            shared_load=total(('LDS', 'LDSM')), shared_store=total(('STS', 'STSM')),
            CTA_barrier=total(('BAR.SYNC',)), shuffle=total(('SHFL',)),
            local_memory=total(('LDL', 'STL'))),
        opcode_histogram=dict(sorted(opcodes.items())),
        IR_convert_sites=dict(total=len(converts), exact_identity=identity,
                              non_identity=len(converts)-identity),
        conversions=converts,
        memory_operations=[line.strip().split(' loc(')[0] for line in ir.splitlines()
                           if re.search(r'\btt\.(?:load|store)\b', line)],
        encoding_definitions=[line for line in ir.splitlines()
                              if re.match(r'#\w+\s*=\s*#ttg\.', line)],
        choice=program.get('choice'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---']*len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def describe(data):
    rows = data['programs']
    by_case = defaultdict(list)
    for row in rows:
        by_case[row['geometry']].append(row)
    counts = []
    for identity, values in by_case.items():
        candidates = [x for x in values if x['variant'].startswith('GL_')]
        g0 = next(x for x in values if x['variant'] == 'G0')
        counts.append([identity, g0['subtype'], len(values), len(candidates),
                       len({x['cubin_SHA256'] for x in values}),
                       g0['resources']['REG'],
                       str(min(x['resources']['REG'] for x in candidates))+'–'+str(max(x['resources']['REG'] for x in candidates)),
                       g0['static_SASS_sites']['CTA_barrier'],
                       str(min(x['static_SASS_sites']['CTA_barrier'] for x in candidates))+'–'+str(max(x['static_SASS_sites']['CTA_barrier'] for x in candidates))])

    lines = [
        '# N2 静态编译证据复核（未读取计时结果）', '',
        '本文件只读取成功的 CPU 编译包及其中的 TTGIR、SASS、CUBIN 资源清单。'
        '不读取 GPU 正确性、计时或 NCU 结果，不给布局打分，也不据此选择赢家。'
        '输入为 [冻结编译结果](compile/attempt_001/invocation_-01/export/result.json)，'
        '所有逐程序提取字段见 [STATIC_REVIEW.json](STATIC_REVIEW.json)。', '',
        f"输入 result.json SHA256：`{data['input_result_SHA256']}`。"
        f"覆盖 {data['counts']['cases']} 个 case、7 种 kernel 子类型、"
        f"{data['counts']['GL_candidates']} 个 GL 候选及 28 个 U/G0，共 138 个逻辑程序。", '',
        '## 计数口径与证据边界', '',
        '表中的 LDG、STG、LDS、STS、BAR、SHFL 数字均为反汇编中的**静态指令位置数**，'
        '包括 LDSM/STSM。它们不是每 CTA 动态执行次数、wavefront、bank conflict、sector 或耗时。'
        '谓词、活跃 warp、循环与分支均会改变动态计数。'
        'IR 中同类型到同类型的 identity convert 也不等价于硬件通信。', '',
        'NCU 将 sector 定义为对齐的 32B 内存块；一次 warp 请求可涉及多个 sector 和 wavefront。'
        '因此，改变指令宽度或静态指令数不能直接推出 L2/DRAM 字节数或性能。'
        '这些区别按 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) 使用。', '',
        'REG/SHARED/LOCAL/STACK 为已编译 CUBIN 的资源记录。全部 138 个程序的 LOCAL/STACK 均为 0，'
        '本次静态提取也未发现 LDL/STL 位置。JSON 另列仅用 64K 寄存器、64 warps、32 CTA 上限得到的'
        '`ideal_register_warp_q_ceiling`，刻意忽略分配粒度及共享内存等限制，**不是 occupancy q**。'
        '实际容量须按具体 CUfunction、block size 和动态 shared bytes 调用 '
        '`cuOccupancyMaxActiveBlocksPerMultiprocessor`；实际运行的活跃 CTA 又是另一项观测。'
        '参见 [Hopper Occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) 与 '
        '[CUDA Driver Occupancy API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html)。', '',
        '## 每个 case 的覆盖与资源变化', '',
        table(['case','子类型','逻辑程序','GL 候选','full-CUBIN 种类','G0 REG','GL REG 范围','G0 BAR 位点','GL BAR 位点范围'], counts), '',
        'full-CUBIN 的去重必须保留几何和运行时参数。共有 128 个 `(geometry, full CUBIN SHA)` 条件；'
        '108 个全局二进制 SHA 不能用作 108 个实验条件，因为同一个通用二进制可在不同几何上运行。'
        'U/G0 通过的是 allocated-ELF/ABI/resources 等价门，调试等非 allocated 内容仍可能令 full CUBIN SHA 不同。', '',
        '## N01：scale swizzle 的输入与输出分布', '',
        '原始算子为 uint8 的 `128×4` tile 重排，输出地址为 '
        '`(row % 32)*16 + (row // 32)*4 + col`。scale 已给定；这里没有 scale 计算、规约或 TMA。'
        '输入默认按四个相邻列分给线程；输出的物理位置则按行的低/高位重新排列。', '',
        '四个 `load_*` 候选只改变输入 encoding，输出保持默认；两个 `output_*` 只改变输出所有权；'
        '两个 `coordinated_*` 令输入所有权也投影到同一组输出线程。'
        '后两组并未改变数学置换或输出域。独立重建输出索引修复了首轮 AutoLayout 冲突；'
        'R103 再次确认修复后的 G0 与 U 的 allocated ELF、ABI、资源完全一致。', '',
        '实际编译已经出现可区分的通信路径：默认没有 shared 指令；两个 output-only 候选引入 1 KiB shared '
        '及一个 BAR 位点；匹配输入的 coordinated 候选又消除了这些 shared/同步操作。'
        '`load_s2x2_t16x2_o10` 采用 SHFL，其他若干 load-only 候选使用 shared 重分布。'
        '`load_s4x1_t8x4_o01` 有 shared 指令但无 BAR.SYNC 位点，不能把所有 shared 转换都记成一次 CTA barrier。', '',
        '命名中的 fully/partially coalesced 是地址分组的预期：前者把原来分散的输出字节在一个 warp 内更集中地分组。'
        '这可以减少输出侧请求的 sector 分散，但 output-only 为此付出了通信；coordinated 对照检验该通信能否避免。'
        '这些是待 NCU 检验的请求层预期，不等于 DRAM 写流量按同一比例下降。输入侧的 stride、缓存复用和活跃线程仍须一起观察。', '',
        '完整输出 linear bases：`reg=[[1],[2]]; lane=[[128],[256],[4],[8],[16]]; warp=[[32],[64]]`；'
        '部分输出 bases：`reg=[[1],[2]]; lane=[[4],[8],[16],[32],[128]]; warp=[[64],[256]]`。'
        '二者都是 shape=512 的合法双射；coordinated 输入把每个 flat basis `b` 投影为 `[b//4,b%4]`。', '',
        '## N02：AWQ unpack / scale / 输出之间的分布', '',
        '每个 tile 从 `32×32` packed int32 权重展开到 `32×256`；零点行也展开，scale 为 `1×256`。'
        '三次 interleave 是值的 join/reshape，不能据此声称硬件加载了八倍输入。'
        'nibble 顺序保持 `[0,4,1,5,2,6,3,7]`，组大小为 128。', '',
        '默认权重/零点 encoding 为 SPT=[1,4], TPW=[4,8], WPC=[4,1]；展开后 SPT=[1,32]。'
        'scale load 为 SPT=[1,2], TPW=[1,32], WPC=[1,4]，输出为 SPT=[1,8], TPW=[1,32], WPC=[4,1]。'
        '整数差先转换为 FP16，再转为输出布局；scale 行先转为输出布局再广播。候选保持这些数值语义边界。', '',
        '`weight_load_vec1/2` 只改变权重加载；`packed_compute_vec1/2` 联动权重、零点、展开计算；'
        '`output_vec2/4` 与 `output_lane_rows2` 改输出 encoding；`scale_load_output_encoding` 改 scale 的加载分布。'
        '这些干预在实际 SASS 上不是等价改名：G0 有 17 个 BAR 位点，packed_compute_vec1 为 1 个，REG 从 72 降至 40；'
        '但 LDG 向更窄指令变化，不能提前断言净收益。只改 weight vec1 则增加到 94 REG、18 个 BAR，并引入额外 shared 转换。', '',
        '输出 vec2/vec4 分别生成 32 个 STG.E / 16 个 STG.E.64 位点，默认是 8 个 STG.E.128。'
        '逻辑输出字节数相同，动态请求、sector、寄存器/驻留与同步暴露程度需要实测配合解释。'
        '直接用 scale 输出编码减少了默认 scale 端的转换，但也改变其线程复制与请求分组。', '',
        '## N03a：FlashAttention interleaved RoPE', '',
        '输入 `[head,token,dim]`、cos/sin 的二维广播、偶/奇 pair 的 reshape/split 与固定输出布局形成不同需求。'
        '`input_vec*` 只改输入，`paired_vec*` 同时改输入与 pair 前的计算布局；trig 和输出布局保持冻结。', '',
        'D=64 的 G0 已有 trig 分布转换：1 KiB shared、3 个 BAR 位点、26 REG。paired_vec4 变为 1 个 BAR、25 REG，'
        '仍有 shared 通信。D=128 的 G0 则没有 shared 指令或 BAR，32 REG；其他较窄候选反而引入 shared/同步。'
        '因此两个 D 不能合并成“RoPE 默认都差”的结论。两种 D 的 input_vec8、paired_vec8 均与 G0 full-CUBIN 完全相同。', '',
        '输入更窄可能增加加载指令及重分布，偶/奇访问的请求分组也可能变化。必须用实际活跃线程与 sector/wavefront 观察，'
        '不能把 vec 名称作为真实内存事务数。', '',
        '## N03b：SGLang 多模态 RoPE', '',
        'Q/K 有不同 head 数，默认 Q 为 vec8，K 在 D=64/128 分别为 vec2/vec4；cos/sin 的三个位置区段分别加载并广播。'
        'Q/K 输出保持原位及原默认编码。改变 Q 或 K 的输入/计算所有权时，trig 和写回消费者也可能需要新的转换。', '',
        'G0 的 8 个 IR convert 都是同类型 identity，SASS 没有 shared/BAR。许多 Q/K 候选引入 shared、SHFL 与 3–15 个 BAR 位点，'
        '这些转换应算进候选的真实代价。已有 12 个候选的 baseline-specific derive_plan 不适用，原因是它假定 trig slice 与 Q/K 输入编码一致；'
        '编译包完整保存了实际 load/store/conversion 类型，不能把这个解析器假设当作候选编译或正确性失败。', '',
        'D=64 的 K_vec2、D=128 的 K_vec4 是默认 K 分布；相应的 QK 干预与只改 Q 的二进制可相同。'
        '所有精确 alias 在下表列出。请求层的待检验点是 trig 区段掩码、跨 head 复制及固定输出的转换；'
        '它与无规约问题有关，但不是输出地址置换类 kernel 的同一种机制。', '',
        '## N04：外部 scale 的 FP8 QKV pack', '',
        'Q/K/V 是 token-major fused projection 的只读视图；Q/K 量化写出，V 还包含 token 置换与转置写出。'
        '本轮只改变 V 输入或中间计算布局，Q/K、外部 scale 与输出格式保持固定。'
        '这是 SM120 来源 kernel 的 H100 移植条件，不代表已测原注意力 backend。', '',
        'G0 已有一次真实 V 转换：1 KiB shared、1 个 BAR、71 REG。候选没有完全消除 shared；'
        'REG 为 69–80，BAR 为 1–7。v_compute_input 把矩阵式 shared 路径换为 32 个 STS.U8 与 8 个 LDS 位点，'
        '不能把“与输入一致”理解为整条输出链免费。v_compute_token_vec8/32 均有额外转换，且静态指令计数相同并不等于 full-CUBIN alias。', '',
        '待检验的是输入的连续 D 访问与 V 转置输出之间的请求分组、FP8 packing、shared wavefront 及同步成本。'
        '全部候选的全局写格式固定且静态 STG.E.128 位点均为 6，但参与线程与地址集合仍须以硬件计数确认。', '',
        '## N05：FLA KDA forward gate（独立逐元素子图）', '',
        '这里只包含带 A/bias 的 exp/softplus 逐元素 gate 与 FP32 写出，没有后续 chunk-cumsum、scan、规约或 backward。'
        'BT/W/stages 由 U-only 原始配置菜单预先选择，布局候选不改变这些参数。'
        'BF16 输入的默认加载分布与 FP32 计算/输出分布不同，bias 广播也参与需求。', '',
        '两个 D 的 G0 都有 1 KiB shared 与 1 个 BAR。feature_vec4 联动加载和计算后，两个 D 的 shared 和 BAR 均为 0；'
        'REG 从 30→25（D64）、48→36（D128）。这是不含规约/TMA 的实际转换消除证据，尚不是性能收益证据。'
        '其他 feature/token 候选会引入不同的窄加载、shared 与 SHFL 路径。MUFU/SFU 与长依赖也可能成为瓶颈，'
        '不能只看通信数就解释净时间。', '',
        '## N06：HND KV cache scatter 对照', '',
        'slot 是每 token 的 block-uniform 运行时索引；负 slot 整 CTA 返回。输入及 HND 输出都保留连续 D 维，'
        'head 分布和写入目标位置改变请求分组，但不像 N01 那样必然有输入/输出所有权置换。', '',
        '所有 N06 候选的 IR convert、shared 指令与 BAR 均为 0；这是有用的无显式通信对照。'
        'vec 改变 LDG/STG 宽度、数量与寄存器（G0 18/20，GL 为 18–48/22–48）；'
        '这里若出现差异，应优先核查请求分组、活跃 warp 和资源约束，而不是套用 shared 转换故事。'
        '连续 D 的合法合并访问不保证所有候选请求完全相同，也不支持事先认定存在默认布局缺陷。', '',
        '## 精确 full-CUBIN alias（同 case）', '',
        table(['case','同一 full CUBIN 的逻辑名称'],
              [[x['geometry'],', '.join('`'+v+'`' for v in x['variants'])]
               for x in data['within_case_aliases']]), '',
        '## 逐程序静态资源与指令位点', '',
        'R=每线程 REG；SM=静态 shared 字节；CVT=IR convert 总数/非 identity 数；'
        'GLD/GST=全局加载/写入位点；SLD/SST=shared 加载/写入位点；BAR/SHFL=对应静态位点。'
        '以下只列 G0 与 GL；U 的完整字段保留在 JSON。'
        'ideal-q 仅为上文定义的宽松资源天花板，禁止作为 CUDA occupancy 或实测驻留。', '',
    ]
    for identity, values in by_case.items():
        lines.extend(['### '+identity, '', table(
            ['variant','R','SM','CVT','GLD/GST','SLD/SST','BAR/SHFL','ideal-q'],
            [[x['variant'],x['resources']['REG'],x['resources']['SHARED'],
              str(x['IR_convert_sites']['total'])+'/'+str(x['IR_convert_sites']['non_identity']),
              str(x['static_SASS_sites']['global_load'])+'/'+str(x['static_SASS_sites']['global_store']),
              str(x['static_SASS_sites']['shared_load'])+'/'+str(x['static_SASS_sites']['shared_store']),
              str(x['static_SASS_sites']['CTA_barrier'])+'/'+str(x['static_SASS_sites']['shuffle']),
              x['ideal_register_warp_q_ceiling']]
             for x in values if x['variant']!='U']), ''])
    return '\n'.join(lines)


def run(stage):
    stage = Path(stage)
    source = stage / 'compile/attempt_001/invocation_-01/export'
    input_result = source / 'result.json'
    original_digest = c.sha(input_result.read_bytes())
    result = c.read(input_result)
    assert result['status'] == 'OK' and not result['GPU_allocated']
    programs = [summarize(program, source) for program in result['programs']]
    assert all(row['resources']['LOCAL'] == row['resources']['STACK'] == 0
               and row['static_SASS_sites']['local_memory'] == 0 for row in programs)
    aliases = defaultdict(list)
    for row in programs:
        aliases[row['geometry'],row['cubin_SHA256']].append(row['variant'])
    summary = dict(status='OK', role='Timing-blind derived static interpretation',
        input_result_SHA256=original_digest,
        input_protocol_SHA256=result['protocol_SHA256'],
        counts=dict(cases=len({row['geometry'] for row in programs}), programs=len(programs),
                    GL_candidates=sum(row['variant'].startswith('GL_') for row in programs),
                    unique_geometry_cubins=len(aliases),
                    global_binary_SHA256_count=len({row['cubin_SHA256'] for row in programs})),
        occupancy_scope='ideal upper ceilings only; no CUDA occupancy query or runtime observation',
        documentation=dict(
            Hopper='https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy',
            CUDA_occupancy='https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html',
            NCU='https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference'),
        within_case_aliases=[dict(geometry=key[0],cubin_SHA256=key[1],variants=value)
                             for key,value in aliases.items() if len(value)>1],
        programs=programs)
    assert summary['counts'] == dict(cases=14,programs=138,GL_candidates=110,
                                     unique_geometry_cubins=128,global_binary_SHA256_count=108)
    assert c.sha(input_result.read_bytes()) == original_digest
    c.write_once(stage / 'STATIC_REVIEW.json', summary)
    c.write_once(stage / 'STATIC_REVIEW.md', describe(summary).encode())
    print('Static review OK:', summary['counts'])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',default=str(c.OUT/'stage_n2_compile_r2'))
    run(parser.parse_args().stage)
