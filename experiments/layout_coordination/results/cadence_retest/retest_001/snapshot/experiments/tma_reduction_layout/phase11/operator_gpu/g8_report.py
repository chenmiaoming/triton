"""Regenerate G8 derived Markdown tables from completed evidence; never run GPUs."""
from pathlib import Path
from collections import defaultdict,Counter
import json
from . import common as c


def counter_section(folder):
    path = folder/'counter_analysis.json'
    if not path.exists():
        return ['', '## 6. NCU 与 PC 归因（等待完整离线核验）', '',
                '原计划 167 份 NCU 报告保持不动；另有 6 份在正式 timing 后冻结配置的探索性诊断。最终版本须在 173 份报告齐全并通过本地重导入和当前 CUBIN 的 PC 对应检查后生成计数结论。']
    data = c.read(path)
    assert data['status'] == 'OK' and data['report_count'] == 173
    assert len(data['local_reimports']) == 173
    rr = data['rows']; index = {}
    for row in rr:
        for tag in [row['tag']]+row.get('logical_aliases_for_this_mode', []):
            key = (tag, row['mode'])
            assert key not in index, key
            index[key] = row
    def get(tag, mode='software'):
        return index[(tag,mode)]
    def numbers(tag):
        row = get(tag); counts = row['counts_per_CTA']; attr = row['resources']
        return (attr['n_regs'],attr['occupancy_capacity_q'],
                '/'.join(f'{counts[k]:g}' for k in ['shuffle','shared_load','shared_store','barrier']))
    def table_row(geometry, variant, tag):
        regs, q, counts = numbers(tag)
        warm = get(tag,'warm_application')['metrics_per_CTA']
        sectors = warm['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
        wavefronts = warm['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']
        return f'| {geometry} / `{variant}` | {regs} / {q} | {counts} | {sectors:.2f} | {wavefronts:.2f} |'
    modes = Counter(row['mode'] for row in rr)
    paired = []
    for geometry in ['C06_s16_d128','C06_s32_d128','C06_s32_d256','C02_n64_d64','C02_n64_d128','C02_n64_d256','C01_p256_n1024','C01_p128_n4096','C01_p96_n8192']:
        group = geometry if geometry.startswith('C01') else 'g2_'+geometry
        u,g = [get(group+':'+label) for label in ['upstream','gluon_anchor']]
        assert u['counts_per_CTA'] == g['counts_per_CTA']
        paired.append(geometry)
    section = ['', '## 6. NCU：减少通信不保证完整程序更快', '',
        f"共 **{data['report_count']} 份报告**：原计划 167 份，另有 6 份 post-timing 探索性诊断（为新观察到的 `s16/d128`、`s32/d128` 的 `vec4:within` 补计数）。三种模式分别为 software {modes['software']}、warm application {modes['warm_application']}、cold isolated {modes['cold_isolated']}。全部已在本地从 `.ncu-rep` 重导入；software 报告的执行 PC 已与各自当前 CUBIN 的反汇编和 debug source 记录逐项匹配，未沿用旧报告硬编码的源码行号。见 [counter_analysis.json](counter_analysis.json)。",
        '', f'{len(paired)} 对 U/G0 的 software 指令族计数逐项相同，进一步支持新基线对齐。硬件 wavefront、bank conflict 等计数来自分别运行的报告，少量浮动不应被解释成等价 U/G0 的源程序差异。', '',
        '下表 `SHFL/LDS/STS/BAR` 是 software 的动态 warp 指令次数/CTA；global-load sectors 与 shared-load wavefronts 是 warm application 的实际网格总和除以 CTA 数。两者来自不同 replay 条件，列在一起用于检查程序结构和请求变化，不能相加作时间成本。`q` 是 driver 查询的容量上限，不是实际达到的 occupancy。', '',
        '| vLLM 几何 / 候选 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |',
        '| --- | ---: | ---: | ---: | ---: |']
    for geometry, variants in [
        ('C06_s16_d128',['upstream','vec1:n_first','vec2:within','vec4:within']),
        ('C06_s32_d128',['upstream','vec1:n_first','vec4:within']),
        ('C06_s32_d256',['upstream','vec4:within'])]:
        for variant in variants:
            section.append(table_row(geometry.removeprefix('C06_'),variant,'g2_'+geometry+':'+variant))
    section += ['', '`s16/d128` 与 `s32/d128` 的新 `vec4:within` 诊断保留了 U 的寄存器、容量 q 和 global-load sector 请求，却把 BAR 从 20 减为 0、warm shared-load wavefront 从 36 减为 4。`s32/d256` 的 vec4:within 也保留了这些资源/请求量并消除 shared-memory 往返，但 SHFL 从 56 增为 64；并非所有通信计数都同时下降。这些 post-timing 诊断解释了为何需要检查新赢家的完整代码结构，不能沿用旧 vec1 的解释。']
    section += ['',
        '`s16/d128` 的 vec1 确实删掉了表中的 shared-memory 通信，但寄存器从 32 升到 56、容量 q 从 16 降到 9，warm global-load sector 请求也增加；`s32/d128` 的 vec1 也在删通信的同时把寄存器从 56 升到 96、q 从 9 降到 5。它们说明“通信少了”与“完整 kernel 快了”不是同一个判断。这里尚未通过独立干预把寄存器、加载请求与依赖链各自造成的时间份额辨识出来，不能把任何一个计数直接乘权重解释全部退化。', '',
        '| SGLang 几何 / 干预 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |',
        '| --- | ---: | ---: | ---: | ---: |']
    for geometry in ['C02_n64_d64','C02_n64_d128']:
        for variant in ['upstream','vec1:n_first']:
            section.append(table_row(geometry.removeprefix('C02_'),variant,'g2_'+geometry+':'+variant))
        for variant in ['qk_only','coordinated']:
            section.append(table_row(geometry.removeprefix('C02_'),variant,'g4a_'+geometry+':'+variant))
    section += ['',
        '例如 `d64` 的 vec1 把 SHFL 从 1,664 降到 672，却把 LDS 从 132 增到 896、STS 从 136 增到 224、BAR 从 264 增到 448，且寄存器 56→79、q 9→6。这与其完整程序退化相伴，足以否定“只看归约 shuffle/fanout 就能预测该图收益”的解释；它尚不是这些变化各自贡献多少 ns 的因果分解。PC→source 表保留了实际中间转换、归约和消费者处的指令，不再用旧源图的行号给新图归因。', '',
        '| LayerNorm merge 几何 / 候选 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |',
        '| --- | ---: | ---: | ---: | ---: |']
    for geometry, variant in [('C01_p256_n1024','vec4:n_first'),('C01_p128_n4096','vec4:n_first'),('C01_p96_n8192','vec4:within')]:
        for label in ['upstream',variant]:
            section.append(table_row(geometry.removeprefix('C01_'),label,geometry+':'+label))
    section += ['', 'LayerNorm 的表格同时给出另一个边界：vec4 候选虽然把容量 q 从 5 降到 4，merge 仍更快。因此 q 降档本身也不足以判定完整程序会变慢；本轮未构造把各因素独立分离的因果时间模型。']
    assert all(row['resources']['local_bytes'] == 0 for row in rr)
    assert all(row['counts_per_CTA']['local_load'] == row['counts_per_CTA']['local_store'] == 0 for row in rr if row['mode'] == 'software')
    section += ['', '173 份 profile 绑定的 local-memory 分配均为 0，69 份 software 报告也未执行 LDL/STL；不能把这里的寄存器增加或退化直接称为 spill。这一结论仅覆盖本轮实际 profile 的程序。']
    warm = [row['metrics'].get('dram__throughput.avg.pct_of_peak_sustained_elapsed') for row in rr
            if row['mode']=='warm_application' and row['geometry'].startswith('C06')]
    warm = [value for value in warm if value is not None]
    section += ['',
        '计数边界必须保留：inference software 使用 batch1/grid1，warm/cold 使用 batch8（SGLang context2048、vLLM length8192），没有 batch64 的独立 NCU。本轮 LN profiles 只覆盖 standalone merge（M=P），不能直接分解完整 backward，尤其不能直接套到 M128 pipeline 中 P 被截为 min(P,M) 的 merge。', '',
        f"vLLM warm 报告的 DRAM 利用率仅 {min(warm):.2f}%～{max(warm):.2f}%；这些合成 batch8 工作集可落在 L2，不能照搬此前大 TMA 图的 HBM roofline 解释。L1TEX global sector 请求数不等于 DRAM 传输字节或 DRAM transaction 次数。",
        '', 'software patch 计数、warm application replay 与 cold isolated replay 分开保留，正式时间来自无 profiler 的 CUDA Graph 测量。NVIDIA 文档说明软件 patch 会引入运行开销，kernel replay 默认刷新缓存，而 application replay 配合关闭 cache control 可保留应用自己的预热状态；因此不能把这些不同条件的时延混成同一个性能估计。[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)']
    return section


def main():
    P = c.OUT/'stage_g8'
    a=json.loads((P/'timing_analysis.json').read_text()); rows=a['rows']; history=json.loads((P/'historical_gain_directions.json').read_text()); proto=json.loads((P/'preregistration/protocol.json').read_text())
    actual_audit = c.read(P/'actual_abi_audit/verification.json')
    assert actual_audit['status'] == 'OK' and actual_audit['reuse_G8_evidence_supported']
    assert len(actual_audit['records']) == 4 and len(actual_audit['upstream_reference_pairs']) == 2
    by=defaultdict(list)
    for r in rows:by[(r['panel'],r['group'],r['batch'],r['length'],r['mode'],r['M'])].append(r)
    def f(x,n=2):return f'{x:+.{n}f}'
    def ci(r):return f"{f(r['gain_vs_U_percent'])}% [{f(r['gain_vs_U_CI95'][0])}, {f(r['gain_vs_U_CI95'][1])}]"
    def best(rs):return max([r for r in rs if r['variant'].split(':')[-1] not in ['upstream','gluon_anchor'] and not r['variant'].endswith(':native')],key=lambda r:r['gain_vs_U_percent'])
    def find(panel,geom,batch,length,mode,M,variant):return next(r for r in rows if (r['panel'],r['geometry'],r['batch'],r['length'],r['mode'],r['M'],r['variant'])==(panel,geom,batch,length,mode,M,variant))
    anchors=[r for r in rows if r['variant'].endswith('gluon_anchor')]
    old={(r['tag'],r['batch'],r['length'],r['mode'],r['M']):r for r in history['rows']}
    # Complete workload table, retaining both output diagnostics and worst candidates.
    m=['# G8 完整 workload 摘要','', '数值均为同轮 U 归一化收益 `100×(U−candidate)/U`；正值表示用时降低。最优候选仅为该已测矩阵内的事后最优，不能解释为已验证的静态选择器。CI 为探索性逐点 95% 区间。全部 517 行及 worker/visit 数据见 [all_timings.csv](all_timings.csv) 与 [timing_analysis.json](timing_analysis.json)。','', '| panel / workload | U (µs) | G0 收益 | 实测最优候选（保持原输出） | 收益 [CI95] | 最差候选（含 native 诊断） | 收益 |','| --- | ---: | ---: | --- | --- | --- | ---: |']
    for key,rs in sorted(by.items(),key=lambda it:str(it[0])):
     panel,g,b,n,mode,M=key
     labels=['actual','B4'] if panel=='g4b' else ['default']
     for label in labels:
      sub=[r for r in rs if r['comparison_label']==label];u=next(r for r in sub if r['variant'].endswith('upstream'));g0=next(r for r in sub if r['variant'].endswith('gluon_anchor'));bb=best(sub);ww=min([r for r in sub if r['variant'].split(':')[-1] not in ['upstream','gluon_anchor']],key=lambda r:r['gain_vs_U_percent']);work=f'{g} / b{b}, n{n}, {mode}' if b is not None else f'{g} / {mode}, M={M}'
      if panel=='g4b':work+=' / '+label
      m.append(f"| `{panel}` {work} | {u['time_us']:.3f} | {f(g0['gain_vs_U_percent'])}% | `{bb['variant']}` | {ci(bb)} | `{ww['variant']}` | {f(ww['gain_vs_U_percent'])}% |")
    (P/'MATRIX.md').write_text('\n'.join(m)+'\n')
    r=['# Stage G8：修正生产 Gluon 基线后的完整重测','', '本轮重测改变了此前生产算子的结论：**源码对齐的 G0 已与原始 Triton U 对齐，但旧 Gluon 程序上的候选收益不能沿用。** SGLang 两个较大 head dimension 的既有收益未在新程序上保留；vLLM 仍有小幅收益，较好的候选转为 `vec2/vec4:within`；LayerNorm 梯度合并仍存在明显收益。与此同时，减小 vector width 能造成很大的退化，不能从 TMA/max 的结论推广出“普遍选 vec1”。','', '本阶段只替换受影响的 G2/G3a/G3b/G4a/G4b 生产算子比较。独立 TMA、D1、M2–M4、A10 证据不因本次缺陷重跑；它们各自的证据边界继续保留。旧样本、原始编译产物与旧报告均不覆盖，逐项传播审计见 [IMPACT_AUDIT.md](IMPACT_AUDIT.md) 与 [impact_audit.json](impact_audit.json)。','', '## 1. 冻结、编译和测量口径','', '- 使用生产来源 kernel 的合成输入复现与冻结设备 pipeline；未运行整个 vLLM/SGLang 框架或真实 LLM 模型，不能将子核收益写成在线服务吞吐收益。','- 固定编译器 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`，标准编译流水线，无 C++/native 修改。复用既有 native/build cache；CPU 编译前后 ccache 计数相同，native rebuild 为 0。','- 新编译 62 个唯一 Gluon 程序，复用 19 个经 SHA 校验的 U/stage2/dX 程序，组成 168 个逻辑编译记录。全部 156 个非 helper 的有效输入/中间/输出布局检查通过；31 对 G0/U 的 ELF allocated sections、ABI 与 launch resource 配置一致。','- 正确性门包含 798 项 inference 检查与 270 项训练检查，全部通过。包括完整输出、输入保留、canary、三种输入模式；训练 merge 参考为精确 BF16 partial 的 FP64 求和后舍入，完整 backward 的 dX 用独立公式核对，最终梯度按实际生成的 partial 核对。它不要求不同浮点累加顺序逐位相同。','- 517 个逻辑计时条件：G2 294、G3b 99、G4a 40、G4b 84。每个 worker 实测 439 个物理条件；3 个 H100 worker × 5 次随机次序 visit × 31 个样本，共 **204,135 个原始时间样本**。精确 CUBIN/ABI 别名仅共享一次物理采样，不重复统计；逻辑展开后的等价样本数为 240,405。','- 正式计时仅用 CUDA driver 加载冻结 CUBIN，GPU 进程无 Triton import/编译，所有 driver 返回码为 0。NCU 在正式计时结束后单独采集。','- 主估计为 15 个 worker/visit 中位数的均值；4,096 次配对分层 bootstrap。收益是 `100×(U−candidate)/U`，均使用本轮同 workload 的 U。负值表示候选耗时增加，例如 −100% 表示约 2 倍用时。CI 为逐点探索性区间，未做多重比较修正。下表的“最优”是事后描述，不是盲测选择器。','', f"78 个 G0/U workload 对照的点估计范围为 **{f(min(x['gain_vs_U_percent'] for x in anchors),3)}%～{f(max(x['gain_vs_U_percent'] for x in anchors),3)}%**，全部 CI95 位于 ±3% 内（所有区间端点范围 {f(min(x['gain_vs_U_CI95'][0] for x in anchors),3)}%～{f(max(x['gain_vs_U_CI95'][1] for x in anchors),3)}%）。这与 G6/G7 的源码、机器代码及实际绑定门禁一致，新的默认基线没有复现旧的 U/G0 偏差。",'', '协议与全量数据：[preregistration/protocol.json](preregistration/protocol.json)、[compile_gate.json](compile_gate.json)、[timing_analysis.json](timing_analysis.json)、[all_timings.csv](all_timings.csv)、[完整 workload 摘要](MATRIX.md)。','', '## 2. SGLang：旧收益不再成立，部分布局干预显著恶化','', 'G2 覆盖 batch 1/8、context 256/2048，并同时测 stage1 与含原 stage2 的 decode。对 `d128/d256`，16 个相关 workload/模式中的最好候选均接近原默认 `vec8:n_first`；没有出现 >3% 的收益。`d64` 仅在短 context、batch 8 有小幅变化，长 context 也未出现实用收益。这里的“不成立”限定在本轮新源图的完整矩阵，不能解释为所有 SGLang shape 或所有硬件均无布局优化空间。','', '| head dim | batch / context | stage1 最优候选与收益 [CI95] | decode 最优候选与收益 [CI95] |','| --- | --- | --- | --- |']
    for geom in ['C02_n64_d128','C02_n64_d256','C02_n64_d64']:
     for b,n in [(1,256),(1,2048),(8,256),(8,2048)]:
      vals=[]
      for mode in ['stage1','decode']:
       bb=best([x for x in rows if x['panel']=='g2' and x['geometry']==geom and x['batch']==b and x['length']==n and x['mode']==mode]);vals.append(f"`{bb['variant']}` {ci(bb)}")
      r.append(f"| {geom.split('_')[-1]} | {b} / {n} | {' | '.join(vals)} |")
    r += ['', 'G4a 的 PV-only、QK-only 与协同干预也必须按新图重读。batch 8、context 2048 的 stage1 收益如下；负向 interaction 仅说明协同干预比简单相加损失小，不表示协同程序比 U 更快。全部 8 个 factorial interaction 与 CI 见 `timing_analysis.json` 的 `contrasts`。','', '| head dim | PV-only | QK-only | coordinated |','| --- | --- | --- | --- |']
    for geom in ['C02_n64_d128','C02_n64_d64']:
     rr=[find('g4a',geom,8,2048,'stage1',None,v) for v in ['pv_only','qk_only','coordinated']];r.append('| '+geom.split('_')[-1]+' | '+' | '.join(ci(x) for x in rr)+' |')
    r += ['', '尤其是 `d64 / b8 / context2048 / qk_only`，耗时增加 **152.43%**（约 2.52 倍 U），协同干预仍增加 146.86%。中间布局/转换的资源和指令归因需与本轮 NCU 对应，不能仅凭 fanout 下降或旧 PC 行号解释。','', '## 3. vLLM：保留部分收益，但 vec1 的旧结论反转','', '三个 split/head-dim 几何均保留 batch 1/8/64。下表列出原输出语义下的矩阵最优，以及同一 workload 的 `vec1:n_first`，防止只展示 batch 8 的获益点。','', '| splits / dim | batch | 最优候选 | 收益 [CI95] | vec1:n_first 收益 [CI95] |','| --- | ---: | --- | --- | --- |']
    for geom in ['C06_s16_d128','C06_s32_d128','C06_s32_d256']:
     for b in [1,8,64]:
      bb=best([x for x in rows if x['panel']=='g2' and x['geometry']==geom and x['batch']==b]);v=find('g2',geom,b,8192,'merge',None,'vec1:n_first');r.append(f"| {geom.removeprefix('C06_')} | {b} | `{bb['variant']}` | {ci(bb)} | {ci(v)} |")
    r += ['', '`s32/d256/b8` 的最优点估计为 +3.85%，但 CI 下界为 +2.88%，因此不能称为“已解析出超过 3% 门槛的收益”。`s32/d128/b64` 的 vec1 耗时增加 74.74%，而 vec4:within 降低 7.24%；对这一点不能把低通信 fanout 等同于低完整程序成本。','', 'G4b 保留原 actual ABI 与 B4 replay 对照。SGLang 对应实际条件在新程序上两者一致，target 没有隐藏收益。vLLM 的 batch 1 specialization 仍能影响结果：`s32/d256/b1` 的 target 在 actual 下 +5.84% [5.34, 6.34]，B4 下 +3.80% [3.15, 4.58]；两者不能混为一个结论。`s16/d128/b64` 的旧 target 在 actual 下为 **−45.10%**，而新矩阵较好的布局是 vec4:within，不能只复测旧赢家。42 个 actual/B4 配对差分见 `contrasts`。','', '对 G4b batch64 的额外 JIT 属性差异，另做了纯 CPU 审计：把 G7 真实 JIT 观察到的 `num_seqs % 16 == 0` 属性带入 `s16/d128`、`s32/d256` 的 G0 与 target 共 4 个程序。4/4 的完整 CUBIN、allocated sections、ABI 和编译资源与 G8 generic 形式相同，另外 2 对 U 也与 G7 真实 JIT 的执行字段一致。见 [actual_abi_audit/analysis.json](actual_abi_audit/analysis.json)。这支持复用本轮冻结二进制的 batch64 测量，但不是新增的候选实际 JIT 调用或 GPU 样本。原审计脚本写出分析后在打印摘要时出现缺失 `json` import 的错误，保留了 trace；独立离线 verifier 重新核验 4 个 Gluon 与 2 个 U 比较，未重新编译或运行 GPU。复现细节见 `REPRODUCE.md`。','', '## 4. LayerNorm backward：合并子核收益保留，完整图收益随占比缩小','', '训练候选包含每个几何全部 7 个合法 layout，以及 vec1/vec8 的 native-output 诊断。下表最优候选均保持原输出语义，不把 native-output 诊断程序当成生产优化。','', '| partial rows / N | merge 最优 / 收益 [CI95] | full backward M128 | full backward M2048 |','| --- | --- | --- | --- |']
    for geom in ['C01_p256_n1024','C01_p128_n4096','C01_p96_n8192']:
     vals=[]
     for mode,M in [('merge',128),('pipeline',128),('pipeline',2048)]:
      bb=best([x for x in rows if x['panel']=='g3b' and x['geometry']==geom and x['mode']==mode and x['M']==M]);vals.append(f"`{bb['variant']}` {ci(bb)}")
     r.append(f"| {geom.removeprefix('C01_')} | {' | '.join(vals)} |")
    r += ['', '完整 backward 指原 dX/stage1 + 新 merge 的设备图，并含三个清零节点；因此 merge 的 15.75%～27.88% 不能直接写成端到端 backward 的收益。M2048 时三几何的实测最优分别为 +11.36%、+4.56%、+2.26%；最后一个 CI 位于 3% 门槛之下。','', '## 5. 旧结论的保留与撤回','', '以下数值均是各自实验内部、对各自同轮 U 归一化的收益。旧图和新图不是只有一个受控变量的同会话实验，因此本表只说明旧证据能否作为修正后布局结论继续引用，不把两列差值解释成源码修正的独立因果贡献。全量 517 条 lineage 见 [historical_gain_directions.json](historical_gain_directions.json)。','', '| 条件 / 原候选 | 旧程序内部收益 | 新程序内部收益 | 证据处理 |','| --- | ---: | ---: | --- |']
    examples=[('g2','C02_n64_d256',8,2048,'stage1',None,'vec4:n_first','撤回旧 +7.58% 布局收益'),('g2','C06_s16_d128',8,8192,'merge',None,'vec1:n_first','旧 vec1 获益不保留'),('g2','C06_s16_d128',64,8192,'merge',None,'vec1:n_first','明显方向反转'),('g2','C06_s32_d128',64,8192,'merge',None,'vec1:n_first','明显方向反转'),('g2','C06_s32_d256',8,8192,'merge',None,'vec4:within','部分收益保留，仍需保留 CI 边界')]
    for panel,geom,b,n,mode,M,v,note in examples:
     x=find(panel,geom,b,n,mode,M,v);h=old[(x['tag'],b,n,mode,M)]['comparisons']['U'];r.append(f"| `{geom}` b{b}, {mode}, `{v}` | {f(h['old_internal_gain_percent'])}% | {f(x['gain_vs_U_percent'])}% | {note} |")
    r += ['', 'LayerNorm 的“收益保留”指优化空间仍在，不表示旧最优布局排序都得到确认。以下先按旧实验的点估计选定旧最优候选，再查该同一候选的新结果；不把新最优与旧最优直接当成同一个程序。候选间的点估计排序也不构成经多重比较验证的排名。','', '| LN 几何 / 范围 | 旧点估计最优候选 | 旧程序内部收益 | 同一候选的新收益 [CI95] |','| --- | --- | ---: | --- |']
    legacy_training = c.read(c.OUT/'stage_g3b/timing_analysis.json')['rows']
    for geometry in ['C01_p256_n1024','C01_p128_n4096','C01_p96_n8192']:
        for mode, m in [('merge',128),('pipeline',2048)]:
            eligible = [row for row in legacy_training if row['geometry'] == geometry and row['mode'] == mode and row['M'] == m and ':vec' in row['tag'] and not row['tag'].endswith(':native')]
            previous = max(eligible,key=lambda row:row['gain_vs_U_percent'])
            variant = previous['tag'].removeprefix(geometry+':')
            current = find('g3b',geometry,None,None,mode,m,variant)
            r.append(f"| `{geometry}`, {mode} M{m} | `{variant}` | {f(previous['gain_vs_U_percent'])}% | {ci(current)} |")
    r += ['', '例如 `p96/n8192` 的旧 merge 最优点为 `vec2:within`（+16.07%），同一候选在新程序上为 +14.44%；新矩阵最优点则是 `vec4:within`（+15.75%）。`p128/n4096` 的旧 merge 最优点为 `vec4:within`，新最优点为 `vec4:n_first`。这些差别应连同完整矩阵和 CI 保留，不能仅写一句“旧赢家已确认”。']
    r += ['', '本轮不据此更新 C3.4，也不将生产算子重测冒充 D1 的新盲测。TMA/max、消费者机制实验与这些生产程序的布局变化不是同一测量图。']
    r += counter_section(P)
    r += ['', '## 7. 复现与交付校验', '', '执行顺序、缓存复用、补充 actual-attribute 审计与解释边界见 [REPRODUCE.md](REPRODUCE.md)。本轮正式 timing 不做按结果决定的重复采样。旧证据受冻结协议中 5,213 个保护 SHA 约束。报告与派生矩阵由 `g8_report.py` 从冻结 JSON 再生；最终独立校验记录见 [final_validation.json](final_validation.json)。', '', '本轮在受影响的生产程序重测范围内结束，不继续启动下一阶段或提交 PR。']
    text='\n'.join(r)+'\n';(P/'REPORT.md').write_text(text)


if __name__ == "__main__":
    main()
