"""M3 hypothesis tests, including counterexamples, without selector refitting."""
from collections import defaultdict
import io
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.m3 import common as c


def save(fig,name):
    for ext in ('png','pdf'):
        buffer=io.BytesIO()
        fig.savefig(buffer,format=ext,dpi=180,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
        c.derived(c.OUT/'figures'/f'{name}.{ext}',buffer.getvalue())


def plots(diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,(m,n,w) in zip(axes,[(32,64,8),(32,128,4)]):
        for placement,style in [('n_first','o-'),('within','s--')]:
            rows=[next(r for r in diag['layout_tests'] if r['layout_key']==f'M{m}_N{n}_w{w}:vec{v}:{placement}') for v in (2,4,8)]
            ys=[r['hardware'][r['layout_key']+':native:ONE_FULL']['sectors_per_CTA']/r['hardware'][r['layout_key']+':fixed:ONE_FULL']['sectors_per_CTA'] for r in rows]
            ax.plot([2,4,8],ys,style,label=placement)
        ax.plot([2,4,8],[1,1,2],':',color='black',label='Frozen request prediction')
        ax.set_xscale('log',base=2);ax.set_xticks([2,4,8],['2','4','8'])
        ax.set_xlabel('Input layout vector width')
        ax.set_ylabel('Native/fixed store-sector requests')
        ax.set_title(f'{m}x{n}/w{w}')
        ax.set_ylim(.8,2.2);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,'store_sector_threshold');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    xs=np.arange(len(diag['consumer_tests']))
    values=[[r['native_LDS'] for r in diag['consumer_tests']],
            [r['expectation']['predicted_broadcast_whole_LDS'] for r in diag['consumer_tests']],
            [r['broadcast_LDS'] for r in diag['consumer_tests']]]
    for i,(v,title,color) in enumerate(zip(values,['Native output','Frozen all-row prediction','All-row consumer measured'],['#245b80','#b8b8b8','#9a5420'])):
        ax.bar(xs+(i-1)*.25,v,.24,label=title,color=color)
    labels=[]
    for r in diag['consumer_tests']:
        meta=next(x for x in c.protocol()['cases'] if x['tag']==r['tag'])
        labels.append(f'{meta["M"]}x{meta["N"]}/w{meta["num_warps"]}\nvec{meta["vec"]}, N-first')
    ax.set_xticks(xs,labels);ax.set_ylabel('Dynamic warp LDS instructions / CTA')
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'result_replica_consumer');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,qkind in enumerate(('low','natural')):
        for col,(m,n,w) in enumerate(((32,64,8),(32,128,4))):
            ax=axes[row,col]
            for placement,policy,title,color,style in [('N_first','fixed','N-first fixed','#245b80','o-'),
                    ('N_first','native','N-first native','#245b80','o--'),
                    ('within','fixed','within fixed','#9a5420','s-'),
                    ('within','native','within native','#9a5420','s--')]:
                values=[]
                for v in (2,4,8):
                    r=next(r for r in diag['summary'] if r['geometry']==[m,n,w] and r['vec']==v and (r['q']==1 if qkind=='low' else r['natural']))
                    values.append(r[f'{placement}_{policy}_ns']['median'])
                ax.plot([2,4,8],values,style,label=title,color=color)
            ax.set_xscale('log',base=2);ax.set_xticks([2,4,8],['2','4','8'])
            ax.set_xlabel('Input layout vector width');ax.set_ylabel('FULL service (ns/tile/SM)')
            ax.set_title(f'{m}x{n}/w{w}, '+('q=1' if qkind=='low' else 'natural q'))
            ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8)
    save(fig,'FULL_vector_output_response');plt.close(fig)


def main():
    diag=c.read(c.OUT/'diagnostics.json');p=c.protocol();analysis=c.read(c.OUT/'analysis.json')
    gate=c.read(c.OUT/'artifact_gate.json');ct=c.read(c.OUT/'counter_gate.json')
    plots(diag)
    lines=['# Stage M3 — 输出转换阈值、写出请求与结果复制消费者', '',
        '本轮针对 M2 新观察提出三个可证伪假说，并在接触 M3 编译/计时前冻结预测。没有更新选择器、重选 η、改变 D1 记分，或修改任何旧证据。', '',
        f'结果：转换策略匹配 **{diag["conversion_matches"]}/12**；fixed/native sector 请求匹配 **{diag["sector_matches"]}/24**；全行消费的 LDS 预测匹配 **{diag["consumer_matches"]}/4**。其中 vec2/8 是已有 M2 锚点；新增 vec4 的前向检验为转换 **4/4**、sector **8/8**。四个全行消费者是新增程序。不能把全部 12/24 个锚点加补点当作完全未见检验。', '',
        '## 1. 设计与原始证据', '',
        '- 两个已有开发几何：32×64/w8、32×128/w4；vec2/4/8，N-first/within，fixed/native 输出，共 24 个 FULL 程序。新增 vec4、复用 M2 的 vec2/8；另含四个旧 vec4 固定输出桥接和两个原 T_control，共 30 个计时程序。',
        '- 另有四个不同语义的诊断消费者：把每列 max 加上逻辑行号，写出 BM×BN 元素。它们只在 grid=1 下采集计数，不进入 FULL 计时、收益或模型评估。所有变化均是有效张量与布局，没有隐藏 lane/warp scalar。',
        f'- 共 34 个逻辑程序，12 个 CPU 编译、22 个 CUBIN 复用，{gate["checks"]} 项完整输出/尾部 canary 检查通过。原编译器 `{c.PIN}` 与 native SHA 不变，复用缓存，无原生重编。',
        f'- 三个 H100 worker，{analysis["event_samples"]:,} 个正式事件样本，主指标仍为 128/256/384 MiB FULL 的 grid/132 服务斜率（ns/tile/SM）。另有 {len(ct["reports"])} 份 NCU 报告并全部本地复读。', '',
        '## 2. H1：warp 内可达，还需满足 shuffle 分解门限', '',
        '受限的静态实现按 contiguous-N Blocked/Slice 的线性位基，投影由 warp 选择的 source lane 位、填充复制零基，再遍历寄存器/lane 置换循环。对应固定编译器 getWarpLayoutConvertDecomposition 与 cvtNeedsWarpShuffle 的规则：混合交换数 <2 时才自动选 shuffle。这个实现仅覆盖本轮布局族。', '',
        '| 布局 | 最小通信范围 | 混合交换数 | 预先预测 | 源行/NCU 观察 | 匹配 |',
        '|---|---|---:|---|---|---|']
    for r in diag['layout_tests']:
        z=r['prediction']
        lines.append(f'| {r["layout_key"]} | {z["minimum_reach"]} | {z["restricted_mixed_transpositions"]} | {z["predicted_fixed_lowering"]} | {r["observed_MO"]} | {r["conversion_prediction_match"]} |')
    lines += ['', '新增 vec4 的四个布局全部走 shared；可在 warp 内完成的三个布局，其受限混合交换数均为 2。阈值是当前编译器的自动策略，不能解释为硬件无法用 shuffle。这是编译器路径预测，不是每种路径的延迟模型；也不能把 static SHFL/LDS 数量直接线性相加为执行时间。', '',
        '## 3. H2：vec4 与 vec8 的 global-store 请求粒度', '',
        '静态预测按每个 warp、每条至多 128-bit 的 FP32 store 枚举访问 sector，跨指令重复访问的同一 sector重复计数。不是只按总输出字节除以 32B，也不是 HBM 写出流量模型。', '',
        '| 布局 | fixed sectors/CTA 预测/实测 | native sectors/CTA 预测/实测 | 实测 native STG 类型与 warp 指令数 |',
        '|---|---:|---:|---|']
    for r in diag['layout_tests']:
        z=r['prediction'];key=r['layout_key'];a=r['hardware'][key+':fixed:ONE_FULL'];b=r['hardware'][key+':native:ONE_FULL']
        lines.append(f'| {key} | {z["predicted_fixed_store_sectors_per_CTA"]}/{a["sectors_per_CTA"]:.3f} | {z["predicted_native_store_sectors_per_CTA"]}/{b["sectors_per_CTA"]:.3f} | {b["store_instructions"]} |')
    lines += ['', 'vec4 的四个新增布局中，native/fixed 的 sector 请求均相同；vec8 仍为 2 倍。比如 32×64 的 native vec4 每个活跃 owner 的四个 FP32 值可由一条 128-bit store 写出，vec8 需要两条，各次请求可覆盖同一 sector。这是 NCU 实测的请求计数，不是推断硬件把加载融合，也不是声称 HBM 写出量翻倍。',
        '', '![写出 sector 请求阈值](figures/store_sector_threshold.png)', '',
        '## 4. H3：更换结果消费者，检验被省去的返回读取', '',
        '四个新增消费者的输入、load layout 和 max 语义相同，但每个逻辑行都需要 max 结果。预计 whole LDS 恢复为 M2 fixed 的 whole LDS 减显式 MO LDS。只比较动态计数与生成结构；输出字节数不同，因此没有原 FULL 性能对比。', '',
        '| N-first 布局 | native LDS | 预先预测全行 LDS | 实测全行 LDS | MAX 计数相同 | SHFL 计数相同 | ML load 计数相同 | regs native/全行 |',
        '|---|---:|---:|---:|---|---|---|---|']
    for r in diag['consumer_tests']:
        lines.append(f'| {r["tag"]} | {r["native_LDS"]} | {r["expectation"]["predicted_broadcast_whole_LDS"]} | {r["broadcast_LDS"]} | {r["same_MAX_count"]} | {r["same_SHFL_count"]} | {r["same_ML_load_count"]} | {r["native_regs"]}/{r["broadcast_regs"]} |')
    lines += ['', '实测 LDS 分别为 17→24、33→40、22→24、29→32，均命中事前预测；四项 MAX、SHFL 和源行可定位 ML load 计数均保持不变。寄存器仍变化，其中 32×128 vec8 从 29→35，容量 q 从 16→12，因此仍不据此推算原 FULL 的返回广播时间。',
        '', '![结果消费者与动态 LDS](figures/result_replica_consumer.png)', '',
        '若计数和上游类别同时匹配，结果支持“消费者参与情况影响 reduction-result 返回读取”，而不是“删除 MO 后的 LDS 全部属于原始 ML”。计数匹配仍不等于独立测得返回广播延迟：不同消费者的排程、寄存器分配和写出均可能改变。', '',
        '## 5. vec2→vec4→vec8 的完整程序响应', '',
        '以下只是本轮同 worker 的完整程序实测，不拟合新模型。三 worker 的配对差分取中位数；保留全部 q、worker 与范围。', '',
        '| 几何 | vec | q | fixed within−NF ns | native within−NF ns | 输出策略交互 ns |',
        '|---|---:|---:|---:|---:|---:|']
    for r in diag['summary']:
        if r['q']!=1 and not r['natural']:continue
        g=r['geometry']
        lines.append(f'| {g[0]}×{g[1]}/w{g[2]} | {r["vec"]} | {r["q"]} | {r["fixed_gap_ns"]["median"]:.3f} | {r["native_gap_ns"]["median"]:.3f} | {r["interaction_ns"]["median"]:.3f} |')
    lines += ['', '![完整程序的 vector/output 响应](figures/FULL_vector_output_response.png)', '',
        '## 6. 对模型意味着什么', '',
        '- 输出操作计划至少需要：转换的寄存器/lane 置换结构、编译器实际选择的 shuffle/shared 路径、全局 store 的指令分组与 sector 请求、消费者对结果副本的需求。仅有 fanout 或两个 MO 布尔标记不足以表达这些差异。',
        '- 本轮验证的是结构机制及其 FULL 响应，不是校准后的独立阶段延迟，也没有证明一个新的选择器优于 fixed_vec1。M2 的 q=1 方向错误仍是冻结时间模型的反例。',
        '- 后续若更新模型，应先以同语义完整程序的操作计划重做开发集 CV，并按相同输入/消费者边界限定适用域；当前计数不能直接作为一组等权成本。',
        '- 本轮结束后停止，不自动进入跨架构、编译器 Pass 或 PR。', '',
        '## 7. 复现', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' \\',
        '  .venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.analyze services',
        '.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.analyze counters',
        '.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.offline_import',
        '.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.diagnose',
        '.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.report',
        '.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.validate --committed',
        '```','']
    c.derived(c.OUT/'REPORT.md','\n'.join(lines).encode())
    print('M3 report and figures rendered',flush=True)


if __name__=='__main__':main()
