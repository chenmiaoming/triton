"""Render M4 stage evidence without changing any model or preregistered result."""
from collections import defaultdict
import csv
import io
import statistics as st
from . import common as c


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=c.protocol();cg=c.read(c.OUT/'compile_gate.json');ag=c.read(c.OUT/'artifact_gate.json')
    a=c.read(c.OUT/'analysis.json');d=c.read(c.OUT/'diagnostics.json');counter=c.read(c.OUT/'counter_gate.json')
    header=f'''# {c.STAGE.upper()} — {'加载服务与同步依赖对照' if c.STAGE=='m4a' else '真实二维消费者与布局选择'}

主要结果与解释见 [FINDINGS.md](FINDINGS.md)，下文保留完整设计、表格与限制。

本阶段为开发域机制实验，旧选择器与旧报告保持不变，没有新选择器拟合或盲测记分。
固定 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；复用既有 compiler/tools image、ccache 与 kernel cache，native rebuild=0。
{len(cg['metadata'])} 个逻辑程序，新增 {cg['new_compiles']}、复用 {cg['reused']}，CPU 拒绝 {len(cg['rejected'])}。
三种输入、完整 2 MiB 输入所对应的全部输出与尾部 canary 共 {ag['checks']} 项通过。
三名独立 H100 worker，{a['event_samples']} 个正式 CUDA Graph 事件样本；NCU 原始报告 {len(counter['reports'])} 份，全部本地 CPU 重导入核对。

主指标是 128/256/384 MiB 连续 BF16 输入对应的 loop-free FULL 服务斜率，单位 ns/tile/SM；128 MiB 原始 µs 单列。
3 visits，每次 2 warmup+4 samples，配对随机顺序，不因输赢追加重复。每个 worker 内先以中位数作三点斜率，再比较同 worker 条件。
下文区间是三个配对 worker 的均值 ± t(2,0.975)×SEM，属于探索性逐项区间，未校正多重比较。
q 是占用 API 给出的 padding 驻留容量，不能当作实际同时活跃的 CTA 数。

'''
    codes={row['tag']:row for row in d['source_code_audits']}
    if c.STAGE=='m4a':
        summary=[]
        for row in d['within_NF_contrasts']:
            x=row['within_minus_NF_ns'];lo,hi=x['paired_worker_t95']
            summary.append([row['geometry'],row['swizzle'],row['q_capacity'],f"{x['mean']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
        deltas=[]
        for row in d['shared_layout_deltas']:
            x=row['service_delta_ns'];lo,hi=x['paired_worker_t95']
            deltas.append([row['tag'],row['q_capacity'],f"{x['mean']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]',row['MR_source_opcode_counts_equal'],row['registers_equal']])
        text=header+'''## 1. H4 的检验口径

检验“加载冲突/吞吐与跨 warp 同步依赖有不同驻留响应”，不是提前认定 H4 已成立。
寄存器布局、native 输出与 max 源程序固定，只改变 NVMMASharedLayout 的 128/64/32/0B swizzle，另配同 swizzle 的 T_control。
36 个程序都通过编译和 GPU 正确性，原 q=2/4/8 曲线直接引用 M2/M3；新配对计时只做 q=1/natural，NCU 补 q=1/2/4/natural。
静态 64 个 ML 冲突的消除预测，见 CPU 准备的 operation_plans_v2.json；真实指令分组、MR 和资源变化在本阶段核对。

## 2. within−N-first 的完整程序差分

负数表示 within 更快。只在双方实际容量相同的 q 档配对；全部 natural 时间与各自容量见 analysis.json。

'''+table(['geometry','shared swizzle B','q_cap','within−NF ns','paired t95 ns'],summary)+'''

## 3. 改变 swizzle 的配对效应与隔离程度

下表是同一寄存器布局、同一消费者相对 128B swizzle 的服务时间差。
MR 相同列只核对源行归属的 opcode 类及动态次数，不代表寄存器操作数、排程或所有依赖均相同。
寄存器不同时也保留数据，在受控 q=1 下消除容量差异，但不将其伪装成机器码完全相同的单变量延迟。

'''+table(['程序','q_cap','相对 sw128 ns','paired t95 ns','MR opcode/count 相同','regs 相同'],deltas)+'''

## 4. 硬件计数与结论边界

计数表见下节及 diagnostics.json：包含全部 shared load/store 波前、bank conflict、动态 LDS/STS/SHFL/BAR.SYNC、寄存器与 q，以及 barrier、short/long scoreboard、MIO 等可用 warp-state 指标。
源行/逐 PC 归属保存在 source_code_audits，不能把整个 kernel 的 shared traffic 都算成 ML。
shared swizzle 也改变 TensorMap 的 box 宽度和 TMA 指令分组；T_control 是匹配诊断，不是可以无条件线性扣除的已知底座。
因此本轮可以检验 H4 的方向、加载冲突是否有可测代价、其驻留响应和隔离程度；不能从这些共变数据直接求出唯一 MR 同步延迟系数。
若全 kernel barrier 百分比与叙事不一致，应保留该反例，而不能只看有利指标。
按 [NVIDIA NCU 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)，short scoreboard/MIO 涉及 shared、特殊函数或分支等多种操作；warp-state 百分比不能直接相加为阶段纳秒。

'''
    else:
        scores=d['scores'];summary=[]
        groups=defaultdict(list)
        for row in scores:groups[(tuple(row['geometry']),row['consumer'])].append(row)
        matrix=[]
        for (geometry,consumer),rows in sorted(groups.items()):
            best=min(rows,key=lambda row:row['service_ns']['mean'])
            x,y=best['gain_vs_default_percent'],best['gain_vs_fixed_vec1_percent']
            summary.append([geometry,consumer,best['tag'].split(':sw')[0],f"{best['service_ns']['mean']:.3f}",
                            f"{x['mean']:+.2f}%",f"{y['mean']:+.2f}%",str([round(z,2) for z in y['paired_worker_t95']])])
            for row in rows:
                code=codes[row['tag']]
                matrix.append([geometry,consumer,row['tag'].split(':sw')[0],row['q_capacity'],code['attributes']['num_regs'],
                               f"{row['service_ns']['mean']:.3f}",f"{row['raw_128MiB_us']['mean']:.3f}",
                               f"{row['gain_vs_default_percent']['mean']:+.2f}%",f"{row['gain_vs_fixed_vec1_percent']['mean']:+.2f}%",str(code['PTX_max'])])
        checked=d['forward_store_predictions']
        text=header+'''## 1. 本轮消费者

native：每列写一个 FP32 max；broadcast：保留原诊断 max(column)+logical_row；centered：真实 `wide=x.to(float32); wide-max(wide,rows)`，写出 FP32 二维 tile。
centered 是实际规约后广播减法子图，不冒充完整 Column-Softmax/LayerNorm。15 个合法唯一布局均测试这三种消费者；另有两个旧 1D T_control 与两个匹配输入/二维写出的 copy 控制。
二维消费者每 CTA 写 BM×BN 个 FP32，输出字节是 BF16 输入的两倍；输出缓冲按真实 stride 分配。
不同消费者输出量不同，各自在同语义内记分；未把三种消费者时间混为一类加速。

## 2. 自然驻留下的实测最快候选

正 gain 表示相对该消费者自身的默认 vec8:N-first / fixed_vec1:N-first 耗时下降。
“最快”是本轮事后观察，不是冻结选择器，也不把简单基线未输的情况隐藏。

'''+table(['geometry','consumer','实测最快布局','ns/tile/SM','gain vs default','gain vs fixed_vec1','vs vec1 paired t95 (%)'],summary)+'''

## 3. 全部候选、实际寄存器、驻留与 max lowering

PTX opcode 站点不是动态执行次数；动态计数另见下一节。宽化输入有两个消费者后，是否仍能打包由实际编译产物决定；未为保留 packed max 改写算子语义。

'''+table(['geometry','consumer','布局','q_cap','regs','service ns','128 MiB µs','gain vs default','gain vs vec1','PTX max sites'],matrix)+f'''

## 4. 静态输出计划的前向核对

在本轮 45 个 candidate/consumer 条件中，store sector 请求预测匹配 {sum(row['matched'] for row in checked)}/{len(checked)}。
全部命中/未命中记录保存在 diagnostics.json 的 forward_store_predictions，不事后覆盖预测。
返回读取需求是结构条件，实际 LDS 可能同时受 FP32 多消费者、MR lowering 和资源影响；不能只靠 whole LDS 差值分配独立广播延迟。
完整 q=1/2/4/8/natural 曲线、零收益和退化候选保存在 analysis.json，未只保留自然驻留下的赢家。

'''
    hw=[]
    for row in d['hardware']:
        meta=ag['metadata'][row['tag']]
        metrics=row['metrics'];code=codes[row['tag']]
        def rate(reason):return metrics.get('smsp__warp_issue_stalled_'+reason+'_per_warp_active.pct')
        hw.append([row['tag'],row['q'],code['q_capacity'],
                   f"{metrics['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']/row['grid']:.2f}",
                   f"{metrics['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']/row['grid']:.2f}",
                   code['whole_counts']['shuffle'],code['whole_counts']['barrier'],
                   rate('barrier'),rate('short_scoreboard'),rate('long_scoreboard'),rate('mio_throttle')])
    appendix='# 全部硬件计数摘要\n\nWF/BC 是全 grid `.sum/CTA数`；SHFL/BAR 是独立单 CTA software 的 warp 指令实例数。\nq condition 是请求档，q_cap 是该二进制 natural 最大容量；具体 padding 容量见 artifact_gate.json。\n\n'+table(
        ['程序','q condition','natural q_cap','load WF/CTA','load BC/CTA','SHFL/CTA','BAR/CTA','barrier %','short SB %','long SB %','MIO %'],hw)+'\n'
    c.derived(c.OUT/'COUNTERS.md',appendix.encode())
    text+='''## 5. 原始证据、复核与停点

[全部计数摘要](COUNTERS.md)；完整 FULL 三点斜率与各 worker 原始时间见 analysis.json，全部原始事件见 timing/invocation_*/raw.json。
compile_gate.json/compiled_payload.zip 固定实际 CUBIN；artifact_gate.json 绑定真实寄存器、local bytes、q 与正确性。
profiles/*/attempt_*/export 保存 ncu-rep、source/raw CSV、worker UUID 与实际加载 SHA；offline_import_validation.json 保存独立本地重导入核对。
诊断只使用本阶段冻结数据，不把历史 26.6 µs 或人工总分引入模型。

M4a 首次 CPU preflight 在编译前发现上传快照遗漏 c16/kernels.py；失败 dispatch/retrieval 记录保留，CPU-only packaging revision 增补依赖，36 个程序与指标清单未改，未产生 GPU 样本。
本阶段通过 validator 后生成 freeze.json、commit/push。只继续用户已授权的 M4a/M4b，全部完成后停下；没有新模型拟合、C++ 修改、PR 或跨架构实验。
'''
    if (c.OUT/'audit_revisions/1/invalidation.json').exists():
        text+='''
## 6. TensorMap 身份去重修正

初版去重只看 CUBIN，错误合并了 32×64/w8 的 sw0/sw128 T_control；两者 TensorMap 不同。
初版原始样本、全部报告、源码、派生产物、freeze 和初版 validator 记录完整保留于原路径与 audit_revisions/1/frozen_state.zip。
初版 validator 未检查 TensorMap 身份，属于本轮测量工具缺陷；不将其通过状态当作最终有效性证明。
revision3 将 TensorMap metadata、extent、dtype 和 geometry 纳入执行身份，正式计时由新三个 worker 完整重做，避免混配不同采集批次。
初版 7,560 个样本仅作历史证据，不进入本报告主统计；新主统计是 7,776 个样本、216 个条件/worker。
原有 8 份有效 NCU 报告复用，只补缺少控制的 2 份报告。最终 36 个不同执行上下文虽然只有 35 个不同 CUBIN，均有独立真实计数。
修正由身份核查触发，36 个案例、假说和指标集合没有改变，未按性能输赢选择重复。
'''
    c.derived(c.OUT/'REPORT.md',text.encode())
    plot(a,d)
    print(c.STAGE,'report/counter appendix/figure written')


def plot(a,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if c.STAGE=='m4a':
        fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        geometries=[tuple(row[:3]) for row in c.protocol()['pair_geometries_vectors']]
        for index,geometry in enumerate(geometries):
            m,n,w=geometry
            for col,condition in enumerate(('1','natural')):
                ax=axes[index,col]
                for placement,color in (('n_first','#315b83'),('within','#9c4f32')):
                    ys=[]
                    for sw in (128,64,32,0):
                        tag=f'M{m}_N{n}_w{w}:vec2:{placement}:sw{sw}:native:ONE_FULL'
                        rows=[row for row in a['oneshot'] if row['tag']==tag and row['q_condition']==condition]
                        ys.append(st.mean(row['tile_slope_ns'] for row in rows))
                    ax.plot([0,1,2,3],ys,'o-',label=placement,color=color)
                ax.set_xticks([0,1,2,3],['128','64','32','0'])
                ax.set_title(f'{m}x{n}/w{w} q={condition}')
                ax.set_xlabel('Shared swizzle (bytes)');ax.set_ylabel('ns/tile/SM');ax.grid(alpha=.2)
        axes[0,0].legend()
    else:
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        groups=defaultdict(list)
        for row in d['scores']:groups[(tuple(row['geometry']),row['consumer'])].append(row)
        for ax,((geometry,consumer),rows) in zip(axes.flat,sorted(groups.items())):
            vals=[row['gain_vs_default_percent']['mean'] for row in rows]
            labels=[row['tag'].split(':sw')[0].split(':',1)[1].replace(':','\n') for row in rows]
            ax.bar(np.arange(len(rows)),vals,color='#315b83')
            ax.set_xticks(np.arange(len(rows)),labels,fontsize=8)
            ax.axhline(0,color='black',lw=.7);ax.axhline(3,color='gray',ls=':',lw=.7)
            ax.set_title(f'{geometry} {consumer}');ax.set_ylabel('Time reduction vs default (%)');ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):
        stream=io.BytesIO();kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        fig.savefig(stream,format=ext,dpi=150,**kw)
        c.derived(c.OUT/'figures'/('responses.'+ext),stream.getvalue())
    plt.close(fig)


if __name__=='__main__':main()
