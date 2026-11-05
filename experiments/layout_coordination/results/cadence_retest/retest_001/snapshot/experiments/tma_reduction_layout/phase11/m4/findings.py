"""Data-bound short findings; hypotheses remain distinct from measurements."""
from collections import defaultdict
import statistics as st
from . import common as c
from .report import table


def main():
    d=c.read(c.OUT/'diagnostics.json');a=c.read(c.OUT/'analysis.json')
    if c.STAGE=='m4a':
        mainrows=[];changes=[];controls=[]
        for geom in ([32,128,4],[16,256,8]):
            for row in d['within_NF_contrasts']:
                if row['geometry']==geom and row['swizzle']==128:
                    x=row['within_minus_NF_ns']
                    mainrows.append([geom,row['q_capacity'],f"{x['mean']:+.2f}",str([round(v,2) for v in x['paired_worker_t95']])])
            m,n,w=geom
            for q in ('1','natural'):
                groups=[]
                for sw in (128,64):
                    tag=f'M{m}_N{n}_w{w}:control:sw{sw}:ONE_TRANSFER'
                    groups.append(st.mean(row['tile_slope_ns'] for row in a['oneshot'] if row['tag']==tag and row['q_condition']==q))
                controls.append([geom,q,f'{groups[1]-groups[0]:+.2f}'])
        for row in d['shared_layout_deltas']:
            if ':within:sw64:' in row['tag'] and ('M32_N128' in row['tag'] or 'M16_N256' in row['tag']):
                x=row['service_delta_ns']
                changes.append([row['tag'],row['q_capacity'],f"{x['mean']:+.2f}",str([round(v,2) for v in x['paired_worker_t95']]),row['registers_equal'],row['MR_source_opcode_counts_equal']])
        text='''# M4a 主要结果（TensorMap 身份修正后的正式批次）

H4 的方向得到支持，但“ML 只是吞吐、MR 只是串行延迟”的二分解释需要收紧。
以下仅使用 revision3 的 7,776 个样本；初版 7,560 个样本、初版 validator 与全部冻结状态保留，未混入主统计。

## 1. q=1 的原反例仍成立

128B swizzle 下 within−N-first 的配对差分，负数表示 within 更快：

'''+table(['geometry','q_cap','within−NF ns/tile/SM','paired worker t95 ns'],mainrows)+'''

## 2. 合法 shared-layout 干预

同一 vec2:within、同一 native 消费者，128B→64B swizzle。两个几何在 q=1 下均精确命中 load WF 128→64、load BC 64→0 的事前预测。

'''+table(['程序','q_cap','FULL 差分 ns','paired t95 ns','regs 相同','MR opcode/count 相同'],changes)+'''

32×128/w4 的 regs 22→22、natural 容量 16→16；16×256/w8 的 regs 14→15、容量 8→8。
两项 MR 的动态 SHFL 分别保持 8 和 16。输入加载实际为 LDSM.16.M88.4，不能按 vec2 字面值猜 LDS 指令宽度。
融合 max 可能归属 raw.load 源行，主要 MR 则归属固定编译器 standard.py::max 的 core.reduce；所有 PC/opcode/来源均保留。

## 3. 仍然存在的混杂与 H4 边界

swizzle 同时改变 TensorMap box、地址计算和 TMA 搬运布局；匹配 T_control 的 64B−128B 差分为：

'''+table(['geometry','q condition','T_control 差分 ns'],controls)+'''

因此不能把 FULL 时间变化全部分配给 bank conflict，也不能把 FULL−control 直接当作已辨识的可加性阶段成本。
32×128/w4 q=1 的 NF/within whole BAR.SYNC warp 指令实例为 24/8；barrier 比例 3.55%/1.37%。
16×256/w8 对应 barrier 比例为 5.89%/1.84%。这些计数与跨 warp 同步路径方向一致，仍属全 kernel 相关诊断。
消除冲突后，within 的 short-scoreboard 比例从 8.90%→5.80%、9.82%→5.98%，而 q=1 MIO throttle 均为 0。
加载冲突也体现为依赖等待，不能简单称为完全无依赖的吞吐开销；这些百分比不等于独立阶段纳秒。
自然驻留时差异被压到较小量：32×128 出现小幅反向变化（区间为正），16×256 的区间跨零。
这支持暴露成本随并发/传输/排程变化，不等于一律完全掩盖为零；没有建立唯一 ML/MR 延迟分解，也没有校准新选择器。

36 个合法执行上下文的正确性 108/108，通过 3 个新 worker 的完整正式配对批次；10 份 NCU 报告全部本地复读。
同一 CUBIN 配不同 TensorMap 也必须独立采样，此身份约束已加入最终 validator。
'''
    else:
        groups=defaultdict(list)
        for row in d['scores']:groups[(tuple(row['geometry']),row['consumer'])].append(row)
        summary=[];comparisons=[]
        for (geom,consumer),rows in sorted(groups.items()):
            best=min(rows,key=lambda row:row['service_ns']['mean'])
            g=best['gain_vs_fixed_vec1_percent']
            summary.append([geom,consumer,best['tag'].split(':sw')[0],f"{best['gain_vs_default_percent']['mean']:+.2f}%",f"{g['mean']:+.2f}%",str([round(x,2) for x in g['paired_worker_t95']])])
            if consumer=='centered':
                for row in rows:
                    if ':within:' in row['tag'] and (':vec2:' in row['tag'] or ':vec4:' in row['tag']):
                        comparisons.append([geom,row['tag'].split(':sw')[0],f"{row['gain_vs_default_percent']['mean']:+.2f}%",f"{row['gain_vs_fixed_vec1_percent']['mean']:+.2f}%",str([round(x,2) for x in row['gain_vs_fixed_vec1_percent']['paired_worker_t95']])])
        text='''# M4b 主要结果

本轮真实消费者是 x.to(float32)-column_max(x.to(float32))，是二维规约后广播减法子图，未冒充完整 softmax/LayerNorm。
native/broadcast/centered 的全部 15 个布局均测量；在每个消费者自身语义内对比，输出字节数不同的消费者之间不直接计算布局收益。

## 1. 自然驻留结果

实测最快是探索性事后统计，不是冻结选择器：

'''+table(['geometry','consumer','最快布局','gain vs default','gain vs fixed_vec1','vs vec1 paired t95 (%)'],summary)+'''

## 2. 事前关心的 vec2/4:within 能否同时击败默认和 vec1

保留全部正、零、负结果；均为 centered 同语义内比较：

'''+table(['geometry','布局','gain vs default','gain vs fixed_vec1','vs vec1 paired t95 (%)'],comparisons)+'''

是否优于简单基线应由上述配对区间与 3% 实用门槛判断，而不能仅凭最快点估计宣称胜出。
宽化输入有第二个消费者后，max lowering、返回读取、store 分组、寄存器与驻留都可能变化；详见 REPORT.md 全矩阵和 diagnostics.json 的实际 PTX/SASS/PC/q。
本轮没有把共享内存波前、shuffle 或 spill 的静态站点数当作总时间成本，也没有为获得期望赢家改变代码或样本口径。
'''
    c.derived(c.OUT/'FINDINGS.md',text.encode())
    print(c.STAGE,'data-bound findings written')


if __name__=='__main__':main()
