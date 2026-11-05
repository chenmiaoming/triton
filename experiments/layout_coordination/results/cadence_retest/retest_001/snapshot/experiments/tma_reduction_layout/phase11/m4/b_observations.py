"""M4b measured consumer effects and explicit limits on the causal story."""
import statistics as st
from . import common as c
from .report import table


def main():
    assert c.STAGE=='m4b'
    d=c.read(c.OUT/'diagnostics.json');a=c.read(c.OUT/'analysis.json')
    codes={row['tag']:row for row in d['source_code_audits']}
    hardware={row['tag']:row for row in d['hardware']}
    resources=[];control_rows=[];controls=[]
    for m,n,w in ((32,64,8),(32,128,4)):
        for vec in (1,8):
            for policy in ('native','broadcast','centered'):
                tag=f'M{m}_N{n}_w{w}:vec{vec}:n_first:sw128:{policy}:ONE_FULL'
                code=codes[tag]
                resources.append([f'{m}x{n}/w{w}',vec,policy,code['attributes']['num_regs'],code['q_capacity'],
                                  code['whole_counts']['shared_load'],code['whole_counts']['max'],str(code['PTX_max'])])
        for tail in ('control:ONE_TRANSFER','copy_control:ONE_COPY','vec8:n_first:sw128:centered:ONE_FULL','vec1:n_first:sw128:centered:ONE_FULL'):
            tag=f'M{m}_N{n}_w{w}:'+tail
            rows=[row for row in a['oneshot'] if row['tag']==tag and row['q_condition']=='natural']
            value={'tag':tag,'service_ns_mean':st.mean(row['tile_slope_ns'] for row in rows),
                   'raw_128MiB_us_mean':st.mean(row['full_us'] for row in rows)}
            controls.append(value)
            control_rows.append([f'{m}x{n}/w{w}',tail,f"{value['service_ns_mean']:.3f}",f"{value['raw_128MiB_us_mean']:.3f}"])
    anomalies=[]
    for m,n,w in ((32,64,8),(32,128,4)):
        for placement in ('n_first','within'):
            tag=f'M{m}_N{n}_w{w}:vec8:{placement}:sw128:centered:ONE_FULL'
            code=codes[tag];hw=hardware[tag];mt=hw['metrics']
            anomalies.append([f'{m}x{n}/w{w}',placement,code['q_capacity'],code['whole_counts']['barrier'],
                              f"{mt['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']/hw['grid']:.2f}",
                              f"{mt['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum']/hw['grid']:.0f}",
                              mt['smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct'],
                              mt['smsp__issue_active.avg.pct_of_peak_sustained_active'],
                              mt['dram__throughput.avg.pct_of_peak_sustained_elapsed']])
    text='''
## 3. 消费者导致真实 lowering 与资源变化

下表 LDS/MAX 为单 CTA 的 warp 指令实例，PTX 是静态站点，不能混为同一数量：

'''+table(['geometry','vec','consumer','regs','q_cap','LDS','dynamic MAX','PTX max sites'],resources)+'''

32×128/w4 vec8:N-first 的 native→broadcast→centered：regs 29→35→60，q_cap 16→12→8，返回 LDS 29→32→32。
centered 的宽化输入有第二个消费者，原 BF16 packed-max 站点转为 FP32，符合事前“多消费者可能改变 packing/liveness”的结构假说。
vec1 的 32→48 个寄存器及 q=16→10 也保留，不能把资源变化只归到默认 vec8。
这些是编译/执行真值，不是从寄存器数直接换算出来的性能损失。
预定义 conversion 指令族计数为0也不表示拓宽免费：32×128/w4 默认 centered 的 widening 源行实际有128次 IMAD.U32和32次 PRMT warp 指令实例；完整 opcode/PC 保留在 diagnostics.json。

## 4. 匹配二维 copy 控制给出的重要限定

copy 控制保留默认 vec8:N-first 输入/输出布局，只去掉 max/广播减法，仍写出同样的 FP32 二维 tile：

'''+table(['geometry','程序','service ns/tile/SM','128 MiB 原始 µs'],control_rows)+'''

32×128/w4 的默认 centered 为 1463.516 ns，默认布局 copy 已达 1453.654 ns（约差 0.68%）。
因此“q 从16降至8”确实发生，但不足以证明它造成约25%的布局差距；输出/传输组织本身已经形成高平台。
原来只有 1D 首行写出的 T_control 约 374.795 ns，不能直接作为这个二维写出程序的传输屋顶。
同样，128 MiB 原始时间与服务斜率存在截距差异，不能混用两种口径证明因果。

本轮只给 copy 控制测了默认布局，没有完整 copy×layout 矩阵，因此仍不能把候选的收益全部分配给 store、输入或 MR。
静态 store-sector 请求在 45/45 个 consumer/layout 条件命中；vec8 的二维请求为 vec1/2/4 的2倍，仍然是请求数，不是 DRAM 写字节数翻倍的结论。

## 5. 新反例：更少 BAR/WF 并不保证更快

centered 的 vec8:within 比默认 vec8:N-first 更慢：32×64/w8 慢约65.98%，32×128/w4 慢约21.18%。
其 BAR 和 shared-load WF 都更少，而输出 sector 请求仍相同：

'''+table(['geometry','placement','q_cap','BAR warp inst','load WF/CTA','store sectors/CTA','MIO %','issue %','DRAM %'],anomalies)+'''

例如 32×64/w8 同为 q_cap=8、store sectors=512，但 within 的 MIO 比例升至52.55%、issue降至15.74%，DRAM利用率降至40.85%。
这是现有简单指令/波前加和模型不能闭合的反例。MIO 是全 kernel 指标，仍不能据此唯一归因于某条 SHFL、shared 指令或全局写出。
后续可以预注册完整 copy×layout/q 与指令依赖/请求组织对照检验该假说；本轮没有看到异常后临时扩展 GPU 采集或改写旧模型。

## 6. 与简单基线的证据边界

两个真实 centered 几何的最快候选相对默认分别快22.68%和24.74%，但相对 fixed_vec1 仅约0.62%和1.38%，均低于3%实用差异门槛。
事前关心的 vec2/4:within 没有在本批证明同时以>3%优势击败默认与 fixed_vec1；32×128/w4 的最快是 vec4:N-first。
本轮扩充并检验了消费者敏感的操作计划，验证了实际优化空间，也明确保留了简单基线仍强和时间模型尚未完全闭合的边界。

![自然驻留各消费者内的平均耗时下降](figures/responses.png)
'''
    marker='\n## 3. 消费者导致真实 lowering 与资源变化'
    current=(c.OUT/'FINDINGS.md').read_text()
    if marker in current:current=current.split(marker)[0]
    c.derived(c.OUT/'FINDINGS.md',(current+text).encode())
    c.derived(c.OUT/'consumer_observations.json',{'status':'OK','controls':controls,
              'forward_sector_matches':sum(row['matched'] for row in d['forward_store_predictions']),
              'forward_sector_checks':len(d['forward_store_predictions']),
              'no_more_GPU_collected':True,'time_model_not_claimed_closed':True})
    print('M4b consumer observations appended; negative/near-tie baseline results retained')


if __name__=='__main__':main()
