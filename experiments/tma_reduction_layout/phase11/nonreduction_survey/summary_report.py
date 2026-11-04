"""Render the final evidence summary only after the complete validator passes."""
from . import common as c


def main():
    validation=c.read(c.OUT/'FINAL_VALIDATION.json');assert validation['status']=='OK'
    a=c.read(c.OUT/'stage_n4/analysis/analysis.json')
    large=c.read(c.OUT/'stage_n5_timing/analysis/analysis.json')
    root=c.OUT
    lines=['# 无规约公开 Triton 内核：布局对照实验汇总','',
        '本轮在 H100 上完成六个算子族、七个计算子类的公开源码对照。已有独立复测证据说明：默认布局的优化空间可以出现在没有 TMA、没有轴规约的程序中。但结果依赖算子、形状和工作集；没有证据支持全局改成 vec1，或将所有候选都称为布局缺陷。','',
        '更准确的研究对象是输入、重排/消费、输出与资源之间的布局协调。最初的 TMA 版本回归仍是一条独立问题；这里没有执行版本间对照，不能将这些新结果称为性能回归。','',
        '## 1. 基线与方法','',
        '- 固定 compiler `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`，未改 C++ 或 compiler pass，复用 native image、ccache 和 kernel cache；native rebuild 为零。',
        '- U 来自固定公开源码；G0 保留实际 Tensor JIT 的参数、specialization、stride/alias、计算次序及所有中间布局。14 个原始 case 的 U/CPU G0/实际 G0 的 allocated ELF、ABI 与资源相同；独立计时全部在 ±3% 等价带内；7 个子类的 U/G0 软件动态 opcode 直方图也相同。',
        '- 初轮成块采样的 7 个不确定控制没有删除。统一改成逐 round 随机交错后，全部 14 个控制通过等价门；新旧样本分开。',
        '- N2 测全部 110 个 GL，连同 U/G0 共 138 个逻辑程序、128 个物理条件、119,040 个样本。保留全部收益与退化，未只测赢家。',
        '- N4 在获得新样本前为全部 14 个配置各固定一个 N2 最快 GL，包括零收益配置；使用新 worker 与新 U/G0，共 38,130 个样本。此为同配置独立复测，不是未见几何或整模型泛化。',
        '- 主估计量：3 workers × 10 visits × 31 samples，visit 中位数的算术平均；4096 次配对 worker/visit bootstrap。下文区间为逐比较 95% CI，无多重比较校正。严格通过要求相对 U、G0 的下界均高于 +3%。','',
        '## 2. 独立复测结果','',
        '| 算子族 | 独立结果（相对 U） | 证据判断 |',
        '|---|---|---|',
        '| torchao scale swizzle | 11008×128：+5.75%，CI [5.02,6.35]；4096×128：+2.47%，CI [1.94,3.09] | 大配置通过双控制 >3%；小配置未通过 |',
        '| vLLM AWQ dequant | 3584²：+6.42%，CI [6.09,6.79]；1024²：+4.22%，CI [3.44,4.91] | 两个配置均通过双控制 >3% |',
        '| FlashAttention RoPE / SGLang MRoPE | 四个配置均未确认 >3% | 保留零结果；默认已避免部分显式布局交换 |',
        '| SGLang FP8 QKV pack | 两个配置均未确认 >3% | 仅 H100 上独立 pack 的可移植性证据，原路径属于 SM120 |',
        '| FLA KDA gate | D64 +2.71%；D128 +4.27%，但后者相对 G0 的 CI 下界为 2.988585% | D128 保留边缘结论，不四舍五入为严格通过 |',
        '| SGLang HND KV scatter | 两个配置均未确认 >3% | 保留负/零结果 |','',
        '初始 14 个配置中，3 个配置、2 个算子族在独立数据上严格通过。不是 14 个都存在已确认缺陷。全部区间、候选、worker 及适用边界见 [N4 报告](stage_n4/REPORT.md)；探索性结果见 [N2 报告](stage_n2/REPORT.md)。','',
        '![独立复测的两个控制](figures/N4_confirmation.png)','',
        '图：蓝色为两个下界均超过 3%；橙色为 D128 gate 的边缘结果；灰色为其他结果。虚线为 3% 门槛。','',
        '## 3. NCU 支持的具体机制','',
        'N3 在读取候选时间前冻结 60 份报告；N3b 的 16 份 gate 报告明确标为后验机制诊断；N5 再补 6 份大工作集硬件报告。正式计时没有附加 profiler。软件 InstructionStats 与完整 grid 的硬件请求、wavefront、缓存及资源分别保存。','',
        '- **Swizzle：输出请求分组与布局协调。** Store sectors 从 64 降到 16/CTA；只改输出会新增 STS/BAR/LDS，协调输入与输出可消除这次新增交换。Global load sectors 仍为 128/CTA。小工作集的 DRAM 读取接近零，不能称为 HBM 带宽收益。',
        '- **AWQ：布局交换明显减少。** 默认的 LDS/LDSM、STS 分别为 36/36，候选为 4/4；BAR 从 68 降到 4 个动态 warp 指令/CTA。寄存器 72→40，真实驻留容量 q 从 7→12。与此同时，global load 指令 16→40，load sectors 保持 160/CTA。因此窄 vec 的收益不能解释为更少的加载指令或 HBM 字节。',
        '- **Gate：消除输入到消费布局的共享交换。** D128 的 shared load wavefront 64→0、BAR 8→0、寄存器 48→36；D64 同样消除交换，但没有通过实用收益门。D128 的 q 容量 5→6 不等于实际占用率提高：512 个 CTA 分布到 132 SM，平均仅 3.879 CTA/SM。AWQ 小配置只有 128 个 CTA，也不能用 q 容量增长解释其收益。',
        '- **反例同样保留。** RoPE/MRoPE 的某些干预增加共享转换；FP8 pack 的一个干预减少请求 sectors，却未减少实际 DRAM 字节，也未获得 >3% 收益；KV scatter 的默认路径已经没有共享交换可删。','',
        '这些事实支持“跨阶段布局存在条件性权衡”，没有把指令计数加权成时间总分，也没有独立辨识每种机制的因果纳秒成本。完整细节见 [N3 机制复核](stage_n3/MECHANISM_REVIEW.md) 和 [N3b 机制复核](stage_n3b/MECHANISM_REVIEW.md)。','',
        '## 4. 大工作集与尾块补测','',
        '两个大配置仅迁移已经选定的同一候选，不重新选优；额外的 131×7 swizzle 使用 stride (19,2)，检查全部八个原声明布局及 padding。三个新配置的实际 U/G0 ELF/ABI/资源相同，48 项完整 oracle/guard/只读检查通过。','',
        '| 新配置 | 相对 U 收益 (%) | CI95 (%) | 结果 |','|---|---:|---|---|']
    for row in large['rows']:
        if row['variant'] in ['U','G0']:continue
        low,high=row['gain_vs_U_CI95']
        lines.append(f'| `{row["case"]}` | {row["gain_vs_U_percent"]:+.3f} | [{low:+.3f},{high:+.3f}] | '+('仍超过 +3%' if low>3 else '候选更慢，但未达到 >3% 实用退化门')+' |')
    lines += ['',
        '同一 AWQ 布局在小工作集获益，在大工作集约慢 2%，这一负结果禁止将 vec1 当成全局规则。数组规模超过 L2 不自动证明纯 HBM 瓶颈；N5 单独采集实际缓存/DRAM 请求，解释以 [N5 报告](stage_n5/REPORT.md) 为准。','',
        '## 5. 官方定义与边界','',
        '按 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 分别解释 request、sector 和 wavefront；sector 是 32B 请求单位，不是 DRAM 必然传输量。Warm application replay/cache-none 与 cold kernel replay/cache-all 分列，profiler 时间不进入主计时。',
        '[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 将合并访问与参与线程的地址分组联系起来；[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#increased-l2-capacity) 给出 H100 的 L2 容量。本文中的实际效果仍来自保存的实验，不能由指南直接推导性能百分比。','',
        '公开源码路径、可选后端、FP8 的架构边界、N05 的 U-only 配置选择及原地 MRoPE 的输入恢复方式见 [N1 报告](stage_n1/REPORT.md)。不声称框架端到端收益、跨 GPU 有效性、所有 workload 普遍次优或版本回归。','',
        '## 6. 交付与复核','',
        '所有原始样本、compiler artifacts、NCU .ncu-rep、导出、调用绑定、失败尝试和修订快照均保留。旧阶段未覆盖或替换。N1 采样修订、N2 AutoLayout 索引修复、N5 尾块枚举失败，以及 validator 的路径/CSV 重放修复均保留前后记录。',
        '最终完整校验：[FINAL_VALIDATION.json](FINAL_VALIDATION.json)，状态 OK。它重新检查来源、编译与加载 SHA、正确性、alias、完整计时覆盖、冻结选择与所有 NCU replay，并重算派生分析。CPU 校验不要求实验必须获得正收益。',
        '本轮完成后停止。未创建 PR，未修改 C++ 或启动编译器集成/下一架构阶段。','']
    assert all(x['classification']=='practically_equivalent' for x in a['baseline_equivalence'])
    c.write_once(root/'SUMMARY.md','\n'.join(lines).encode())
    print(root/'SUMMARY.md')


if __name__=='__main__':main()
