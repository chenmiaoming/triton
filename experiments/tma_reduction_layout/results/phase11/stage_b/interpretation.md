# 阶段B结果解释

本轮使用固定upstream的未修改native，在同一个Gluon kernel中显式指定BlockedLayout。性能结论来自完整kernel计时，没有成本总分、拟合权重或旧模型的正确性评估。

## 主要结果

以下是full grid中vec1相对vec8的配对耗时减少率。点估计来自7次worker中位数的平均log ratio，区间是同条件六个布局比较的95% Bonferroni/t区间；它不等于表格中两个跨worker中位数直接相除。

| BF16/max条件 | 配对耗时减少 | 同条件同时区间 |
|---|---:|---|
| 32×128 / 4 warps，host descriptor | 2.332% | [0.801%, 3.841%] |
| 32×128 / 4 warps，device descriptor | 1.266% | [-0.047%, 2.561%] |
| 32×64 / 8 warps，host descriptor | 46.652% | [41.056%, 51.718%] |
| 32×64 / 8 warps，device descriptor | 41.297% | [39.229%, 43.294%] |

原始32×128/4-warps几何在H100上差异较小。按预声明3%实用差异，四种布局都仍可能最优；不能据此指定vec1或vec2作为唯一选择，也不能声称所有差异都已证明小于3%。

32×64/8-warps的BF16结果支持一个明确的优化机会：vec1和vec2都明显优于vec8，vec4也被排除出可能最优集合。但vec1与vec2尚未分出超过3%的稳定差异，因此本轮不能指定其中一个为唯一最优。

固定版本中BF16默认vec8，FP32默认vec4。因此FP32中手工vec8较慢不能称为当前默认FP32布局的性能回归。FP32/8-warps仍可能最优的集合包含vec1、vec2、vec4。

## Gluon是否复现了普通Triton入口

8组默认布局对照的完整输入blocked/shared布局与过滤PTX指令序列一致。对应binary的动态shared-load warp指令数在各自三次NCU采集中一致；平均wavefront差异最大约0.054%。独立计时对照的95%描述区间均落在约±0.7%内。

这些证据支持本轮默认入口上的Gluon复现。过滤签名覆盖完整kernel，包含初始加载与输出转换；它不是纯规约计数，也不是所有机器指令/地址完全一致的证明。改变布局后的资源和lowering共同变化全部保留。将来真正修改编译器时仍须重新验证实际生成的binary。

阶段B只为默认入口的复现对应关系采集NCU，并未对vec1/2/4/8全部进行机制profiling。现在不能把收益归因于更少bank conflict、更少wavefront或更少跨warp通信。

## 下一步应调整的范围

建议阶段C把vec1补入原先vec2/vec8的控制范围，优先研究32×64/8-warps的BF16机会：先采集四种布局的动态加载、wavefront、规约/输出访问和资源，再保持vec不变改变同列线程的warp归属。vec1与vec2都是当前候选，不能只分析一个预期赢家。

原因明确后再考虑对应的编译器变换与跨架构性能复现。阶段A的B200/SM120工具覆盖不是新布局的性能验证；本轮H100结果也不能替代原GB300/SM103回归系统。

## 执行和证据

40个artifact通过三类完整输出正确性，共120项检查。7次独立worker调用保留22,400个单kernel CUDA-event样本；实际取得7个不同Driver GPU UUID，但不由此宣称覆盖全部H100系统。所有正式计时进程均未导入Triton、调用编译器或启动子进程。

最终40次kernel编译查询全部命中持久缓存，native与ccache统计未变。96份独立NCU报告及其原始导出与source计数保留。早期采集器/错误oracle的三次调用及协议修订全部保留；修正oracle的依据是固定源码中的PropagateNan.NONE与原始IR中的arith.maxnumf。没有修改kernel来迎合结果，也没有正式计时的结果驱动重跑。

工作停在阶段B，未执行C/D、修改PPT或创建PR。
