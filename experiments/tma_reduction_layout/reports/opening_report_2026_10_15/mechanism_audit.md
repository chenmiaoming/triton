# Triton 当前如何选择 TMA 与普通 load 的布局

源码核对日期：2026-10-15。实验固定 upstream 为 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；A10 使用本地 Triton 3.7.1，另读取 upstream v3.7.1 源码作为参考。
本次实时读取的官方 main 为 `4eda6cc91e86553a2517d1c764ea2967985553ad`。
这三个来源分别归档在 `source_audit/`，含 SHA256 与获取时间。没有构建或计时新的 main。

**对：默认布局有时已经接近已测候选中的较快者。但这里的默认选择来自访存启发式与局部布局优化，不是枚举 vec1/2/4/8 并预测完整 kernel 时间后选出的赢家。**

所核对 main 的 Coalesce.cpp、CoalesceUtils.cpp、Utility.cpp、OptimizeThreadLocality.cpp、RemoveLayoutConversions.cpp 与实验固定 upstream 文件逐字相同。其他文件存在演进，因此不把这句话推广到整个编译器或所有代码生成结果。

## 1. 同一条规则为什么在两个配置上结果不同

A10 同批完整 kernel 时间，BF16 输入、FP32 输出、逐列 partial max，单位 µs：

| 配置 | 普通 Triton 默认 | Gluon vec1 | Gluon vec2 | Gluon vec4 | Gluon vec8 |
|---|---:|---:|---:|---:|---:|
| 32×128 / 4 warps | 298.368 | 313.264 | 297.984 | 296.960 | 298.880 |
| 32×64 / 8 warps | 422.400 | 279.440 | 285.696 | 368.640 | 421.888 |

第一行中默认实现接近已测候选的较快者；第二行有明显优化空间。这里是三个进程的描述性对照，不据此宣称唯一最优或全空间最优。普通 Triton 与 Gluon 属于不同 pipeline，受控布局收益主要看同一个 Gluon kernel 内的对照。

源码规则和最终 TTGIR 对应如下，省略 batch 维，数组按 `[M,N]`：

| 配置 | 平均元素/线程 | SPT | TPW | WPC | 每列贡献线程 | 每线程该列值数 |
|---|---:|---|---|---|---:|---:|
| 32×128 / 4 warps | 32 | [1,8] | [2,16] | [4,1] | 8 | 4 |
| 32×64 / 8 warps | 8 | [1,8] | [4,8] | [8,1] | 32 | 1 |

两者都是 vec8，但缩短 N、增加 warp 后，一列被分散到更多线程。取数、规约、输出转换和资源的综合效果需要测量。这里尚未把两个配置的性能差分别归因于这些分项。

## 2. TMA / descriptor 路径

入口是 `Coalesce.cpp::pickDescriptorLoadStoreLayout`。对于本例的一 CTA、BF16、row-major descriptor load，核心规则为：

```text
E = max(floor(本 CTA 的元素数 / (num_warps × 32)), 1)
V = min(E, 128 / element_bitwidth)
sizePerThread = [1, ..., V]
order = row-major，N 优先
BlockedEncodingAttr::get(shape, sizePerThread, order, num_warps, 32, ...)
```

128 的单位是 **bit**，表示最多 16 bytes 的向量化目标；BF16 对应 V 上限 8，FP32 上限 4。
这是寄存器分布的选择，不是 TMA global→shared 传输宽度。

对 32×128/w4：E=4096/128=32，V=min(32,8)=8。
对 32×64/w8：E=2048/256=8，V=min(8,8)=8。
函数不读取后续 reduce axis，也不计算 max、shuffle、barrier 或输出转换的时间后比较多个 V。

随后 `OptimizeDescriptorEncoding` 分配/协调 descriptor 的 shared-memory 编码；这是与 distributed register encoding 不同的一层。
NVIDIA 的 `TMALowering` 再生成 shared allocation、TMA copy/wait 和 local load 等路径。
本例最终输入 blocked encoding 由实际 TTGIR 核对，不能只凭早期函数就推断所有后续程序的最终布局。

源码：

- [Coalesce.cpp L32–L71](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp#L32-L71)
- [实验 upstream 的 OptimizeDescriptorEncoding.cpp](https://github.com/triton-lang/triton/blob/20d9d98fead78c2f65e1b3ba5a7881d86a836dad/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp)

## 3. 普通 tl.load 路径

入口是 `Coalesce.cpp` 的 pointer-tensor 分支，调用 `buildCoalescedEncoding`：

1. `ModuleAxisInfoAnalysis` 分析指针张量的连续性与可证明的字节整除/对齐信息。
2. `order` 优先安排连续程度较高的维度；本例为连续的 N 维。
3. `getNumElementsPerThread` 根据该维的连续长度、对齐及普通 load 的 128-bit 目标，给出初始 perThread。
4. 在相关且相同 shape/order 的内存操作之间协调 perThread，再以本 CTA 的平均每线程元素量限制它。
5. 将 perThread 放在 `sizePerThread[order[0]]`，调用同一个 blocked-layout 构造器分配 lane/warp。

对普通 load，辅助函数中的主要计算是：

```text
maxMultiple = max(字节整除度 / 元素字节数, 1)
maxContig = min(可证明连续长度, 对应维的 shapePerCTA)
alignment = min(maxMultiple, maxContig)
currPerThread = min(alignment, 128 / element_bitwidth)
```

这是本例与普通 load 的解释，不推广到具有特殊限制的 atomic。实际访存宽度仍可能被 mask、布局转换和后端 lowering 限制，SPT 不等于硬件事务数。

本例 BF16 的 N 连续，runtime 参数已向编译器提供足够对齐，因此允许 vec8；两个配置的平均每线程元素量也都至少为 8。
输出只有 N 维，与输入三维张量 shape 不同，不能把相关内存操作协调理解为对这条 load→reduce→store 做完整性能建模。

源码：

- [Coalesce.cpp L108–L148](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp#L108-L148)
- [CoalesceUtils.cpp](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp#L17-L97)
- [Utility.cpp L177–L197](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/Utility.cpp#L177-L197)

## 4. threadsPerWarp 和 warpsPerCTA 从哪里来

`BlockedEncodingAttr` 的 shape-based builder 从 `order` 指定的连续维开始，把 lane 和 warp 尽量放到该维能覆盖的范围，再向其他维分配。
它使用 shape、SPT、warp 数和 CTA 布局；不根据 reduction 时间挑 lane/warp 分配。

以两个一 CTA 配置的 N 维为例：

- 32×128/w4：N 上有 128/8=16 个线程位置，因此 N 占 16 lanes；剩下 2 lanes 放 M。N 已覆盖完，4 warps 全沿 M，得到 TPW=[2,16]、WPC=[4,1]。
- 32×64/w8：N 上只有 64/8=8 个线程位置，剩下 4 lanes 放 M；8 warps 全沿 M，得到 TPW=[4,8]、WPC=[8,1]。

第一个布局基本覆盖 8×128，再用同一组线程沿 M 覆盖四次；第二个布局基本覆盖完整 32×64。

源码：[shape-based builder](https://github.com/triton-lang/triton/blob/20d9d98fead78c2f65e1b3ba5a7881d86a836dad/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L827-L863)。
`audit.py` 对该分配过程做 CPU 重建，并与已保存的六份 A10 release/shape 最终 TTGIR 以及两份 H100 host-TMA 最终 TTGIR 比较；这是源码推导加产物核对，不是新增 instrumented compiler trace。

## 5. 后续为什么没有变成另一种规约布局

NVIDIA pipeline 中先运行 Coalesce，再多次运行 RemoveLayoutConversions，包含 OptimizeThreadLocality，之后还有 TMA/shared 编码相关处理与最终 lowering。

`RemoveLayoutConversions`：

- descriptor 操作是 layout anchor；本例普通 load 也因元素数足够多被视为 expensive memory op，因此作为 anchor。
- 传播 anchor 的布局，穿过 elementwise/cast、reduce、reshape 等操作；reduce 结果编码由输入编码推导。
- 存在其他启发式，如冲突解决和 rematerialization 成本估计。因此不能说整个编译器“没有成本机制”。
- 这些规则不等于为本例枚举 vec1/2/4/8 并校准完整时间的选择器。

`OptimizeThreadLocality`：

- 包含特定 reshape/gather 优化，以及一种循环累加规约变换。
- 所核对的源码里，循环规约变换要求 `axis == rank-1`，并匹配直接 load、单一使用链和特定 `scf.for` 累加结构等条件。
- 本例输入 `[1,BM,BN]`、axis=1，不是最后一维，且无该循环结构，因此不匹配。
- 所核对版本中没有名为 `shouldOptimizeThreadLocality` 的函数；相关判定位于 pass 的 `runOnOperation`。

因此，本例最后保留 vec8 有可查的规则链。不能仅因为第一行性能较好，就推断编译器已经评估并理解了加载/规约间的完整成本权衡。

源码：

- [NVIDIA pass pipeline](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/third_party/nvidia/backend/compiler.py#L303-L369)
- [layout anchors 与传播](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L221-L405)
- [thread-locality 的具体限制](https://github.com/triton-lang/triton/blob/4eda6cc91e86553a2517d1c764ea2967985553ad/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp#L254-L314)

## 6. 能汇报的研究判断

已有机制是**以访存连续性/向量化为主的布局构造，加特定模式的局部优化**。它在一些配置上已接近所测较快候选，在另一些部分规约配置上留下明显优化空间。
研究任务应限定为：在适用模式下，用编译期可得信息和经过校准的参数，预测完整取数—规约—输出路径的成本，从而识别值得改选的配置，并在差异小或不确定时回退。
这仍需微基准可辨识性、模型冻结、未见配置预测与实际 binary 验证。本次只更新文档和进行源码/既有证据审计。
