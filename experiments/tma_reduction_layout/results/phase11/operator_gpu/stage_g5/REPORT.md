# Stage G5 — U / G0 全编译链基线审计

## 1. 结论与范围

此前 G0 只对齐了主要输入/输出的布局参数，没有对齐 U 的完整中间运算图与优化流水线。因此“同默认 vec”不等于“同一基线”。原始 Triton 与旧 Gluon 的差异确实会混入 GL 相对 U 的加速，不能全部归因于候选 layout。

本阶段在固定编译器 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`、原生库 SHA `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`、`cuda:90` 下完成 CPU 编译干预：

- 9/9 组冻结的原始 U/G0 CUBIN 均逐字节复现（共 18 份），旧 G0 的机器指令与 U 为 **0/9 相同**。
- 修正后的 Gluon 基线为 **9/9 可执行代码、所有 ELF `SHF_ALLOC` 区段、运行时 ABI、资源分配相同**。它们不是复制 U 的 IR/PTX/CUBIN，而是从合法 Gluon 源程序重新编译得到。
- 另补验 G4b 的四种额外真实调用编译形式（batch=1 参数特化、SGLang 实际 stride/辅助参数 ABI）：4/4 新编译对同样通过；映射回 G4b 为 **10/10 工作负载**有与其原始 U 相同的 Gluon 基线。
- 全程 **0 GPU 分配、0 新计时、0 新 NCU、0 原生重编译**。共 59 次 CPU 程序编译，含消融、未通过修复、缓存恢复及真实 ABI 补验。没有修改旧样本、旧编译产物或 C++。

九组门禁见 [equivalence_gate.json](equivalence_gate.json)；真实调用补验见 [actual_ABI_analysis.json](actual_ABI_analysis.json)。本阶段不宣称候选 GL 在新基线下的加速，候选尚未重新编译、验证与计时。

## 2. 为什么同输入布局仍生成不同程序

固定版本的后端为 Triton 与 Gluon安排了不同的前半段流水线：

| 流水线 | 涉及本问题的步骤 |
| --- | --- |
| Triton `make_ttgir` | Coalesce；多轮 RemoveLayoutConversions；canonicalize/CSE；Triton LICM；后续清理与指令重排 |
| Gluon `gluon_to_ttgir` | 显式布局解析、Gluon 自有 canonicalizer、SCCP/CSE 等；常规路径不运行上述 Triton RMLC/LICM 组合 |

源码：[backend/compiler.py](compiler_source/third_party/nvidia/backend/compiler.py)，`make_ttgir` 从 302 行、`gluon_to_ttgir` 从 389 行开始。Gluon 的 [Canonicalize.cpp](compiler_source/lib/Dialect/Gluon/Transforms/Canonicalize.cpp) 第 58 行明确排除了改变 layout 的 canonicalization，尤其是 ConvertLayoutOp。**这是显式布局接口的设计边界，不能据此直接认定 Gluon 有编译器 bug。**

旧手写翻译虽然给主输入指定了相同的 blocked layout，但辅助索引、指针、mask、statistics、转换位宽与循环位置没有完整复刻。原先的 effective-layout gate 不足以保证 U/G0 程序等价。

## 3. vLLM segment merge：统计量重物化与输出转换

以 `C06_s32_d256` 为例，主矩阵 layout 均为：

```
L = blocked<[1,4], [1,32], [2,2], order=[1,0]>
row = SliceLayout(dim=1, L)
stat = blocked<[1], [32], [4]>
out = blocked<[2], [32], [4]>
```

U 的第一次 RemoveLayoutConversions 把图改成了两个 statistics 分支：一个在 `stat` 中做标量 max/sum，另一个直接在 `row` 中加载/重算供主矩阵广播使用。旧 G0 则只在 `stat` 中加载一次，算出 scale 后再转换到 `row`。

```
U:      load m,z in stat -> scalar statistics
        load m,z in row  -> row scale -> matrix consumer

old G0: load m,z in stat -> scalar statistics + scale
                                       -> convert scale to row -> matrix consumer
```

在 U 的原生 pass trace 中，Coalesce 后共有 27 个 `ttg.convert_layout`；第一次 RMLC 后成为 **1 个**，静态 `tt.load` 数从 **6 增至 8**：增加的是两份小向量 statistics 加载，包含标量加载在内的总数，不是硬件动态访存次数。证据：[023.before](pass_checkpoints/C06_s32_d256/U/023.before.mlir)、[023.after](pass_checkpoints/C06_s32_d256/U/023.after.mlir)、[diff](pass_checkpoints/C06_s32_d256/U/023.diff)。

这来自 [RemoveLayoutConversions.cpp](compiler_source/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) 的 backward rematerialization：小 load、索引和逐元素计算可以被复制以避免 layout conversion；[Utility.cpp](compiler_source/lib/Dialect/TritonGPU/Transforms/Utility.cpp) 的 `isExpensiveLoadOrStore` 不把不足 CTA 线程数的小向量 load 当成昂贵锚点。这里是实际编译器的重物化启发式，不是本项目之前的性能预测模型。

输出也有独立差异：

```
U:      FP32 result -> BF16 -> convert_layout(BF16, out) -> store
old G0: FP32 result -> convert_layout(FP32, out) -> BF16 -> store
```

旧 G0 还转换输出指针及 mask；U 在目标 layout 里直接重建它们。因此旧 G0 最终有 5 个 conversion，U 只有一个 BF16 conversion。指针、mask 可以重算，数据值则需要通信，不能把这些转换只按同一个“次数”处理。

诊断中，仅给旧 G0 补 RMLC 得到 `5 -> 2` 个 conversion，仍保留 statistics scale 转换，机器代码仍不相同。这一负结果说明仅补 pass 不够：前端运算图不同会改变后向重物化选择。

最终 [g5_vllm_literal3.py](../../../../phase11/operator_gpu/g5_vllm_literal3.py) 显式保留 U 的两个统计量分支、向量 offset 的运算顺序、mask 写法、先 BF16 后转换，以及输出指针计算的位置，再经过诊断后缀编译。三个 vLLM 几何全部通过可执行区段等价门禁。`literal`、`literal2` 的未通过结果均原样保留；literal2 已有两组一致，剩余 `s32_d128` 的 LLVM/PTX 差异是输出地址计算位于 BF16 转换/共享内存转换之前或之后。literal3 将其移回 U 的位置后三组均一致，未据此声称任何单独的纳秒收益。

## 4. SGLang decode stage1：循环内的转换与不变量外提

以 `C02_n64_d128` 为例，旧 G0 有 9 个 conversion，U 有 2 个。旧 G0 中还存在相同 layout 到自身的 identity conversion；其余多出来的转换涉及 d 索引、mask、输出指针与 Query。

逐步干预观察：

1. common canonicalize 将 9 个 conversion 减至 5 个；
2. RMLC 将 5 个减至 2 个，重建索引、mask、指针；
3. 此时 Query 的 BF16 `blocked -> SliceLayout` 转换仍在 KV 的 `scf.for` 内；
4. Triton LICM 把 Query 转换及其他循环不变量移到循环外。最终诊断后缀生成与 U 完全相同的可执行区段。

因此问题不只是 conversion 总数。`G0_rmlc` 与 U 都剩 2 个转换，但前者的 Query 转换仍每轮执行，位置不等价。证据保留在 [G0_fullsuffix/intervention_steps](pilot/attempt_001/export/C02_n64_d128/G0_fullsuffix/intervention_steps)，特别是 `02_remove_layout_conversions` 与 `05_triton_licm` 的 before/after。U 的原生 [039.before](pass_checkpoints/C02_n64_d128/U/039.before.mlir) / [039.after](pass_checkpoints/C02_n64_d128/U/039.after.mlir) 也记录了外提。

三组 d64/d128/d256 都通过最终代码等价检查。d128 寄存器由旧 G0 的 80 回到 U 的 83；d256 由 144 回到 U 的 127。不能概括成“少用寄存器就一定快”，循环位置和寄存器生存期都改变了。

## 5. LayerNorm backward dw/db merge：先窄化，再转换

三组 LayerNorm merge 的主输入 layout 一致，旧 G0 均有 5 个 conversion，U 均为 2 个：

- 旧 G0 转换一个输出 mask、两个输出指针，以及两个 FP32 规约结果；
- U 在目标 layout 重建 mask/指针，并只转换两个已经窄化的 BF16 结果；
- RMLC 消除了多余转换并改变数据转换的位宽；LICM 外提循环内可重用的 range、广播/指针等表达式。

证据：[U 第一次 RMLC](pass_checkpoints/C01_p256_n1024/U/017.diff)、[U LICM](pass_checkpoints/C01_p256_n1024/U/030.diff) 与 [G0 后缀逐步 IR](pilot/attempt_001/export/C01_p256_n1024/G0_fullsuffix/intervention_steps)。

`G0_rmlc` 的转换数与 U 一样仍不足以保证代码相同；整个诊断后缀后才通过。后缀包含 instruction reorder，但在这两个代表案例中，该步没有修改 IR；本报告不把“流水线里存在的 pass”都说成已证实的原因。步骤实际变化记录在 `compiler_analysis.json` 的 `intervention_steps` 中。

三个代表案例还做了对原 U 删除全部 RMLC 的反向干预，结果不是“Gluon 只要同 vec 就天然等价”：

| 干预 | vLLM s32/d256 conversion / n_regs | SGLang d128 conversion / n_regs | LN merge conversion / n_regs |
| --- | --- | --- | --- |
| U 原始 | 1 / 96 | 2 / 83 | 2 / 96 |
| 旧 G0 | 5 / 80 | 9 / 80 | 5 / 96 |
| G0 + common cleanup | 5 / 80 | 5 / 80 | 5 / 96 |
| G0 + cleanup + RMLC | 2 / 80 | 2 / 80 | 2 / 96 |
| G0 + 完整诊断后缀 | 2 / 80，未对齐 | 2 / 83，代码对齐 | 2 / 96，代码对齐 |
| U 删除 RMLC | 18 / 180 | 22 / 188 | 13 / 188 |

该对照验证 pass 会改变最终程序，但没有对各干预计时；不能把寄存器或静态 conversion 变化换算成性能贡献百分比。vLLM 需要第 3 节的源图修复，完整后缀本身没有完成其对齐。

## 6. 九组完整结果

conversion 列为最终 TTGIR 的静态操作数，寄存器为 cuobjdump 的物理分配；三项依次为 U / 旧 G0 / 修正基线。它们不代替动态 NCU 计数或计时。

| 几何 | conversion：U / 旧 G0 / 修正 | n_regs：U / 旧 G0 / 修正 | 修正后可执行区段、ABI、资源 |
| --- | --- | --- | --- |
| `C01_p128_n4096` | 2 / 5 / 2 | 96 / 96 / 96 | 一致 |
| `C01_p256_n1024` | 2 / 5 / 2 | 96 / 96 / 96 | 一致 |
| `C01_p96_n8192` | 2 / 5 / 2 | 96 / 96 / 96 | 一致 |
| `C02_n64_d128` | 2 / 9 / 2 | 83 / 80 / 83 | 一致 |
| `C02_n64_d256` | 2 / 9 / 2 | 127 / 144 / 127 | 一致 |
| `C02_n64_d64` | 1 / 9 / 1 | 56 / 56 / 56 | 一致 |
| `C06_s16_d128` | 1 / 5 / 1 | 32 / 32 / 32 | 一致 |
| `C06_s32_d128` | 1 / 5 / 1 | 56 / 48 / 56 | 一致 |
| `C06_s32_d256` | 1 / 5 / 1 | 96 / 80 / 96 | 一致 |

普通的 TTGIR 文本规范化比较仍为不等：常量排列、无效 identity reshape 等文本差异尚在。最终门禁比较的是完整机器指令与所有已分配 ELF 区段、参数类型/次序及 launch 资源，而不是只比较转换数、指令计数、输入 layout 或计时接近程度。CUBIN 文件的 debug 信息和源码位置可以不同，不能要求整个容器文件的 SHA 相同。

## 7. 真实调用 ABI 补验

G4b 与原始默认编译还存在参数特化差异，不能默认为已覆盖。新增 CPU 对照包括：

| 编译形式 | 补验内容 | 结果 |
| --- | --- | --- |
| vLLM s16/d128，batch=1 | `num_seqs=1` 被调用特化 | 与原 U 一致 |
| vLLM s32/d256，batch=1 | `num_seqs=1` 被调用特化 | 与原 U 一致 |
| SGLang d128，真实调用 | 实际 stride 宽度、辅助参数特化 | 与原 U 一致 |
| SGLang d64，真实调用 | 实际 stride 宽度、辅助参数特化 | 与原 U 一致 |

这四种编译形式与前述默认编译合起来，逐项映射回 G4b 的 10 个工作负载全部通过；原始 U 的四份新 CUBIN 也与 G4b 冻结文件 SHA 一致。详见 [actual_ABI_analysis.json](actual_ABI_analysis.json)，其中 `G4b_workload_bindings` 明确记录了每一个 workload 使用的修正产物。

## 8. 对旧结论的修正与后续实验门禁

原始 G4b 中，vLLM s32/d256、batch=8、length=8192 为：U `7.652267 us`，旧 G0 `7.293867 us`，旧目标 `7.249067 us`。旧 G0 相对 U 的差异为 `4.6836%`，目标相对旧 G0 为 `0.6142%`，目标相对 U 为 `5.2690%`。本阶段已证明 U 与旧 G0 是不同生成程序；不能把总 `5.2690%` 都说成候选 vec/placement 的收益，也不能仅凭 conversion/寄存器数把 `4.6836%` 分摊给各 pass。

原始计时、NCU 和候选源代码保持原样，依然是那些具体程序的测量。GL 对旧 G0 的差分描述的是旧 Gluon 图上的 layout 敏感性；从该结果外推到原始 Triton，需要新的对齐基线与候选测量。这里没有重写历史报告。

后续应执行：

1. 默认 Gluon 必须先通过本阶段的 U 等价门禁；每个实际 ABI、特化及编译选项分别绑定。
2. 从这个已对齐源图出发改变候选布局；重新审核所有关键输入、统计量、输出布局及消费者。
3. 诊断后缀含 RMLC，会重新选择布局。不能无条件给全部 Gluon 候选启用，再假定 requested layout 被保留；必须检查实际生效布局，未保留则该候选不能用于对应的因果比较。
4. 等候选门禁、正确性与冻结产物齐备后，才能新测 U / 对齐 G0 / GL。当前没有对这些新候选作性能断言。

本阶段证明的是“在现有固定编译形式中可以构造出等价的 Gluon 默认基线”。没有证明 Gluon 默认编译链与 Triton 天然相同，也没有把实验后端后缀当作可合入的全局编译器修复。

## 9. 证据、缓存与验证

`preregistration` 保留每次 CPU 调查前的协议及源码快照；18 份原生 pass trace 无损 gzip 保存，并记录解压内容 SHA。`compiler_analysis.json` 包含 45 个非恢复程序与全部 pass 轨迹/干预；真实 ABI 的 8 个程序单列。

首次 CPU pilot 发现一个方法问题：Python 后端干预不进入原始内核 cache key。V1 的干预使用了共享内核缓存，故新增 V2 明确将干预放到独立目录，并重新编译恢复了三个 pilot 的六个 U/G0 受影响 key。恢复产物均与原始 CUBIN SHA 相同，恢复 key 逐一与先前干预 key 对上。初始记录、恢复记录、修订前后 worker 快照全部保留。后续干预均隔离缓存。没有 GPU 在这些临时干预产物上运行，也没有原生库重编译。

每个远端任务均记录 `ccache_before == ccache_after`，复用既有原生构建。前序 G3/G4 最终 validator 也已通过，所有保护清单保持不变。本阶段最终校验见 [final_validation.json](final_validation.json)，复核命令见 [REPRODUCE.md](REPRODUCE.md)。
