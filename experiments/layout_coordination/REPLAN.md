# 跨阶段布局协调：重新规划

状态：供复核的研究与实现方案。2026-11-05 完成只读源码/证据核查；以下新增实验和编译器实现均未启动。分支为 `explore/layout-coordination`。

## 1. 问题与目标

现有证据支持把问题范围扩展到 **SSA 数据流中的布局协调**：一个值的线程归属有利于输入访问，却可能增加后续 reshape/interleave、广播、转换、规约或输出的代价。更合适的布局需要同时考虑这些约束及资源。

TMA 与 reduction 仍是具有额外约束的实例；它们不再是总问题的限定条件。原始版本回归与本轮发现的默认布局优化空间保持区分。

目标是构建一套可复用的规划框架，而非预先承诺一个固定 vec 或一个纯静态规则覆盖所有执行条件。框架应统一描述合法候选、实际生成路径与可用上下文，再以可验证的策略选择。当前没有证据保证所有已见或未知算子都能获益。

范围先限于一个 kernel 内的透明数据流区域，不同时解决跨 kernel 融合、全模型调度、tile/warps 自动选择。第一轮固定算法、tile、warps、dtype、mask、stride 与数学表达，只改变合法所有权和由此产生的转换。

## 2. 为什么深入 LinearLayout，但不先重写它

LinearLayout 以硬件坐标位到逻辑元素坐标位的映射表达分布，已有组合、逆/伪逆、shape 传播与转换代码生成能力。相关论文还包含向量化、shuffle 和 shared swizzle 的生成优化；这部分能力应复用。[Linear Layouts, ASPLOS 2026](https://arxiv.org/html/2505.23819v5)

需要补的是其上层的**候选与选择策略**，以及分析与 lowering 共享的目标执行计划。只把 BlockedEncoding 换成 LinearLayout，无法自动知道哪一个布局在完整 kernel 中更快。

当前源码已有图传播、冲突消解、backward rematerialization、cast/broadcast hoisting，不能以“编译器没有全局传播”为问题前提。核查了三个版本：

- 实验 compiler：`20d9d98fead78c2f65e1b3ba5a7881d86a836dad`。
- 当前实验分支原 HEAD：`706602f87e098f404c01e2d164f3c63ae17798c0`。其 C++ 树含旧 forced-vec 实验 hook，不能当成上述测量 compiler。
- 本次在线核查的 upstream：`f8cddb947b471d6902a39f62d0894d1e552e5e75`。只读核查，不把旧测量转称为该版本的性能结果。

| 现有实现与可核查位置 | 对方案的影响 |
|---|---|
| upstream `Coalesce.cpp:77–118`、`CoalesceUtils.cpp:39–169` 已有访存切片分析、vectorization 汇总和 store 自身限制，最终给出 BlockedEncoding | 改进应复用这些分析；不能说它完全不看 store，也不能只在每个 load 上重新枚举 vec |
| upstream `RemoveLayoutConversions.cpp:384–398` 的冲突选择仍以候选顺序和 Blocked/MMA 类别偏好处理 | 检查是否需要 consumer 提供的反向候选及更合理的冲突选择 |
| 同文件 `getConvertCost:1007–1023` 对非寄存器重排使用 shared round-trip 的粗估 | 应查询真实 target conversion plan；但只改此成本函数不能产生缺失的 memory-anchor 候选 |
| `minimalCvtLayout`、`cvtNeedsWarpShuffle`、`GenericSwizzling` 已有通信范围、shuffle 分解和 shared swizzle/bank 工具 | 共用这些例程，避免另写一个与实际后端分叉的预测器 |

以上行号对应冻结 upstream SHA；详细 blob/SHA 与本地、实验版本对照见 `reference_audit.json`。

必须补充的输入：

- LinearLayout 加 AxisInfo、地址表达式、对齐、mask 与参与线程，才能分析实际 global 请求；任意运行时 stride/gather 不能只靠 GF(2) 映射精确推出。
- Shared padding 不包含在 LinearLayout 的线性部分，需保留物理布局、descriptor view 及 padding 信息。
- 后端实际 shuffle/shared 路径不只取决于最小通信范围，还受分解及目标规则影响。
- 目标规则按设备/指令查询，例如当前 NVIDIA global load/store lowering 区分 H100 的 128-bit 与满足条件的 SM100+ 的 256-bit；不能把 H100 的位宽假设写成跨架构定律。
- 寄存器物理分配、驻留容量、实际 active warps 与 cache residency 不是同一个量。静态阶段不能冒充已经取得 ptxas 或 NCU 真值。

## 3. 统一框架的结构

建议统一四个接口，而不是先构造一个新总分：

| 层 | 输入与输出 |
|---|---|
| 布局/语义约束 | SSA 值和算子端口、shape/dtype、别名、memory view、显式布局/同步约束；输出合法布局及传播关系 |
| 有界候选生成 | 默认、输入友好、消费者/输出友好、传播后协调的种子；输出有限的区域联合赋值 |
| 目标执行计划 | 源/目标布局、dtype、target、参与度与物理 shared 信息；输出 register/shuffle/shared 路径、访问分组、scratch 和同步结构 |
| 选择策略 | 上述结构、可观察的尺寸/上下文及已校准响应；输出默认、候选集合或有限多版本选择 |

目标执行计划先描述“会生成什么”，不直接把每个指令乘一个人为权重。优先提取纯规划例程，让分析与 lowering 消费同一逻辑；lowering 仍只检查当前 operation，跨 operation 的推理放在独立 analysis/transform 中。

候选变量需覆盖 producer 输出和 consumer 输入端口。允许在不同位置转换，也允许一个转换服务多个消费者；必须考虑支配关系、fanout 与跨算子的活跃值。不能把各 use 独立计价后重复计算同一次交换。

第一版区域包含 elementwise、cast、broadcast、reshape/trans、join/split 等可解释的数据流。MMA、复杂控制流、显式异步同步、跨 CTA/warp-specialized 边界先作为固定约束；reduction 和 descriptor/TMA 以显式约束逐步纳入，保留原始回归作为回归集。

例如 gate 的默认路径可概括为：

```text
BF16 load 的所有权 A → A 到 B 的交换 → FP32/softplus 的所有权 B → store B
```

通用候选包括让输入采用 B、让消费者采用从 A 合法传播的布局，以及保留默认。合法性和实际请求均需核对。这里使用 cast/consumer/ownership 关系生成候选，不为 KDA 函数名写特例。

LinearLayout 可用于跨 shape 操作传播种子、生成少量 register/lane/warp 基变化；不无界枚举所有二进制矩阵。广播和重复所有权必须合法保留，不能要求所有布局都一一映射。首轮预算暂定每 region 最多 8 个种子、每 kernel 最多 16 个联合方案，超限采用确定性的裁剪并记录被裁剪项。只有证明等价后才能去重；完整 CUBIN/ABI/resource alias 单独记账。

## 4. 最关键的信息边界：通用框架不等于无条件静态选择

AWQ 的两种尺寸给出一个具体实例：N4 `3584×3584` 上，`packed_compute_vec1` 相对 U 快约 6.42%；N5 `8192×8192` 上，同一候选约慢 1.955%，后者仍在 ±3% 实用带内。

本次进一步比较了两个条件的 U 与 GL：**分别是完全相同的 CUBIN**，运行 ABI 保留 `num_rows/num_cols`，不是把全图尺寸固化进该二进制。自动核查记录见 `reference_audit.json`。

这说明：只看当前 tile/layout/dtype 和相同 specialization 的选择器，无法为这两个观测条件主动作出不同决策。它并不证明不存在第三个兼顾两者的候选，也不单独证明缓存是排序反向的唯一原因。接下来仍需同 worker、匹配运行条件的尺度/cache 对照辨识原因。

选择策略必须明确分层：

1. **编译器静态层**：在现有信息下生成、裁剪和比较候选。上下文不足时允许保留默认或可比集合，并公开由此漏失的收益；全部弃权不能宣称模型有效。
2. **有上下文的选择层**：调用方可提供全局尺寸、stride 或复用信息时，使用显式选择元数据/JIT key/host dispatch。优先保持原 kernel 的运行 ABI，避免把 constexpr 常量折叠收益混入布局收益；若修改 specialization，必须对默认与候选同条件处理并重新对齐 U/G0。
3. **可选的有限多版本层**：静态信息不足且值得额外编译预算时，保留默认加少量候选，用经授权的有限 autotuning 或上下文 dispatch 选择。版本数、缓存 key 和调优预算先冻结；这不能包装成单次编译的纯静态选择器。

全局尺寸与 footprint 可以是特征，不等于已知实际 cache residency。调用方复用、其他 kernel 和实际访问分组仍可能影响结果。不在没有证据的情况下给 warm/cold 情况指定概率，再用期望值隐藏最差场景。

## 5. 先比较哪些解决路线

| 路线 | 优先验证的价值 | 当前限制 |
|---|---|---|
| 小型、通用的 consumer-aware 传播/anchor 改进 | 复用 Coalesce/RLC，自动产生输出兼容的输入布局；可能以小改动覆盖主要收益 | 不能预先假设能处理需要联合改变多个端口的案例 |
| 有界区域联合搜索 | 处理多个候选、分支与转换位置的联动 | 编译预算及选择模型复杂；应先证明比小改写多覆盖实用收益 |
| 固定布局下改进 conversion lowering | 在同一个合法布局对上改进 shuffle/shared 生成 | 不会自动修复分散的 global store 请求或缺失的布局候选；单独作为对照 |
| 全图尺寸/上下文感知的有限多版本 | 处理静态信息不足、排序随执行条件变化的场景 | 需要可用上下文或调优预算，涉及 launcher/API，不能默认对所有调用开启 |

不预设大求解器一定有价值。现有 N01 大配置中，output-only 与 coordinated 的 N2 时间区间重叠；协调版本结构更简洁，但没有确认其时间实用地优于 output-only。若小的通用传播改进足以覆盖主要收益，就先以小改动推进 PR。

## 6. 分阶段推进与停点

| 阶段 | 要做什么 | 交付与继续条件 |
|---|---|---|
| R0：来源、信息与 pass 溯源 | 对固定测量版本与冻结的当前 upstream 分别审计；从 memory anchor、传播、冲突消解到后续 pass，记录每个残留交换的来源；列清编译期和启动期可见信息 | 输出逐 pass IR/布局来源图、信息契约。当前 upstream 的性能必须重新确认，旧数据只作历史证据。本地 forced-vec hook 不得混入新 compiler |
| R1：结构规划与候选覆盖 | CPU 优先复用 LL/AxisInfo/target planner；比较现有传播、小型 consumer-aware 改写和有界区域候选。用 Gluon 做显式对照，先验证候选能否表达有价值的程序 | 结构计划核对真实 lowering；U/G0 门先通过。随后在冻结预算内测候选覆盖/oracle，仅用于评价搜索空间，不能当可部署选择器。若局部方案已足够，不继续堆大 solver |
| R2：选择机制与上下文辨识 | 匹配 worker、shape、cache priming 的对照；以同条件 FULL 时间和候选相对默认差分为主要校准目标。microbench 解释局部机制，组合关系独立验证 | whole-geometry/whole-family 留出；报告误差、单侧上尾覆盖、改选率、regret、实用退化、默认保留率及资源覆盖。比较静态层与有上下文/有限多版本层，保留无收益和反向结果 |
| R3：最小编译器集成与泛化 | 根据 R0–R2 决定扩展 RLC/Coalesce 还是新增独立 region pass；分析与 lowering 共用 target plan；独立检验 conversion lowering 改进 | 先 feature flag 和 CPU/lit 正确性，再 H100；与 lowering 改进做交叉消融，防止混记收益。跨架构只用可获取的 Modal 设备，SM103 非必需。PR 创建前单独征得用户同意 |

当前 N1–N5、G8 及历史 TMA/reduction 数据全部属于已见证据。新盲测 kernel/geometry/context 在下一轮拟合或搜索调参前冻结；不能把已看过的零结果重新命名为盲测。

每阶段保留原始样本、IR/PTX/SASS/CUBIN、实际加载 SHA、错误/重试、源码和配置快照，完成 validator 后独立 commit。复用 ccache/native image；C++ 改动后先按仓库要求 rebuild，再运行对应 lit/pytest。纯 Python/文档阶段不重建 native。

最终 PR 从干净 upstream 拆出最小 patch 和必要测试；实验分支保留研究证据，不把历史 forced-vec hook、原始样本或整个实验目录带入上游 PR。

## 7. 评价准则与本次停点

- 候选生成质量与选择器质量分开报告：候选池是否包含好方案、选择器是否选对，是两个问题。
- 以默认布局、固定 vec1/2/4、简单 consumer-aware、conversion-only/resource-only 及区域方案作公平对照；不以“复杂模型必须胜出”或“放行已知赢家”选参。
- 正式时间使用无 profiler 的冻结 binary；NCU 单独采集。指令、request、sector、wavefront、DRAM 字节分别解释，不以 `32/vec` 代替实测，也不构造未校准加权总分。
- `n_regs`、q 容量和实际 active warps 分列；寄存器区间用真实编译结果校准，不复用未经校准的硬上界/双层裕量。运行时编译所有候选后取得资源真值的方案须明确记为多版本编译。
- 在可观测上下文或冻结假设范围内评价改选；不承诺所有执行场景严格支配默认。

本次完成的是改名和方案重写，不启动上述新 GPU 实验或 C++ 实现。建议下一步先推进 **R0 和 R1 的 CPU 部分**，以代码溯源和候选覆盖决定后续实现范围，再冻结 GPU 预算。
