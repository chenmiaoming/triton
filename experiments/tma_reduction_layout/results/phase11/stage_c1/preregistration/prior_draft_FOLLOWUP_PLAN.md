# Phase 11 后续方案：Gluon microbench 与布局选择 cost model

状态：2026-10-15 review draft，等待用户审阅。本文件不授权执行。
本次只编写方案和核对逻辑布局；没有启动 GPU、编译、profiling 或计时。
已完成的阶段 A/B、原始证据、冻结协议和失败记录保持原样。
本草案替代 `PLAN.md` 中尚未执行的 C/D 建议，不改写 A/B 的设计历史。

## 1. 问题、范围与固定路径

研究问题：TMA load 后做 partial max 时，怎样根据编译期可获得的信息，
权衡 shared → registers 访问、规约通信、输出布局转换和资源占用，选择输入布局？
模型必须预测尚未用于调优的组合；完整 kernel 的实测赢家只作为评价参考。

已核对 [原始 issue #9703](https://github.com/triton-lang/triton/issues/9703)：
host 端 `TensorDescriptor(x, ..., [1, BLOCK_M, BLOCK_N])`，
kernel 内 `x_desc.load([0, off_m, off_n])`，接 BF16 → FP32、`max(axis=1)` 和 store。
原报告为 GB300/SM103，tile 为 32×128、4 warps。
因此，后续完整 kernel 和含 TMA 的 microbench 全部采用 **host-created descriptor**。
descriptor 在 event 计时前构建，kernel 内没有 make_tensor_descriptor。
纯寄存器探针可以没有 TMA；这不构成另一种 descriptor 路径。

主问题先固定：

- BF16 输入、FP32 max 及输出；保留原源码中的 widening。
- 全局输入 `[1,8192,8192]`，tile 内沿 M 规约，每列产生一个输出。
- 核心开发几何：32×128/w4 与已有的 32×64/w8；后者用于研究已观察到的布局敏感性。
- 固定 upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`、未修改 native、assembler 和编译选项。
- 同一几何内固定 descriptor/shared layout、TMA 字节量与显式输出布局。
- 通过 Gluon `BlockedLayout` 和 `smem.load(register_layout)` 控制元素归属。
  本轮实验不修改 C++ compiler，不强制 LDS/LDSM opcode。
- 首版选择对象限定为普通 descriptor load → widening → 单一 partial max → 输出路径。
  gather/scatter、多用户、其他算子或跨 CTA 规约不自动纳入。

原 issue 使用 CUDA Graph 计时；本轮沿用已验证的单 kernel event 计时。
不同计时方式的历史绝对时间不直接相除。H100 上的现象不能替代原 SM103 的验证。
SM103 不作为建立模型、保留测试或准备限定架构改动的必要前置条件。
缺少 SM103 时，结论限定在实际测试的设备，不声称已经修复原 GB300 系统的回归。

硬件范围限定为当前账户实际能调度的 Modal GPU：H100 用于开发与同架构保留测试，
B200/SM100 和 RTX PRO 6000/SM120 用于迁移与资源边界验证。
H100 使用 `H100!` 避免自动升级到 H200；B200 不使用允许自动升级的 `B200+`，
所有结果仍核对实际设备名称、CC 与 Driver UUID。
2026-10-15 的 [Modal GPU 文档](https://modal.com/docs/guide/gpu)也列出 B300，
[NVIDIA CC 表](https://developer.nvidia.com/cuda/gpus)将其列为 SM103。
这只是可选扩展：目录列出不代表当前账户与时段一定能调度，暂不为此启动探针。
B300 若后续可用，可补同架构确认；其 CUDA 13.1+ 环境要求按官方文档另行核对。

## 2. 布局控制：vec 与同列线程的 warp 归属分开

初始候选为 vec1/2/4/8，每种包含 N-first 与 within-warp 两种分配。
去掉完全相同的布局后，两种几何共 **15 个布局**，不是 16 个。
32×128/w4 的 vec1 两种构造相同。完整 rank-3 参数及逐行归属见
`followup_review_matrix.json`。以下数组按 `[M,N]`，order 固定 N-first：

| 几何 | vec | N-first TPW / WPC | within-warp TPW / WPC | 同列贡献线程 | 同列涉及 warp，前 / 后 |
|---|---:|---|---|---:|---:|
| 32×128/w4 | 1 | [1,32] / [1,4] | 相同 | 1 | 1 / 1 |
| 32×128/w4 | 2 | [1,32] / [2,2] | [2,16] / [1,4] | 2 | 2 / 1 |
| 32×128/w4 | 4 | [1,32] / [4,1] | [4,8] / [1,4] | 4 | 4 / 1 |
| 32×128/w4 | 8 | [2,16] / [4,1] | [8,4] / [1,4] | 8 | 4 / 1 |
| 32×64/w8 | 1 | [1,32] / [4,2] | [4,8] / [1,8] | 4 | 4 / 1 |
| 32×64/w8 | 2 | [1,32] / [8,1] | [8,4] / [1,8] | 8 | 8 / 1 |
| 32×64/w8 | 4 | [2,16] / [8,1] | [16,2] / [1,8] | 16 | 8 / 1 |
| 32×64/w8 | 8 | [4,8] / [8,1] | [32,1] / [1,8] | 32 | 8 / 1 |

每组固定 vec 的干预保持同列贡献线程总数不变，改变其 lane/warp 归属。
它也可能改变实际加载、输出转换和资源；这些共同变化全部记录。
它不是天然的“只改变跨 warp 通信”的实验。
请求布局必须与实际 IR 元素归属一致；别名 binary 合并物理执行并保留逻辑身份。

## 3. Microbench 具体测什么

测量单位首先是**合法 Gluon 子图的有效延迟与吞吐**。
不预先承诺能单独得到每条裸 LDS、LDSM 或 shuffle 的真实延迟。
人工反馈、循环控制、输入准备和 checksum 都有成本，必须在分析中显式保留。

| 编号 | 子图与输入准备 | 改变的量 | 主要观察与建模用途 |
|---|---|---|---|
| ML | host TMA 填充固定 shared 缓冲，wait；循环读取指定布局，elementwise checksum，最后写出；没有 max | 15 布局；独立流 D=1/2；循环次数 K；驻留上限 | LDS/LDSM 的真实访问模板、wavefront、加载到 consumer 的有效成本，以及加载侧吞吐曲线 |
| MR | 输入一次进入寄存器并 widening；循环执行 gl.max 和有数据依赖的有界反馈；最后写出 | 15 布局；D=1/2；K；驻留上限 | 规约依赖链、线程内 max、warp 内交换、跨 warp shared/barrier 的组合成本与吞吐 |
| MO | 非常量、已按实际规约结果 SliceLayout 分布的向量，转换到固定输出布局并 store | 每个布局对应的结果布局；并发规模 | 输出路径的实际代码与资源。计时包含输入准备，不直接称为纯转换时间 |
| MC | 同一个 host TMA → shared-load → widening → max → 固定输出路径 | 15 布局；自然与受限驻留 | 检验 ML/MR/MO 组合后的预测；K=1 保留真实完整 kernel，作为开发验证 |

ML 的初稿使用两个固定 shared tile 槽位，所有 D 使用相同缓冲分配。
循环索引是 block-uniform 的 runtime 值，轮换合法 memdesc view；
各次 load 的数据进入可观察的 checksum。D=2 使用独立 accumulator。
缓冲容量、TMA 次数在同一比较中固定，不随 K 增长。
这个方法是否能保留预期动态 load，要由 C1 验证，不能只凭源码判断。
额外缓冲限制了可达到的 occupancy，因此必须记录其覆盖范围，
不能把低驻留微基准的参数直接外推到原 kernel 的高驻留区间。

MR 不会重复 `max(x)` 后求和来冒充重复规约。
每轮输入需要依赖上一轮结果，示意为：

```text
r_i = max(v_i, axis=M)
acc += r_i
v_(i+1) = a(m,i) * v_i + b(m,i) * broadcast(r_i) + c(m,n,i)
```

其中 a/b/c 由逻辑 tensor 坐标构造、随行变化且有界，避免溢出及简单幂等化。
广播、更新、额外转换与循环都作为 fixture 的真实工作记录。
配套控制程序给这些附加工作的成本提供约束；
不能把两个程序的时间相减直接命名为原 kernel 的纯 max 时间。
特别检查 BF16 widening 与 max 是否被合并、打包或重排：
若 MR 与原 kernel 的操作模板不同，它只能校准匹配的部分，不能整段套用。

MO 不反复执行可能被折叠的 `convert(A→B→A)`。
若独立 MO 的转换被消去、移动或变成不同模板，就只保留代码生成诊断，
输出成本由原 kernel 实际存在的相邻子图联合辨识。
联合项只能合并原程序中的工作，不能把人工反馈当成原程序成本。

还准备一组独立参数探针，用于把布局变化拆开，而非每个布局拟合一个系数：

| 探针 | 参数 | D | 逻辑程序上限 |
|---|---|---|---:|
| 线程内 max | 每线程同列值 p=1,2,4,8,16,32,64；同列不跨线程 | 1/2 | 14 |
| warp 内 max | 同列 lane 数 u=1,2,4,8,16,32；每线程同列一个值，不跨 warp | 1/2 | 12 |
| 跨 warp max | 同列 warp 数 w=1,2,4,8；每 warp 先有一个贡献值 | 1/2 | 8 |
| BF16→FP32/consumer | 每线程元素数 e=2,8,32；无规约 | 1/2 | 6 |

这些仍使用 Gluon 的合法 tensor 布局，让 compiler 生成交换与同步。
不在源码里读取 lane/warp ID，不引入硬件 ID scalar、PTX 时钟或手写访存汇编。
循环反馈的干扰若不能辨识，则报告可识别的组合成本/界限，停止声称单项参数已知。

核心矩阵上限：ML 30、MR 30、MO 15、MC 15，加参数探针 40，
合计 **130 个逻辑程序**；别名、无效布局、代码生成不匹配分别列出。
K 和额外 shared 预留是运行参数，不因此重新编译相同程序。
这是 C1 通过后才考虑的 C2 预算上限，不是这次立即启动的任务。

## 4. 怎么测，哪些结果才有效

### 重复、延迟与吞吐

- 初始 K 为 runtime 参数 `[1,16,64,256]`，同一 program 使用同一 CUBIN。
  K=1 用于连接原程序，斜率分析报告所有 K 的数据和残差。
- D=1 的依赖流、单 CTA 的 event 时间提供子图依赖链的有效成本约束；
  D=2 及多 CTA 提供延迟隐藏和吞吐约束。D 改变了活跃寄存器，单独记录。
  不能将 D=1 的时间直接称为裸指令延迟。
- 单 CTA grid=1；吞吐 grid=`64 × 实际 SM 数`。这是 microbench 配置，
  不将这个 grid 当作未来编译器的已知输入。
- 吞吐探针设置目标驻留上限 q=1/2/4/自然上限，去重并只执行可行配置。
  通过 launch 时额外预留未使用的动态 shared memory 限制驻留，
  不改变原 memdesc view 或插入无关计算；同步、地址和动态指令数需保持一致。
  它只提供资源受限的控制实验，实际驻留仍由资源和 NCU 核对。
- 额外预留遵守设备单 CTA 上限。无法达到的 q 保留为缺失，不能填补为理想数据。
- 同时固定可控制的 shared/L1 carveout 与函数属性，记录实际设置；
  若设备/driver 不能固定，则它与驻留的共同变化单独标记，不能解释为纯 occupancy 效应。
- 若 event 分辨率不足，统一扩大该 fixture 家族的 K 到预登记上限 1024，
  保留原尝试。只按量化精度/技术有效性决策，不能按哪个布局快来选择 K。

同一 binary、grid、D、q 下分析 `T(K)`，初始拟合截距与每次 body 的增量。
截距包含 launch、输入准备、TMA、最终写出等一次性工作，不能叫 TMA 纯延迟。
非线性或资源条件不匹配时保留成本曲线，不强制压成一个斜率。

### 必须通过的代码生成门

1. 完整输出与预先定义的 reference 一致。max 的 NaN 规则沿用固定 upstream 的 maxnum 语义；
   主计时用有限 quarter-pattern，另检验随机有限与浮点边界。
2. 同一 CUBIN 的 K=16/256 两点进行动态 source counter 核对。
   待测 body 的实际执行随 K 对应增长；记录 loop/consumer 的全部附加指令。
3. ML 的待测 load 不得被移出循环或替代成寄存器复用；MR 的 max 不得被合并。
   源码中的“重复次数”不构成执行次数证据。
4. 匹配原 kernel 的 shared layout、实际访问/规约模板；不匹配时标注范围，不能冒充复现。
5. 初始无 spill 的探针用于常规参数校准；发生 spill 的配置保留为资源边界数据，
   不悄悄删除后再给全域模型。首版对预测存在 spill 的候选回退。
6. 同一 binary 在不同 K 上的资源天然固定；D、布局和程序变体的资源逐个记录。
7. 若合法 Gluon 程序无法保持某种预期干预，C1 停下汇报，修订方案。
   不临时改 compiler、注入隐藏线程 ID 或手改 PTX 来补齐矩阵。

### 计时与 NCU

校准用 3 次独立 worker 调用，每 cell 固定 20 个单 kernel event samples、
固定 warmup 与平衡顺序。正式 D 使用 7 次调用和原有配对设计。
launch 样本不是独立 GPU，记录实际 Driver UUID、时钟/功耗/温度及环境变化。
不使用 profiler duration 替代独立计时，也不把多个 kernel 批次除以次数。
首版用 ns/us 计量，不用标称频率把 event 时间冒充精确的指令 cycle 延迟。

NCU 软件/source 与硬件 pass 分开；archive CUBIN、参数、grid、K、D、q 和 source SHA。
所有重复 fixture 做两点动态执行核对；正式硬件采集优先覆盖 ML/MR 的 D=1 模板，
在 q=1/自然上限各固定 3 次，并核对 MC 的对应 PC。
只有新模板或未解释的问题才扩展采集，扩展前追加协议。

记录实际 shared 动态 warp 指令、wavefront、可用的 ideal/excessive、bank conflict、
shuffle/转换/max/barrier、SMEM、寄存器、spill、occupancy、TMA/global 流量。
用 PC/source 区分初始 load、规约临时访问和输出转换，无法归属的保持 unattributed。
直接 shared Requests 在阶段 A 不可用，继续记为缺失，不用推算值冒充硬件指标。
matrix conflict 正对照尚未完成，因此其 bank-conflict 指标不作为首版模型的唯一依据。
优先用已验证覆盖的动态执行与 wavefront，再检查其与访问预测是否对应。

NVIDIA 区分 shared 的 Instructions、Requests 和 Wavefronts；
宽访问本身可能需要多个 wavefront，不能把全部 wavefront 都当冲突。
见 [Nsight Compute shared-memory table](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#shared-memory)。
software instrumentation 会影响行为与耗时，见
[Metric collection / replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#replay)。

## 5. 从测量建立可执行的选择模型

### 5.1 编译期特征与静态操作计划

输入是 shape、dtype、reduce axis、完整 register/shared layout、warp 数、目标架构、
已知的输入/输出转换和数据依赖。先用 Python 构建独立的只读分析原型，
不通过改 lowering 观察其他 op，也不在候选预测时启动 GPU。

静态计划至少预测：

- shared 访问的地址/字节分组、谓词和可能的 LDS/LDSM 模板；
  根据 layout 与实际 lowering 的访问构造来算，禁止用 `32/vec` 代替。
- 实际线程内 combine 树、warp 内交换、跨 warp shared/barrier 与输出转换拓扑；
  contributors 或 warp 数只是结构输入，不能单独代表时间。
- widening/packed max 等真实 codegen 模式。
- scratch/shared 分配与活跃 tensor 值，用于 SMEM 和寄存器估计。

在开发 artifacts 上验证静态计划对指令种类、访问和依赖的预测，再接 NCU 检查动态量。
测试 case 的真实 SASS、NCU、寄存器数和时间不能进入正常预测。
若只有读取最终 binary 才能得到某特征，应列为“使用最终 binary 的诊断上界”，
不把它混进声称可在早期 compiler pass 使用的模型。

早期 register estimate 与寄存器分配后数量不完全相同。
首版使用开发数据验证的估计区间，推导可能驻留范围；
预测越过资源上限、出现 spill 或对估计不确定性敏感时回退默认。
SM120 等设备上不可行的候选属于合法性/资源门，不能给它一个很大时间参加排名。

### 5.2 校准什么参数

每架构校准一组共享参数：访问模板的有效延迟/服务速率、max/转换/交换子图的成本、
barrier 的参与/等待关系，以及随独立工作和驻留变化的服务能力。
参数有 ns、warp-operation/ns、wavefront/ns 等单位，不再使用无量纲总分。

依赖链与吞吐探针共同约束参数；反馈、checksum、loop 的工作也在校准图里。
用开发数据做受约束拟合，给出参数区间与残差；
通过留出独立参数组合检验可辨识性，不给 15 个布局分别拟合自由权重。
若两个参数无法分别辨识，则只合并原 kernel 中实际相邻且被共同测量的模板，
或给出区间；不能靠任意固定一个权重得到看似确定的结果。

### 5.3 首版模型如何组合成本

首版采用小型的**依赖与资源调度模型**，它是一项待验证假设：

1. 静态操作计划生成每 warp 的依赖图，节点是匹配的 load、combine、exchange、
   convert、store 或经过验证的组合模板。
2. 节点只有在前驱完成、对应服务资源可用时开始。
   延迟来自校准的有效延迟；资源占用时间来自预测服务需求除以校准服务速率。
   例如 wavefront 的服务时间与依赖链延迟分开处理，不能互相代替。
3. barrier 等待其参与 warp 的前序工作。允许独立 warp、列和 CTA 重叠。
   明确调度/发射假设，并用不同 D/q 的混合 microbench 检验这些假设。
4. 根据静态资源估计放入可驻留 CTA，输出单 CTA 完成延迟及稳态 SM tile 完成间隔。
   TMA 是固定 host 路径的共同前序，仍保留其依赖和资源交互；
   不把两个 fixture 的截距之差当作 TMA 参数。
5. 用 ML+MR+MO 的组合及 MC 检验预测。过不了组合检验就修订开发模型，
   不使用盲测时间补权重，也不声称已具备解释机制。

这不是逐 cycle 模拟全部 GPU 的项目；仅建当前子图所需的操作模板和资源约束。
首版不预设“几个 isolated latency 相加”，也不预设“取几个总数的 max 就等于时间”。
完全串行的路径按依赖累积，独立工作在调度模型中重叠。
额外残差不得按具体 shape/layout 身份记忆；确需新增结构项时只用开发数据、登记版本。

模型最终对每个合法候选预测成本及不确定性，选出稳健更优的布局。
编译期不知道 grid 时不能偷用实验中的 full-grid 数值：
首版同时检查单 CTA 延迟与稳态服务成本；两者出现无法凭静态上下文解决的交叉，
或在可能驻留区间内结论反转，就保留默认并报告覆盖率。
不得通过“全部回退默认”宣称模型有效：必须分别报告支持范围、实际改选率和改选收益。

## 6. 开发、冻结与评价

旧阶段所有已看过的完整 kernel 结果（包括本轮 A/B）都视为开发/复现数据。
不能给它们换种子或重新计时后叫 held-out shape。

拟定未见组合候选池：BM=8/16/32/64/128，BN=32/64/128/256/512，warps=2/4/8。
先做纯元数据历史审计，排除过去已经用于选择/调优的组合，并通过布局合法性静态门。
从余下组合按覆盖线程内值数、同列 lane/warp 数与资源边界选 8 个，
在 C2 正式校准前冻结 case 列表和 hash。参数探针可覆盖相同原语，
但不能提前测这些完整 kernel 的优劣。
如果没有 8 个合格的新组合，先汇报并修订候选池；不把旧组合悄悄改称盲测。

候选规则由相同的显式布局生成器产生，不按模型偏好删掉合法竞争布局。
每 case 至少包含 upstream 默认布局；计时前冻结模型参数、版本和预测选择。
正式评价使用完整 host-descriptor kernel，full grid 为主、1/16 grid 为诊断。
新候选与普通 Triton 的对应关系也需重新核对；默认 bridge 不能证明任意新布局等价。

评价并列给出：

- 默认布局、固定 vec 基线、只看规约 fanout 的简单基线、完整模型。
  这些是用于判断新模型是否增加价值的基线，不是重新研究旧计分函数。
- 选择 regret：`T(模型选择) / min(T(同一候选集)) - 1`，同时给测量不确定性。
  实测最小值是有噪声的 oracle 参考，不能假定它是绝对真最优。
- 相对默认的收益/退化、全部 case 的中位与最坏 regret、改选覆盖率及回退原因。
- 对有明确优化空间的 case，模型取得多少收益；差异小的 case 保留可比最优集合。
- 静态访问预测误差、时间/排名误差、去掉加载/通信/资源项的消融。

沿用 3% 作为实验的实用差异口径，不把它说成理论阈值或普遍非回归保证。
性能主比较按独立 worker 的配对统计形成区间，不能把全部 launch 当独立样本。
同时报告全部原始点和不确定结果，不只报均值，也不只报选对率。

模型若在盲测失败，保留失败并标记该版本；修改后必须另留新的测试集合。
不能反复在同一批盲测上调参后继续称其 held-out 验证。

## 7. 分阶段执行与停点

| 步骤 | 工作与范围 | 完成产物 / 进入下一步的条件 |
|---|---|---|
| C1：最小可行性 | H100；每个几何只用 natural vec1/2/8 与 within vec8，共 8 布局；ML/MR，D=1，共最多 16 个程序 | correctness、动态重复、实际模板与资源检查；能测什么/不能测什么的报告。失败先停，不直接铺开 130 个程序 |
| C2：正式校准 | 冻结数据划分和协议；上述不超过 130 个逻辑程序；3 次校准 worker | 原始 event/NCU/全部 artifacts、成本曲线、静态预测核对、参数辨识与组合误差；若缺关键可辨识成本则先调整方案 |
| C3：模型原型 | 只用开发数据实现静态计划、校准参数和依赖/资源组合；不改 C++ | 可调用的 Python 选择器、基线/消融、参数与模型冻结、8 个测试 case 的预先预测 |
| D1：同架构保留测试 | 冻结后的 H100 8 个 case；7 次调用 | 选择 regret、收益/退化与覆盖率，确认模型是否真能推广到未见组合 |
| D2：可用架构迁移 | Modal 实际可调度的 B200/SM100、PRO6000/SM120；B300/SM103 条件允许时追加 | 分开报告参数直接迁移、仅重校准硬件参数的迁移；资源失败作为边界证据。SM103 不阻塞 C/D，缺测时不推断其收益 |

每个完成的子阶段单独 commit，保留执行源代码与原始证据；阶段边界先汇报结果。
C1 通过不自动意味着后续模型成立；C3 冻结前不启动 D。
当前只提交这份 review draft 供用户决定，尚未开始 C1。
PR 创建/更新和扩大到其他算子均不在本轮授权内。

## 8. 缓存、证据与运行预算约束

- 复用已验证的未修改 native 镜像与持久 kernel cache；使用匹配该 native 的 Gluon Python 源码。
  本地旧实验 C++/环境开关不能混入执行环境。记录 source/native/assembler SHA。
- 仅写 Python 和显式布局不运行 make，不重编译 Triton native；保留 cache hit/miss 与 ccache 前后统计。
- 新 constexpr 才产生新 kernel；runtime K、额外 shared 预留等复用既有 CUBIN。
- 正式计时只加载冻结 binary；不导入 Triton、不 JIT、不编译、不启动编译子进程。
- raw samples、原始 compiler artifacts、NCU report 和失败调用 write-once；修订追加编号。
  数据解析与模型等 derived artifacts 可以版本化更新，禁止覆盖原样本迎合预测。
- 复用已校验的 metric mapping；不重复保存庞大 query 输出，不对所有架构先铺完整矩阵。
- C1 首批最多 16 个程序；C2 的 130 是逻辑上限，K/q 不扩编译数量。
  未知 GPU 小时与费用不虚构估计，C1 报告实测开销后再给 C2 预算。
- 不足的额度、不可用的目标或权限问题按实际原因汇报，不自行切换账户或扩展重试。

参考源码：`phase11/kernels.py`、`phase11/launch.py`、已冻结的 A/B 结果；
Gluon API 与同步方式参照 [官方 TMA 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html)，
具体执行以固定 upstream 的源码为准。
