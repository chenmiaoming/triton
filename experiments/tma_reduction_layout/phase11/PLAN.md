# Phase 11：布局性能与回归定位实验方案

状态：2026-10-15 review draft。尚未冻结执行协议，尚未启动新GPU任务。

目的：解决descriptor/TMA load之后按列partial max的性能问题。先测清不同布局在什么条件下更快，再通过受控干预寻找可用于编译器改动的结构依据。本轮不使用旧计分函数选择候选、不拟合分项权重、不做旧模型的正确性评估，也不修改PPT。

## 要回答的问题

1. 同一个显式Gluon kernel、输入、输出、tile、warp数和编译上下文下，vec1/2/4/8的实际性能分别怎样？这些实验与普通Triton的默认代码生成能否建立复现对应关系？
2. 差异主要伴随初始shared加载的请求/wavefront变化，还是规约通信、输出转换、寄存器/SMEM/occupancy变化？
3. 保持vec不变，把同列贡献线程从多个warp移到一个warp，完整kernel有什么变化？
4. 若NCU证据指向初始shared-load指令选择，下一轮如何登记针对该选择的干预？当前一轮不预先加入compiler loader开关。
5. 哪些结论能在H100、B200、SM120复现，哪些只能限定在某架构/代码生成条件下？

性能比较直接使用运行时间；NCU计数用于定位和解释实际访问。PTX/SASS用于确认最终指令、宽度、地址计算和位置，不用静态条数代替运行时请求、wavefront或延迟。单凭计数差异也不把原因命名为“硬件融合”：先核对最终SASS、路径/谓词、计数覆盖和可能的调度影响。

## 第一轮固定矩阵

全局输入`[1,8192,8192]`，输出FP32 `[256,8192]`，沿M取max。源码仍是合法block program，descriptor origin为block-uniform，不引入source-level硬件ID或lane-varying scalar。

| tile | warps/CTA | 用途 |
|---|---:|---|
| 32×128 | 4 | 原始reproducer的几何；主要问题入口 |
| 32×64 | 8 | 已有实验中的另一个具体布局问题入口 |

每个几何覆盖BF16/max与FP32/max，host/device descriptor各一条路径，共8个条件。BF16/max是主问题；FP32/max保持同一数学操作，用来观察dtype、packed max及widening相关代码生成差异。FP32输入字节量是BF16的两倍，因此它是诊断对照，不把两种dtype的耗时差当成纯转换代价。

每条件包含两种明确分开的实验对象：未修改upstream编译的普通Triton kernel作为问题锚点，以及同一个Gluon kernel通过显式BlockedLayout指定vec1/2/4/8的四个候选。四个候选使用同一Gluon pipeline；普通Triton与Gluon不是同一pipeline，不能将跨pipeline时间差直接解释为布局的因果收益。Triton锚点记录实际observed encoding，不假设所有dtype/架构默认都是vec8。没有旧规则的自动选择组。

核心规模：每target 8个Triton锚点加32个Gluon候选，共40个逻辑artifact身份；H100/B200/SM120共120。若复现Triton锚点需要四个候选以外的Gluon布局，另登记最多8个复现身份，不悄悄加进四候选的主排名。若CUBIN、函数、ABI、资源及参数完全相同，保留身份并合并物理执行，避免把相同binary的噪声当成布局收益。请求布局与最终TTGIR的实际布局不符，标记干预失败，不用请求vec代替实际vec。

## 公平控制与源码

- 固定upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`，先复用同一版本以便与已有编译证据衔接。历史3.5.1/3.6.0和原GB300报告仅作背景；本轮不跨历史版本混合比较。
- 主实验沿用已有Gluon显式布局方法：`gl.BlockedLayout`作为`gl.constexpr`参数传入kernel，在TMA完成并执行合法mbarrier wait后调用`smem.load(register_layout)`。通过`size_per_thread`、`threads_per_warp`、`warps_per_cta`和`order`指定元素归属，后续`gl.max`使用这个tensor的布局。无需修改C++编译器、加入自动选择pass、手改IR或注入lane/warp ID。
- 尽可能固定descriptor/shared布局、TMA逻辑数据量、dtype、数学操作、CTA grid、warp数、assembler、编译选项和输出路径。若任何后续项实际改变，完整记录并作为共同变化分析。
- source、native、ptxas、TTIR/TTGIR/LLIR/PTX/SASS/CUBIN、资源、实际layout与worker代码SHA均归档。不得手改归档PTX/SASS/CUBIN或原始samples。
- 复用与固定upstream对应的未修改baseline native、镜像与ccache，执行前核对native/source SHA，不能误用含旧计分原型的patch镜像。改变Gluon constexpr会编译新的kernel，不需要重新编译Triton；只有Python/方案变化时按AGENTS不运行make。以后若另行登记native改动，才按AGENTS先make并做相应lit/pytest。

现有实现位于`../gluon/kernel.py`与`../phase7/kernels.py`。例如对`[1,32,128]` tile，以下两个rank-3布局都使用4个warp、128个线程，vec都为8，但同列线程的warp归属不同：

```python
# 维度顺序：[batch, M, N]；order按N、M、batch分配。
spread = gl.BlockedLayout([1, 1, 8], [1, 2, 16], [1, 4, 1], [2, 1, 0])
within = gl.BlockedLayout([1, 1, 8], [1, 8, 4], [1, 1, 4], [2, 1, 0])
# kernel内部，在同一TMA加载及mbarrier等待之后：
x = smem.load(register_layout)  # register_layout为spread或within，gl.constexpr
y = gl.max(x.to(gl.float32), axis=1)
```

第一种布局一列的8个贡献线程分布在4个warp，第二种仍是8个贡献线程但都在一个warp。既有harness的输入组织、descriptor origin和store需要适配本轮`[1,8192,8192]`与grid；复用显式布局方法，不直接把旧kernel当作已适配的新reproducer。Gluon候选使用相同的显式输出布局和store语义；布局改变导致的输出转换、lowering及资源变化仍需记录。

在把Gluon结果用于原始Triton问题前，单独登记canonical↔Gluon复现核对：logical shape、axis、num_warps、完整blocked/shared encoding、完整ordered LocalLoad sequence、含相关immediates的reduction fingerprint、输出转换/store和资源。布局相同不足以证明二进制或性能相同；代码生成、NCU行为及独立计时的差异都要保留。不能建立对应关系时，结果只支持所测Gluon程序，不能声称原始Triton回归已解决。复现核对与四候选性能对照使用分开的evidence record。

## 阶段A：工具与计数口径检查

先用已归档的两个布局binary做小规模探针，确认目标平台可运行NCU；再用已知访问行为的校准fixture核对计数。fixture仅用于检查仪器，不作为Triton编译器改动的理由。

1. 保留ncu版本、query-metrics/sections输出、driver版本、GPU名称/CC、CUDA Driver UUID、PCI/设备映射、call id、时间、完整命令和退出状态。
2. 分别检查普通shared load与matrix load（LDS/LDSM/实际相应SASS）的动态执行、Requests、Wavefronts计数覆盖；不要只查旧版的一个bank-conflict聚合指标。
3. 普通load的无冲突与有冲突对照校验bank/wavefront计数。matrix load单独做执行与覆盖校验；不把普通load校验成功推及LDSM。
4. NCU报告、原始导出和独立report-import结果保留。缺失/unsupported/N/A与有效0分开；动态执行证据为正但wavefront显示不可信的0，标记coverage unverified。
5. 每target最多两个基础设施重试；只重试工具/权限/身份失败，保留所有失败。不能因性能或计数方向不合预期重跑。NCU不能采集时，可继续独立性能比较，但相关机制问题保持未回答。

这个阶段冻结各架构实际可用的metric mapping，再执行正式profiling。指标名称尚未查询，当前draft列的是概念和候选metric family，不假装已验证可用。

## 阶段B：Gluon内布局性能对照与Triton复现核对

先做H100核心矩阵。所有候选先通过完整输出正确性和资源门，再归档binary，计时worker不导入Triton、不触发JIT。

主要输入使用已有有限quarter-multiple模式，oracle可以零容差验证max。另加预先固定的随机有限输入与max相关浮点边界正确性检查；边界比较以所用tl.max语义为准，保存NaN/有符号零等结果，不假设简单bit equality。随机/边界检查不替换主要计时输入。

计时设计：

- 主指标为full grid下完整kernel的耗时。所有候选使用同一数据、同一stream、同一worker配对。
- 每架构固定7次独立worker调用；记录UUID，允许物理GPU重复，不宣称7块独立GPU。
- 沿用此前exact-binary计时口径：每个scalar sample仅包含start event、一次完整kernel launch、end event、synchronize和elapsed_us。不把多个kernel的批次时间除以次数，也不引入CUDA Graph作为本轮主计时。
- 1/16 grid下同样使用单次launch event计时，作为规模诊断，不替代full grid主指标，不从中择优报告。
- 每个条件将四个Gluon候选的K个唯一物理binary按Williams/cyclic与反向顺序平衡，固定2K轮、每轮每binary 5个samples。每调用每binary共10K个样本，K≤4。warmup、初始化与SHA检查不进入samples。
- 四候选的主比较在Gluon内进行；Triton锚点/额外复现身份使用单独登记的对照和顺序，不混入四候选的最优声明。若需额外复现身份，在执行前冻结该对照的样本预算和顺序。
- kernel输入/输出不因计时顺序改变数学工作。正确性、module/ABI/CUBIN SHA和实际grid绑定保留。

报告每种布局的原始时间、比值、全部worker结果和资源，不合成总分。实际最优有不确定性时给出“仍可能最优的候选集合”，不强排一个赢家。

统计草案：按worker的配对log-time ratio形成区间；同条件内所有候选比较使用95% simultaneous区间（Bonferroni/t，df=6），3%为预声明的实用差异。报告t区间的分布/worker变化假设，不能把launch数当独立设备数。若某候选相对其他所有候选的耗时优势下界都超过3%，才声明明确更快；否则保留不确定/差异小的候选。跨条件/架构不作全局显著性或普遍非回归声明。完整原始时间排名也同时给出，区分观测最低与已明确更快。

## 阶段C：NCU与受控机制干预

以BF16/max、两个核心几何、host/device两路径为主要机制对象。完整kernel是主要实验对象；不先把它拆成六项或用不同kernel的时间相减。

### C1：固定vec，改变同列线程分布

32×128/w4的布局如下，二元组按[M,N]、N优先：

| 布局 | sizePerThread | threadsPerWarp | warpsPerCTA | 同列贡献线程 | 跨warp数 |
|---|---|---|---|---:|---:|
| vec8，N-first常规分配 | [1,8] | [2,16] | [4,1] | 8 | 4 |
| vec8，放在一个warp中 | [1,8] | [8,4] | [1,4] | 8 | 1 |
| vec2，N-first常规分配 | [1,2] | [1,32] | [2,2] | 2 | 2 |
| vec2，放在一个warp中 | [1,2] | [2,16] | [1,4] | 2 | 1 |

每线程仍持有32个元素。两组保持vec和同列贡献线程总数不变，改变这些线程所在的lane/warp。32×64/w8的对应布局在draft_protocol.json中列出，每线程8个元素。

这项干预也可能改变加载、输出转换和资源；因此同时监测这些项，避免把全部耗时差直接归因跨warp通信。

### C2：后续loader干预的登记条件

本轮C1同样使用Gluon BlockedLayout，沿用未修改compiler的实际loader。第一轮机制矩阵为两个几何、host/device两路径、四种布局，共16个逻辑身份；与阶段B相同的binary复用。

只有NCU和最终代码生成证据指向loader选择，才另行登记保持完整逻辑布局而改变初始加载策略的下一轮干预。Gluon BlockedLayout指定元素归属，并不保证某个LDS/LDSM opcode；不能把改变layout或shared swizzle冒充保持布局不变的纯opcode干预。

若后续确需诊断compiler patch，先形成可审查实现、限定被干预的初始LocalLoad并验证generic fallback、数据/线程参与/同步语义，再冻结新增矩阵。lowering只能检查自身op，跨op定位放在独立分析/变换中。不能顺带改掉规约或输出的shared load，不能把不能实现的cell改写成等价替代。这不是本轮布局实验的前置工作，也尚未进入执行协议。

### NCU记录

每唯一可运行CUBIN固定3次profile launch，和独立计时分开。warmup与输入初始化在profile region之外；使用普通单kernel launch。独立计时和profiling的同一CUBIN、ABI、参数和full grid都有SHA绑定。

主要指标：

- 动态shared-load/shared-store/matrix-load指令、active/predication与请求数。
- shared load/store的实际、ideal及excessive wavefront；bank-conflict计数与计数覆盖。
- 动态shuffle、combine、转换与barrier指令；相关stall/issue及pipeline利用率只作为辅助证据。
- 寄存器、SMEM、spill/local memory、实际occupancy；global/TMA相关流量用于检查是否存在共同变化。
- 通过SASS PC/source mapping分别标注初始tile加载、规约临时shared访问、输出转换；不能可靠归属的计数保持unattributed，不按总量硬分。

硬件计数pass与software/source instrumentation pass分开标记；保留replay/cache/clock设置、passes和collection方法。主profile请求cache-control=none、clock-control=none以贴近warm运行，所有实际设置记录并经可用性检查；profiler的重放时间不进入正式性能统计。

对相同条件按CTA数、有效数据字节或已测动态warp执行规范化，同时给原始总量。wavefront/request、conflict/request与调度数据联合阅读，不把wavefront增加都叫bank冲突，也不把计数较少直接叫更快。

如仍无法解释差异，再单独登记loading-only或reduction-only的有效Triton/Gluon诊断kernel。它们会改变优化、数据依赖和资源，只用于交叉核对，不直接测“原kernel的纯阶段耗时”，不把分项简单相加。

## 阶段D：跨架构复现、形状边界与改动决策

先在B200/SM100与RTX PRO 6000/SM120重复核心性能矩阵；C1显式布局干预和NCU按阶段A的架构覆盖执行，缺失字段保持未回答。C2只有另行登记之后才可执行。SMEM超限、编译失败、正确性失败均单列，不静默删除或填0收益。

若获得可重复现象，再冻结下一轮形状/warp范围，按结构选case，做新的边界验证。开发阶段可使用已有几何，但不能再把其当作独立confirmatory held-out。后续规则与新验证矩阵在看新结果前分别冻结。

最终应交付性能曲线/候选集合、NCU访问与通信证据、反例和可用范围，以及一个具体编译器改动方向，例如：

- 某种同列线程分布减少实际临时访存/同步且完整kernel更快，可研究相应ownership/降低转换的变换。
- 同一布局下generic与matrix加载存在明确差异，可研究LocalLoad指令选择或布局与该选择的交互。
- 默认布局已经更好或变化主要来自输出/descriptor路径，就围绕对应路径继续定位，不强改vec。
- 小差异/未能解释/跨架构不一致，保留结果并缩小范围，暂不形成默认全TMA策略。

这些是研究分支，不是已经验证的机制或预先指定的编译规则。成功标准是定位真实可重复的性能机会并形成可审查改动依据，而不是得到某个vec的预期结论。

H100/SM90、B200/SM100、SM120不能替代原GB300/SM103。若无法取得SM103，本轮只报告相应架构上的复现/优化机会，不声称原GB300回归已经解决。

## 证据、执行与提交

各阶段保存proposal/protocol、请求、source snapshot与SHA、dispatch、tool/driver/UUID、环境、原始samples或ncu-rep、compiler artifacts、失败记录和derived分析。新原始证据写入新的phase11目录且write-once；既有phase1–10资料不可变。PPT和discussion材料本轮不更新。

每阶段完成并通过相应validator后单独commit；实验分支仍为`explore/tma-reduction-layout`。是否push按已有实验流程处理，禁止创建/更新PR；提交PR前仍须询问用户。Modal执行使用`chenmiaoming`，复用ccache与现有镜像；额度或profile失败时报告用户，不自动切换账户。当前仅形成可review方案，stage A的metric mapping、适配后的Gluon/Triton源码、确切baseline native与worker SHA还需在执行前冻结。

参考：

- 原回归报告：https://github.com/triton-lang/triton/issues/9703
- NCU Shared Memory计数定义：https://docs.nvidia.com/nsight-compute/ProfilingGuide/#shared-memory
- NCU Source/Replay/Graph profiling限制：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 同期源码NVIDIA LocalLoad lowering：https://github.com/triton-lang/triton/blob/20d9d98fead78c2f65e1b3ba5a7881d86a836dad/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp
